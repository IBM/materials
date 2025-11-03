from __future__ import absolute_import, division, print_function

import inspect
import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from ase.data import covalent_radii
from torch import Tensor
from torch import nn as nn
from torch.nn.init import constant_
from torch_geometric.nn import MessagePassing

from ....utils.graph import scatter

zeros_initializer = partial(constant_, val=0.0)


def get_split_sizes_from_lmax(lmax: int, start: int = 1) -> List[int]:
    return [2 * l + 1 for l in range(start, lmax + 1)]  # noqa: E741


def split_to_components(tensor: Tensor, lmax: int, start: int = 1, dim: int = -1) -> List[Tensor]:
    split_sizes = get_split_sizes_from_lmax(lmax, start=start)
    components = torch.split(tensor, split_sizes, dim=dim)
    return components


class PolynomialCutoff(nn.Module):
    """
    Polynomial cutoff function, as proposed in DimeNet.

    Smoothly reduces values to zero based on a cutoff radius using a polynomial decay.
    Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float): Cutoff radius.
        p (int, optional): Exponent for the polynomial decay. Defaults to 6.
    """

    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(r: Tensor, rcut: float, p: float = 6.0) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        out = out * (rscaled < 1.0).float()
        return out.clamp(min=0.0)

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=float(self.p))

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay (0th order Bessel functions).

    Used in DimeNet. Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float, optional): Radial cutoff distance. Defaults to 5.0.
        n_rbf (int): Number of basis functions.
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-6.
        trainable (bool, optional): Whether the Bessel weights are trainable. Defaults to False.
    """

    def __init__(self, cutoff: float = 5.0, n_rbf: int = None, eps: float = 1e-6):
        super(BesselBasis, self).__init__()
        self.n_rbf = n_rbf
        self.eps = eps

        freqs = math.pi * torch.arange(1, n_rbf + 1, dtype=torch.get_default_dtype()) / cutoff
        self.register_buffer("freqs", freqs)

        self.register_buffer("prefactor", torch.tensor(math.sqrt(2 / cutoff), dtype=torch.get_default_dtype()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.unsqueeze(-1)
        numerator = torch.sin(self.freqs * x)
        return self.prefactor * (numerator / (x + self.eps))


def get_weight_init_by_string(init_str):
    """
    Get a weight initialization function based on its string name.

    Args:
        init_str (str): Name of the initialization method (e.g., 'zeros', 'xavier_uniform').

    Returns:
        Callable: The corresponding weight initialization function.

    Raises:
        ValueError: If the initialization string is unknown.
    """
    if init_str == "":
        # No-op
        return lambda x: x
    elif init_str == "zeros":
        return torch.nn.init.zeros_
    elif init_str == "uniform":
        return torch.nn.init.uniform_
    elif init_str == "xavier_uniform":
        return torch.nn.init.xavier_uniform_
    else:
        raise ValueError(f"Unknown initialization {init_str}")


class Dense(nn.Linear):
    """
    Fully connected linear layer with optional activation and normalization.
    Applies a linear transformation followed by optional normalization and activation.
    Borrowed from https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If False, the layer will not adapt bias. Defaults to True.
        activation (callable, optional): Activation function. If None, no activation is used. Defaults to None.
        weight_init (callable, optional): Weight initializer. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initializer. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or None). Defaults to None.
        gain (float, optional): Gain for weight initialization if applicable. Defaults to None.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        activation=None,
        weight_init=nn.init.xavier_uniform_,
        bias_init=zeros_initializer,
        norm=None,
        gain=None,
    ):
        # initialize linear layer y = xW^T + b
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super(Dense, self).__init__(in_features, out_features, bias)
        # Initialize activation function
        if inspect.isclass(activation):
            self.activation = activation()
        self.activation = activation

        if norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = F.linear(inputs, self.weight, self.bias)
        if self.norm is not None:
            y = self.norm(y)
        # add activation function
        if self.activation is not None:
            y = self.activation(y)
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden dimensions and activations.

    Args:
        hidden_dims (List[int]): List defining the dimensions of each layer,
            including input and output (e.g., [in_dim, hid1_dim, ..., out_dim]).
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        activation (callable, optional): Activation function for hidden layers. Defaults to None.
        last_activation (callable, optional): Activation function for the output layer. Defaults to None.
        weight_init (callable, optional): Weight initialization function. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initialization function. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or ''). Defaults to ''.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        bias=False,
        activation=None,
        last_activation=None,
        weight_init=nn.init.xavier_uniform_,
        bias_init=zeros_initializer,
        norm="",
    ):
        super().__init__()

        # hidden_dims = [hidden, half, hidden]

        dims = hidden_dims
        n_layers = len(dims)

        DenseMLP = partial(Dense, bias=bias, weight_init=weight_init, bias_init=bias_init)

        self.dense_layers = nn.ModuleList(
            [DenseMLP(dims[i], dims[i + 1], activation=activation, norm=norm) for i in range(n_layers - 2)]
            + [DenseMLP(dims[-2], dims[-1], activation=last_activation)]
        )

        self.layers = nn.Sequential(*self.dense_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x):
        return self.layers(x)


class TensorLayerNorm(nn.Module):
    def __init__(self, lmax: int, num_channels: int, eps: float = 1e-12):
        super().__init__()
        self.lmax = lmax
        self.eps = eps
        self.split_sizes = [1] + get_split_sizes_from_lmax(lmax)
        self.D = sum(self.split_sizes)
        self.C = num_channels

        deg_w = torch.zeros(self.D, 1)
        off = 0
        for size in self.split_sizes:
            deg_w[off : off + size, 0] = 1.0 / size
            off += size
        self.register_buffer("balance_degree_weight", deg_w)

        self.affine_weights = nn.ParameterList([nn.Parameter(torch.ones(num_channels)) for _ in range(lmax + 1)])

    def forward(self, h: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([h, X], dim=1)
        w = self.balance_degree_weight.view(1, self.D, 1)
        norm2 = (x.pow(2) * w).sum(dim=1, keepdim=True)
        inv_rms = (norm2.mean(dim=2, keepdim=True) + self.eps).pow(-0.5)

        weights_list = []
        for i, w_l in enumerate(self.affine_weights):
            weights_list.append(w_l.view(1, self.C).repeat(self.split_sizes[i], 1))
        affine_W = torch.cat(weights_list, dim=0).unsqueeze(0)
        out = x * inv_rms * affine_W
        return out[:, 0, :].unsqueeze(1), out[:, 1:, :]

    def reset_parameters(self):
        for p in self.affine_weights:
            nn.init.ones_(p)


def normalize_string(s: str) -> str:
    """
    Normalize a string by converting to lowercase and removing dashes, underscores, and spaces.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string.
    """
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def get_activations(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions.

    Args:
        optional (bool, optional): If True, include an empty string key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "mish": torch.nn.Mish,
            "swish": torch.nn.SiLU,
            "selu": torch.nn.SELU,
        }
    )

    if optional:
        activations[""] = None

    return activations


def get_activations_none(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions,
    excluding softplus-based activations.

    Args:
        optional (bool, optional): If True, include an empty string and None key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "selu": torch.nn.SELU,
        }
    )

    if optional:
        activations[""] = None
        activations[None] = None

    return activations


def dictionary_to_option(options, selected):
    """
    Select an option from a dictionary based on a key, handling potential class instantiation.

    Args:
        options (Dict): Dictionary of options (e.g., activation functions).
        selected (Optional[str]): The key of the selected option.

    Returns:
        Optional[Callable]: The selected option (possibly instantiated if it's a class).

    Raises:
        ValueError: If the selected key is not in the options dictionary.
    """
    if selected not in options:
        raise ValueError(f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))}')

    activation = options[selected]
    if inspect.isclass(activation):
        activation = activation()
    return activation


def str2act(input_str, *args, **kwargs):
    """
    Convert an activation function name string to the corresponding function/class instance.

    Args:
        input_str (Optional[str]): Name of the activation function (case-insensitive, ignores '-', '_', ' ').
                                   If None or "", returns None.

    Returns:
        Optional[Callable]: The instantiated activation function or None.
    """
    if not input_str:  # Handles None and ""
        return None

    act = get_activations(*args, optional=True, **kwargs)
    out = dictionary_to_option(act, input_str)
    return out


def str2basis(input_str):
    """
    Convert a radial basis function name string to the corresponding class.

    Args:
        input_str (Union[str, Callable]): Name of the basis function ('BesselBasis', 'GaussianRBF', 'expnorm')
                                          or already a callable class.

    Returns:
        Callable: The radial basis function class.

    Raises:
        ValueError: If the input string is unknown.
    """
    if not isinstance(input_str, str):
        return input_str  # Assume it's already a callable class

    normalized_input = normalize_string(input_str)

    if normalized_input == "besselbasis":
        radial_basis = BesselBasis
    else:
        raise ValueError("Unknown radial basis: {}".format(input_str))

    return radial_basis


class TensorInit(nn.Module):
    """
    Initializes high-degree tensor representations based on edge vectors using spherical harmonics.

    Computes real spherical harmonic components up to degree `l` from Cartesian edge vectors.

    Args:
        l (int, optional): Maximum degree (lmax) of spherical harmonics to compute. Defaults to 2.
    """

    def __init__(self, l=2):  # noqa: E741
        super(TensorInit, self).__init__()
        self.l = l

    def forward(self, edge_vec):
        edge_sh = self._calculate_components(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh

    @property
    def tensor_size(self):
        return ((self.l + 1) ** 2) - 1

    @staticmethod
    def _calculate_components(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate Cartesian spherical harmonic components up to lmax.

        Based on explicit formulas (potentially derived or adapted from sources like e3nn).
        This implementation calculates components excluding the l=0 (scalar) part.

        Args:
            lmax (int): Maximum degree to compute.
            x (Tensor): x-components of vectors. Shape: [num_edges]
            y (Tensor): y-components of vectors. Shape: [num_edges]
            z (Tensor): z-components of vectors. Shape: [num_edges]

        Returns:
            Tensor: Stacked spherical harmonic components. Shape: [num_edges, (lmax+1)**2 - 1]
        """
        # l=1 components (proportional to x, y, z)
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        # (x^2, y^2, z^2) ^2

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)

        # Borrowed from e3nn: https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_spherical_harmonics.py#L188
        sh_3_0 = (1 / 6) * math.sqrt(42) * (sh_2_0 * z + sh_2_4 * x)
        sh_3_1 = math.sqrt(7) * sh_2_0 * y
        sh_3_2 = (1 / 8) * math.sqrt(168) * (4.0 * y2 - x2z2) * x
        sh_3_3 = (1 / 2) * math.sqrt(7) * y * (2.0 * y2 - 3.0 * x2z2)
        sh_3_4 = (1 / 8) * math.sqrt(168) * z * (4.0 * y2 - x2z2)
        sh_3_5 = math.sqrt(7) * sh_2_4 * y
        sh_3_6 = (1 / 6) * math.sqrt(42) * (sh_2_4 * z - sh_2_0 * x)

        if lmax == 3:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                ],
                dim=-1,
            )

        sh_4_0 = (3 / 4) * math.sqrt(2) * (sh_3_0 * z + sh_3_6 * x)
        sh_4_1 = (3 / 4) * sh_3_0 * y + (3 / 8) * math.sqrt(6) * sh_3_1 * z + (3 / 8) * math.sqrt(6) * sh_3_5 * x
        sh_4_2 = (
            -3 / 56 * math.sqrt(14) * sh_3_0 * z
            + (3 / 14) * math.sqrt(21) * sh_3_1 * y
            + (3 / 56) * math.sqrt(210) * sh_3_2 * z
            + (3 / 56) * math.sqrt(210) * sh_3_4 * x
            + (3 / 56) * math.sqrt(14) * sh_3_6 * x
        )
        sh_4_3 = (
            -3 / 56 * math.sqrt(42) * sh_3_1 * z
            + (3 / 28) * math.sqrt(105) * sh_3_2 * y
            + (3 / 28) * math.sqrt(70) * sh_3_3 * x
            + (3 / 56) * math.sqrt(42) * sh_3_5 * x
        )
        sh_4_4 = -3 / 28 * math.sqrt(42) * sh_3_2 * x + (3 / 7) * math.sqrt(7) * sh_3_3 * y - 3 / 28 * math.sqrt(42) * sh_3_4 * z
        sh_4_5 = (
            -3 / 56 * math.sqrt(42) * sh_3_1 * x
            + (3 / 28) * math.sqrt(70) * sh_3_3 * z
            + (3 / 28) * math.sqrt(105) * sh_3_4 * y
            - 3 / 56 * math.sqrt(42) * sh_3_5 * z
        )
        sh_4_6 = (
            -3 / 56 * math.sqrt(14) * sh_3_0 * x
            - 3 / 56 * math.sqrt(210) * sh_3_2 * x
            + (3 / 56) * math.sqrt(210) * sh_3_4 * z
            + (3 / 14) * math.sqrt(21) * sh_3_5 * y
            - 3 / 56 * math.sqrt(14) * sh_3_6 * z
        )
        sh_4_7 = -3 / 8 * math.sqrt(6) * sh_3_1 * x + (3 / 8) * math.sqrt(6) * sh_3_5 * z + (3 / 4) * sh_3_6 * y
        sh_4_8 = (3 / 4) * math.sqrt(2) * (-sh_3_0 * x + sh_3_6 * z)
        if lmax == 4:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                ],
                dim=-1,
            )

        sh_5_0 = (1 / 10) * math.sqrt(110) * (sh_4_0 * z + sh_4_8 * x)
        sh_5_1 = (1 / 5) * math.sqrt(11) * sh_4_0 * y + (1 / 5) * math.sqrt(22) * sh_4_1 * z + (1 / 5) * math.sqrt(22) * sh_4_7 * x
        sh_5_2 = (
            -1 / 30 * math.sqrt(22) * sh_4_0 * z
            + (4 / 15) * math.sqrt(11) * sh_4_1 * y
            + (1 / 15) * math.sqrt(154) * sh_4_2 * z
            + (1 / 15) * math.sqrt(154) * sh_4_6 * x
            + (1 / 30) * math.sqrt(22) * sh_4_8 * x
        )
        sh_5_3 = (
            -1 / 30 * math.sqrt(66) * sh_4_1 * z
            + (1 / 15) * math.sqrt(231) * sh_4_2 * y
            + (1 / 30) * math.sqrt(462) * sh_4_3 * z
            + (1 / 30) * math.sqrt(462) * sh_4_5 * x
            + (1 / 30) * math.sqrt(66) * sh_4_7 * x
        )
        sh_5_4 = (
            -1 / 15 * math.sqrt(33) * sh_4_2 * z
            + (2 / 15) * math.sqrt(66) * sh_4_3 * y
            + (1 / 15) * math.sqrt(165) * sh_4_4 * x
            + (1 / 15) * math.sqrt(33) * sh_4_6 * x
        )
        sh_5_5 = -1 / 15 * math.sqrt(110) * sh_4_3 * x + (1 / 3) * math.sqrt(11) * sh_4_4 * y - 1 / 15 * math.sqrt(110) * sh_4_5 * z
        sh_5_6 = (
            -1 / 15 * math.sqrt(33) * sh_4_2 * x
            + (1 / 15) * math.sqrt(165) * sh_4_4 * z
            + (2 / 15) * math.sqrt(66) * sh_4_5 * y
            - 1 / 15 * math.sqrt(33) * sh_4_6 * z
        )
        sh_5_7 = (
            -1 / 30 * math.sqrt(66) * sh_4_1 * x
            - 1 / 30 * math.sqrt(462) * sh_4_3 * x
            + (1 / 30) * math.sqrt(462) * sh_4_5 * z
            + (1 / 15) * math.sqrt(231) * sh_4_6 * y
            - 1 / 30 * math.sqrt(66) * sh_4_7 * z
        )
        sh_5_8 = (
            -1 / 30 * math.sqrt(22) * sh_4_0 * x
            - 1 / 15 * math.sqrt(154) * sh_4_2 * x
            + (1 / 15) * math.sqrt(154) * sh_4_6 * z
            + (4 / 15) * math.sqrt(11) * sh_4_7 * y
            - 1 / 30 * math.sqrt(22) * sh_4_8 * z
        )
        sh_5_9 = -1 / 5 * math.sqrt(22) * sh_4_1 * x + (1 / 5) * math.sqrt(22) * sh_4_7 * z + (1 / 5) * math.sqrt(11) * sh_4_8 * y
        sh_5_10 = (1 / 10) * math.sqrt(110) * (-sh_4_0 * x + sh_4_8 * z)
        if lmax == 5:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                ],
                dim=-1,
            )

        sh_6_0 = (1 / 6) * math.sqrt(39) * (sh_5_0 * z + sh_5_10 * x)
        sh_6_1 = (1 / 6) * math.sqrt(13) * sh_5_0 * y + (1 / 12) * math.sqrt(130) * sh_5_1 * z + (1 / 12) * math.sqrt(130) * sh_5_9 * x
        sh_6_2 = (
            -1 / 132 * math.sqrt(286) * sh_5_0 * z
            + (1 / 33) * math.sqrt(715) * sh_5_1 * y
            + (1 / 132) * math.sqrt(286) * sh_5_10 * x
            + (1 / 44) * math.sqrt(1430) * sh_5_2 * z
            + (1 / 44) * math.sqrt(1430) * sh_5_8 * x
        )
        sh_6_3 = (
            -1 / 132 * math.sqrt(858) * sh_5_1 * z
            + (1 / 22) * math.sqrt(429) * sh_5_2 * y
            + (1 / 22) * math.sqrt(286) * sh_5_3 * z
            + (1 / 22) * math.sqrt(286) * sh_5_7 * x
            + (1 / 132) * math.sqrt(858) * sh_5_9 * x
        )
        sh_6_4 = (
            -1 / 66 * math.sqrt(429) * sh_5_2 * z
            + (2 / 33) * math.sqrt(286) * sh_5_3 * y
            + (1 / 66) * math.sqrt(2002) * sh_5_4 * z
            + (1 / 66) * math.sqrt(2002) * sh_5_6 * x
            + (1 / 66) * math.sqrt(429) * sh_5_8 * x
        )
        sh_6_5 = (
            -1 / 66 * math.sqrt(715) * sh_5_3 * z
            + (1 / 66) * math.sqrt(5005) * sh_5_4 * y
            + (1 / 66) * math.sqrt(3003) * sh_5_5 * x
            + (1 / 66) * math.sqrt(715) * sh_5_7 * x
        )
        sh_6_6 = -1 / 66 * math.sqrt(2145) * sh_5_4 * x + (1 / 11) * math.sqrt(143) * sh_5_5 * y - 1 / 66 * math.sqrt(2145) * sh_5_6 * z
        sh_6_7 = (
            -1 / 66 * math.sqrt(715) * sh_5_3 * x
            + (1 / 66) * math.sqrt(3003) * sh_5_5 * z
            + (1 / 66) * math.sqrt(5005) * sh_5_6 * y
            - 1 / 66 * math.sqrt(715) * sh_5_7 * z
        )
        sh_6_8 = (
            -1 / 66 * math.sqrt(429) * sh_5_2 * x
            - 1 / 66 * math.sqrt(2002) * sh_5_4 * x
            + (1 / 66) * math.sqrt(2002) * sh_5_6 * z
            + (2 / 33) * math.sqrt(286) * sh_5_7 * y
            - 1 / 66 * math.sqrt(429) * sh_5_8 * z
        )
        sh_6_9 = (
            -1 / 132 * math.sqrt(858) * sh_5_1 * x
            - 1 / 22 * math.sqrt(286) * sh_5_3 * x
            + (1 / 22) * math.sqrt(286) * sh_5_7 * z
            + (1 / 22) * math.sqrt(429) * sh_5_8 * y
            - 1 / 132 * math.sqrt(858) * sh_5_9 * z
        )
        sh_6_10 = (
            -1 / 132 * math.sqrt(286) * sh_5_0 * x
            - 1 / 132 * math.sqrt(286) * sh_5_10 * z
            - 1 / 44 * math.sqrt(1430) * sh_5_2 * x
            + (1 / 44) * math.sqrt(1430) * sh_5_8 * z
            + (1 / 33) * math.sqrt(715) * sh_5_9 * y
        )
        sh_6_11 = -1 / 12 * math.sqrt(130) * sh_5_1 * x + (1 / 6) * math.sqrt(13) * sh_5_10 * y + (1 / 12) * math.sqrt(130) * sh_5_9 * z
        sh_6_12 = (1 / 6) * math.sqrt(39) * (-sh_5_0 * x + sh_5_10 * z)
        if lmax == 6:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                    sh_6_0,
                    sh_6_1,
                    sh_6_2,
                    sh_6_3,
                    sh_6_4,
                    sh_6_5,
                    sh_6_6,
                    sh_6_7,
                    sh_6_8,
                    sh_6_9,
                    sh_6_10,
                    sh_6_11,
                    sh_6_12,
                ],
                dim=-1,
            )

        sh_7_0 = (1 / 14) * math.sqrt(210) * (sh_6_0 * z + sh_6_12 * x)
        sh_7_1 = (1 / 7) * math.sqrt(15) * sh_6_0 * y + (3 / 7) * math.sqrt(5) * sh_6_1 * z + (3 / 7) * math.sqrt(5) * sh_6_11 * x
        sh_7_2 = (
            -1 / 182 * math.sqrt(390) * sh_6_0 * z
            + (6 / 91) * math.sqrt(130) * sh_6_1 * y
            + (3 / 91) * math.sqrt(715) * sh_6_10 * x
            + (1 / 182) * math.sqrt(390) * sh_6_12 * x
            + (3 / 91) * math.sqrt(715) * sh_6_2 * z
        )
        sh_7_3 = (
            -3 / 182 * math.sqrt(130) * sh_6_1 * z
            + (3 / 182) * math.sqrt(130) * sh_6_11 * x
            + (3 / 91) * math.sqrt(715) * sh_6_2 * y
            + (5 / 182) * math.sqrt(858) * sh_6_3 * z
            + (5 / 182) * math.sqrt(858) * sh_6_9 * x
        )
        sh_7_4 = (
            (3 / 91) * math.sqrt(65) * sh_6_10 * x
            - 3 / 91 * math.sqrt(65) * sh_6_2 * z
            + (10 / 91) * math.sqrt(78) * sh_6_3 * y
            + (15 / 182) * math.sqrt(78) * sh_6_4 * z
            + (15 / 182) * math.sqrt(78) * sh_6_8 * x
        )
        sh_7_5 = (
            -5 / 91 * math.sqrt(39) * sh_6_3 * z
            + (15 / 91) * math.sqrt(39) * sh_6_4 * y
            + (3 / 91) * math.sqrt(390) * sh_6_5 * z
            + (3 / 91) * math.sqrt(390) * sh_6_7 * x
            + (5 / 91) * math.sqrt(39) * sh_6_9 * x
        )
        sh_7_6 = (
            -15 / 182 * math.sqrt(26) * sh_6_4 * z
            + (12 / 91) * math.sqrt(65) * sh_6_5 * y
            + (2 / 91) * math.sqrt(1365) * sh_6_6 * x
            + (15 / 182) * math.sqrt(26) * sh_6_8 * x
        )
        sh_7_7 = -3 / 91 * math.sqrt(455) * sh_6_5 * x + (1 / 13) * math.sqrt(195) * sh_6_6 * y - 3 / 91 * math.sqrt(455) * sh_6_7 * z
        sh_7_8 = (
            -15 / 182 * math.sqrt(26) * sh_6_4 * x
            + (2 / 91) * math.sqrt(1365) * sh_6_6 * z
            + (12 / 91) * math.sqrt(65) * sh_6_7 * y
            - 15 / 182 * math.sqrt(26) * sh_6_8 * z
        )
        sh_7_9 = (
            -5 / 91 * math.sqrt(39) * sh_6_3 * x
            - 3 / 91 * math.sqrt(390) * sh_6_5 * x
            + (3 / 91) * math.sqrt(390) * sh_6_7 * z
            + (15 / 91) * math.sqrt(39) * sh_6_8 * y
            - 5 / 91 * math.sqrt(39) * sh_6_9 * z
        )
        sh_7_10 = (
            -3 / 91 * math.sqrt(65) * sh_6_10 * z
            - 3 / 91 * math.sqrt(65) * sh_6_2 * x
            - 15 / 182 * math.sqrt(78) * sh_6_4 * x
            + (15 / 182) * math.sqrt(78) * sh_6_8 * z
            + (10 / 91) * math.sqrt(78) * sh_6_9 * y
        )
        sh_7_11 = (
            -3 / 182 * math.sqrt(130) * sh_6_1 * x
            + (3 / 91) * math.sqrt(715) * sh_6_10 * y
            - 3 / 182 * math.sqrt(130) * sh_6_11 * z
            - 5 / 182 * math.sqrt(858) * sh_6_3 * x
            + (5 / 182) * math.sqrt(858) * sh_6_9 * z
        )
        sh_7_12 = (
            -1 / 182 * math.sqrt(390) * sh_6_0 * x
            + (3 / 91) * math.sqrt(715) * sh_6_10 * z
            + (6 / 91) * math.sqrt(130) * sh_6_11 * y
            - 1 / 182 * math.sqrt(390) * sh_6_12 * z
            - 3 / 91 * math.sqrt(715) * sh_6_2 * x
        )
        sh_7_13 = -3 / 7 * math.sqrt(5) * sh_6_1 * x + (3 / 7) * math.sqrt(5) * sh_6_11 * z + (1 / 7) * math.sqrt(15) * sh_6_12 * y
        sh_7_14 = (1 / 14) * math.sqrt(210) * (-sh_6_0 * x + sh_6_12 * z)
        if lmax == 7:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                    sh_6_0,
                    sh_6_1,
                    sh_6_2,
                    sh_6_3,
                    sh_6_4,
                    sh_6_5,
                    sh_6_6,
                    sh_6_7,
                    sh_6_8,
                    sh_6_9,
                    sh_6_10,
                    sh_6_11,
                    sh_6_12,
                    sh_7_0,
                    sh_7_1,
                    sh_7_2,
                    sh_7_3,
                    sh_7_4,
                    sh_7_5,
                    sh_7_6,
                    sh_7_7,
                    sh_7_8,
                    sh_7_9,
                    sh_7_10,
                    sh_7_11,
                    sh_7_12,
                    sh_7_13,
                    sh_7_14,
                ],
                dim=-1,
            )

        sh_8_0 = (1 / 4) * math.sqrt(17) * (sh_7_0 * z + sh_7_14 * x)
        sh_8_1 = (1 / 8) * math.sqrt(17) * sh_7_0 * y + (1 / 16) * math.sqrt(238) * sh_7_1 * z + (1 / 16) * math.sqrt(238) * sh_7_13 * x
        sh_8_2 = (
            -1 / 240 * math.sqrt(510) * sh_7_0 * z
            + (1 / 60) * math.sqrt(1785) * sh_7_1 * y
            + (1 / 240) * math.sqrt(46410) * sh_7_12 * x
            + (1 / 240) * math.sqrt(510) * sh_7_14 * x
            + (1 / 240) * math.sqrt(46410) * sh_7_2 * z
        )
        sh_8_3 = (
            (1 / 80)
            * math.sqrt(2)
            * (
                -math.sqrt(85) * sh_7_1 * z
                + math.sqrt(2210) * sh_7_11 * x
                + math.sqrt(85) * sh_7_13 * x
                + math.sqrt(2210) * sh_7_2 * y
                + math.sqrt(2210) * sh_7_3 * z
            )
        )
        sh_8_4 = (
            (1 / 40) * math.sqrt(935) * sh_7_10 * x
            + (1 / 40) * math.sqrt(85) * sh_7_12 * x
            - 1 / 40 * math.sqrt(85) * sh_7_2 * z
            + (1 / 10) * math.sqrt(85) * sh_7_3 * y
            + (1 / 40) * math.sqrt(935) * sh_7_4 * z
        )
        sh_8_5 = (
            (1 / 48)
            * math.sqrt(2)
            * (
                math.sqrt(102) * sh_7_11 * x
                - math.sqrt(102) * sh_7_3 * z
                + math.sqrt(1122) * sh_7_4 * y
                + math.sqrt(561) * sh_7_5 * z
                + math.sqrt(561) * sh_7_9 * x
            )
        )
        sh_8_6 = (
            (1 / 16) * math.sqrt(34) * sh_7_10 * x
            - 1 / 16 * math.sqrt(34) * sh_7_4 * z
            + (1 / 4) * math.sqrt(17) * sh_7_5 * y
            + (1 / 16) * math.sqrt(102) * sh_7_6 * z
            + (1 / 16) * math.sqrt(102) * sh_7_8 * x
        )
        sh_8_7 = (
            -1 / 80 * math.sqrt(1190) * sh_7_5 * z
            + (1 / 40) * math.sqrt(1785) * sh_7_6 * y
            + (1 / 20) * math.sqrt(255) * sh_7_7 * x
            + (1 / 80) * math.sqrt(1190) * sh_7_9 * x
        )
        sh_8_8 = -1 / 60 * math.sqrt(1785) * sh_7_6 * x + (1 / 15) * math.sqrt(255) * sh_7_7 * y - 1 / 60 * math.sqrt(1785) * sh_7_8 * z
        sh_8_9 = (
            -1 / 80 * math.sqrt(1190) * sh_7_5 * x
            + (1 / 20) * math.sqrt(255) * sh_7_7 * z
            + (1 / 40) * math.sqrt(1785) * sh_7_8 * y
            - 1 / 80 * math.sqrt(1190) * sh_7_9 * z
        )
        sh_8_10 = (
            -1 / 16 * math.sqrt(34) * sh_7_10 * z
            - 1 / 16 * math.sqrt(34) * sh_7_4 * x
            - 1 / 16 * math.sqrt(102) * sh_7_6 * x
            + (1 / 16) * math.sqrt(102) * sh_7_8 * z
            + (1 / 4) * math.sqrt(17) * sh_7_9 * y
        )
        sh_8_11 = (
            (1 / 48)
            * math.sqrt(2)
            * (
                math.sqrt(1122) * sh_7_10 * y
                - math.sqrt(102) * sh_7_11 * z
                - math.sqrt(102) * sh_7_3 * x
                - math.sqrt(561) * sh_7_5 * x
                + math.sqrt(561) * sh_7_9 * z
            )
        )
        sh_8_12 = (
            (1 / 40) * math.sqrt(935) * sh_7_10 * z
            + (1 / 10) * math.sqrt(85) * sh_7_11 * y
            - 1 / 40 * math.sqrt(85) * sh_7_12 * z
            - 1 / 40 * math.sqrt(85) * sh_7_2 * x
            - 1 / 40 * math.sqrt(935) * sh_7_4 * x
        )
        sh_8_13 = (
            (1 / 80)
            * math.sqrt(2)
            * (
                -math.sqrt(85) * sh_7_1 * x
                + math.sqrt(2210) * sh_7_11 * z
                + math.sqrt(2210) * sh_7_12 * y
                - math.sqrt(85) * sh_7_13 * z
                - math.sqrt(2210) * sh_7_3 * x
            )
        )
        sh_8_14 = (
            -1 / 240 * math.sqrt(510) * sh_7_0 * x
            + (1 / 240) * math.sqrt(46410) * sh_7_12 * z
            + (1 / 60) * math.sqrt(1785) * sh_7_13 * y
            - 1 / 240 * math.sqrt(510) * sh_7_14 * z
            - 1 / 240 * math.sqrt(46410) * sh_7_2 * x
        )
        sh_8_15 = -1 / 16 * math.sqrt(238) * sh_7_1 * x + (1 / 16) * math.sqrt(238) * sh_7_13 * z + (1 / 8) * math.sqrt(17) * sh_7_14 * y
        sh_8_16 = (1 / 4) * math.sqrt(17) * (-sh_7_0 * x + sh_7_14 * z)
        if lmax == 8:
            return torch.stack(
                [
                    sh_1_0,
                    sh_1_1,
                    sh_1_2,
                    sh_2_0,
                    sh_2_1,
                    sh_2_2,
                    sh_2_3,
                    sh_2_4,
                    sh_3_0,
                    sh_3_1,
                    sh_3_2,
                    sh_3_3,
                    sh_3_4,
                    sh_3_5,
                    sh_3_6,
                    sh_4_0,
                    sh_4_1,
                    sh_4_2,
                    sh_4_3,
                    sh_4_4,
                    sh_4_5,
                    sh_4_6,
                    sh_4_7,
                    sh_4_8,
                    sh_5_0,
                    sh_5_1,
                    sh_5_2,
                    sh_5_3,
                    sh_5_4,
                    sh_5_5,
                    sh_5_6,
                    sh_5_7,
                    sh_5_8,
                    sh_5_9,
                    sh_5_10,
                    sh_6_0,
                    sh_6_1,
                    sh_6_2,
                    sh_6_3,
                    sh_6_4,
                    sh_6_5,
                    sh_6_6,
                    sh_6_7,
                    sh_6_8,
                    sh_6_9,
                    sh_6_10,
                    sh_6_11,
                    sh_6_12,
                    sh_7_0,
                    sh_7_1,
                    sh_7_2,
                    sh_7_3,
                    sh_7_4,
                    sh_7_5,
                    sh_7_6,
                    sh_7_7,
                    sh_7_8,
                    sh_7_9,
                    sh_7_10,
                    sh_7_11,
                    sh_7_12,
                    sh_7_13,
                    sh_7_14,
                    sh_8_0,
                    sh_8_1,
                    sh_8_2,
                    sh_8_3,
                    sh_8_4,
                    sh_8_5,
                    sh_8_6,
                    sh_8_7,
                    sh_8_8,
                    sh_8_9,
                    sh_8_10,
                    sh_8_11,
                    sh_8_12,
                    sh_8_13,
                    sh_8_14,
                    sh_8_15,
                    sh_8_16,
                ],
                dim=-1,
            )
        return torch.tensor(0)


class NodeInit(MessagePassing):
    """
    Node initialization layer for message passing networks.

    Initializes scalar node features based on atom types and their local environment
    using message passing. Implements Eq. 1 and 2 from the GotenNet paper.

    Args:
        hidden_channels (Union[int, List[int]]): Dimension of hidden channels. If a list, defines MLP layers.
        num_rbf (int): Number of radial basis functions used for edge features.
        max_z (int, optional): Maximum atomic number for embedding lookup. Defaults to 100.
        activation (Callable, optional): Activation function. Defaults to F.silu.
        proj_ln (str, optional): Type of layer normalization for projection ('layer' or ''). Defaults to ''.
        weight_init (Callable, optional): Weight initialization function. Defaults to nn.init.xavier_uniform_.
        bias_init (Callable, optional): Bias initialization function. Defaults to nn.init.zeros_.
        bias (bool, optional): Whether to use biases in W_nrd_nru.
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        max_z=100,
        activation=F.silu,
        proj_ln="",
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        bias: bool = False,
    ):
        super(NodeInit, self).__init__(aggr="sum")
        if type(hidden_channels) == int:  # noqa: E721
            hidden_channels = [hidden_channels]

        last_channel = hidden_channels[-1]
        self.A_nbr = nn.Embedding(max_z, last_channel)
        self.W_ndp = MLP(
            [num_rbf] + [last_channel],
            activation=None,
            norm="",
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=None,
            bias=False,
        )

        self.W_nrd_nru = MLP(
            [2 * last_channel] + hidden_channels,
            activation=activation,
            norm=proj_ln,
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=None,
            bias=bias,
        )

        self.density_fn = Dense(
            num_rbf,
            1,
            bias=False,
            activation=activation,
            norm=None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.A_nbr.reset_parameters()
        self.W_ndp.reset_parameters()
        self.W_nrd_nru.reset_parameters()
        self.density_fn.reset_parameters()

    def forward(self, z: Tensor, h: Tensor, edge_index: Tensor, varphi_r0_ij: Tensor) -> Tensor:
        h_src = self.A_nbr(z)
        r0_ij_feat = self.W_ndp(varphi_r0_ij)

        # propagate_type: (h_src: Tensor, r0_ij_feat:Tensor, varphi_r0_ij:Tensor)
        m_i, density = self.propagate(edge_index=edge_index, h_src=h_src, r0_ij_feat=r0_ij_feat, varphi_r0_ij=varphi_r0_ij, size=None)
        m_i = m_i / (density + 1)
        return self.W_nrd_nru(torch.cat([h, m_i], dim=1))

    def message(self, h_src_j: Tensor, r0_ij_feat: Tensor, varphi_r0_ij: Tensor) -> Tuple[Tensor, Tensor]:
        density = torch.tanh(self.density_fn(varphi_r0_ij) ** 2)
        return h_src_j * r0_ij_feat, density

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        h, D = features
        h = scatter(h, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        D = scatter(D, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return h, D

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs


class EdgeInit(MessagePassing):
    """
    Edge initialization layer for message passing networks.

    Initializes scalar edge features based on connected node features and radial basis functions.
    Implements Eq. 3 from the GotenNet paper.

    Args:
        num_rbf (int): Number of radial basis functions.
        hidden_channels (int): Dimension of hidden channels (must match node features).
        activation (Callable, optional): Activation function (currently unused). Defaults to None.
    """

    def __init__(self, num_rbf, hidden_channels, activation=None):
        super(EdgeInit, self).__init__(aggr=None)
        self.W_erp = nn.Linear(num_rbf, hidden_channels, bias=False)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_erp.weight)

    def forward(self, edge_index, phi_r0_ij, h):
        # propagate_type: (h: Tensor, phi_r0_ij: Tensor)
        out = self.propagate(edge_index, h=h, phi_r0_ij=phi_r0_ij)
        return out

    def message(self, h_i, h_j, phi_r0_ij):
        return (h_i + h_j) * self.W_erp(phi_r0_ij)

    def aggregate(self, features, index):
        # no aggregate
        return features


class DistanceWeighting(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.register_buffer("covalent_radii", torch.tensor(covalent_radii, dtype=torch.get_default_dtype()))
        if trainable:
            self.q = nn.Parameter(torch.tensor(3.2139, requires_grad=True))
            self.p = nn.Parameter(torch.tensor(-15.2013, requires_grad=True))
        else:
            self.register_buffer("q", torch.tensor(3.2139, dtype=torch.get_default_dtype()))
            self.register_buffer("p", torch.tensor(-15.2013, dtype=torch.get_default_dtype()))

    def forward(self, z: torch.Tensor, edge_distance: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index[0], edge_index[1]
        r0 = self.covalent_radii[z[sender]] + self.covalent_radii[z[receiver]]
        t = edge_distance / r0
        p = F.softplus(self.p, 0.5) + 1
        q = F.softplus(self.q, 0.5) + 1
        a = -2 * (p + q - 2 * q * p) / (p**2 + p + q**2 + q)
        w = (a * t.pow(q)) / (1.0 + t.pow(q - p) + a * t.pow(q))
        return w


class IdentityDistanceWeighting(nn.Module):
    def forward(self, z: torch.Tensor, edge_distance: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return edge_distance
