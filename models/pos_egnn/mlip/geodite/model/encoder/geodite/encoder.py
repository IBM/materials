import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor

from geodite.utils import DataInput

from ....utils.graph import scatter
from .._base_encoder import AbstractEncoder
from .ops import (
    MLP,
    Dense,
    DistanceWeighting,
    EdgeInit,
    IdentityDistanceWeighting,
    NodeInit,
    PolynomialCutoff,
    TensorInit,
    TensorLayerNorm,
    get_split_sizes_from_lmax,
    get_weight_init_by_string,
    split_to_components,
    str2act,
    str2basis,
)


class Interaction(MessagePassing):
    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        n_rbf: int,
        weight_init: Callable = nn.init.xavier_uniform_,
        aggr: str = "add",
        node_dim: int = 0,
        epsilon: float = 1e-7,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        lmax: int = 2,
        edge_ln: str = "",
    ):
        """
        Graph Attention Transformer Architecture.

        Args:
            n_atom_basis: Number of features to describe atomic environments.
            activation: Activation function to be used. If None, no activation function is used.
            weight_init: Weight initialization function.
            aggr: Aggregation method ('add', 'mean' or 'max').
            node_dim: The axis along which to aggregate.
            epsilon: Small constant for numerical stability.
            cutoff: Cutoff distance for interactions.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            edge_updates: Whether to update edge features.
            last_layer: Whether this is the last layer.
            evec_dim: Dimension of edge vector features.
            emlp_dim: Dimension of edge MLP features.
            lmax: Maximum angular momentum.
        """
        super(Interaction, self).__init__(aggr=aggr, node_dim=node_dim)
        self.epsilon = epsilon
        self.edge_updates = edge_updates
        self.activation = activation
        self.n_rbf = n_rbf

        # Parse edge update configuration
        update_info_str = {"gated": ""}

        update_info_bool = {
            "rej": True,
            "mlp": False,
            "mlpa": False,
        }

        update_info_int = {
            "lin_w": 0,
            "lin_ln": 0,
        }

        update_parts = edge_updates.split("_") if isinstance(edge_updates, str) else []
        allowed_parts = ["gated", "gatedt", "norej", "norm", "mlp", "mlpa", "act", "linw", "linwa", "ln", "postln"]

        if not all([part in allowed_parts for part in update_parts]):
            raise ValueError(f"Invalid edge update parts. Allowed parts are {allowed_parts}")

        if "gated" in update_parts:
            update_info_str["gated"] = "gated"
        if "gatedt" in update_parts:
            update_info_str["gated"] = "gatedt"
        if "act" in update_parts:
            update_info_str["gated"] = "act"
        if "norej" in update_parts:
            update_info_bool["rej"] = False
        if "mlp" in update_parts:
            update_info_bool["mlp"] = True
        if "mlpa" in update_parts:
            update_info_bool["mlpa"] = True
        if "linw" in update_parts:
            update_info_int["lin_w"] = 1
        if "linwa" in update_parts:
            update_info_int["lin_w"] = 2
        if "ln" in update_parts:
            update_info_int["lin_ln"] = 1
        if "postln" in update_parts:
            update_info_int["lin_ln"] = 2

        self.update_info_str = update_info_str
        self.update_info_bool = update_info_bool
        self.update_info_int = update_info_int

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax

        multiplier = 3
        self.multiplier = multiplier

        # Initialize layers
        InitDense = partial(Dense, weight_init=weight_init, bias=False)

        # Implementation of gamma_s function
        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation, bias=False),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None, bias=False),
        )

        self.num_heads = num_heads

        # Query and key transformations
        self.W_q = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.W_k = InitDense(n_atom_basis, n_atom_basis, activation=None)

        # Value transformation
        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )

        # Edge feature transformations
        self.W_re = InitDense(
            n_atom_basis,
            n_atom_basis,
            activation=activation,
            bias=False,
        )

        # Initialize MLP for edge updates
        InitMLP = partial(MLP, weight_init=weight_init, bias=False)

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim

        if self.update_info_bool["mlp"] or self.update_info_bool["mlpa"]:
            dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
        else:
            dims = [n_atom_basis, n_atom_basis]

        self.gamma_t = InitMLP(
            dims, activation=activation, last_activation=None if self.update_info_bool["mlp"] else self.activation, norm=edge_ln, bias=False
        )

        self.W_vq = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

        self.W_vk = nn.ModuleList([InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False) for _i in range(self.lmax)])

        modules = []
        if self.update_info_int["lin_w"] > 0:
            if self.update_info_int["lin_ln"] == 1:
                modules.append(nn.LayerNorm(self.edge_vec_dim))
            if self.update_info_int["lin_w"] % 10 == 2:
                modules.append(self.activation)

            self.W_edp = InitDense(
                self.edge_vec_dim, n_atom_basis, activation=None, norm="layer" if self.update_info_int["lin_ln"] == 2 else "", bias=False
            )

            modules.append(self.W_edp)

        if self.update_info_str["gated"] == "gatedt":
            modules.append(nn.Tanh())
        elif self.update_info_str["gated"] == "gated":
            modules.append(nn.Sigmoid())
        elif self.update_info_str["gated"] == "act":
            modules.append(nn.SiLU())
        self.gamma_w = nn.Sequential(*modules)

        # Spatial filter
        self.W_rs = InitDense(
            n_atom_basis,
            n_atom_basis * self.multiplier,
            activation=None,
            bias=False,
        )

        # Normalization layers
        self.tensor_layernorm = TensorLayerNorm(num_channels=n_atom_basis, lmax=self.lmax, eps=self.epsilon)

        self.density_fn = Dense(
            n_rbf,
            1,
            bias=False,
            activation=activation,
            norm=None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        self.tensor_layernorm.reset_parameters()

        self.density_fn.reset_parameters()

        for l in self.gamma_s:  # noqa: E741
            l.reset_parameters()

        self.W_q.reset_parameters()
        self.W_k.reset_parameters()

        for l in self.gamma_v:  # noqa: E741
            l.reset_parameters()

        self.W_rs.reset_parameters()

        if self.edge_updates:
            self.gamma_t.reset_parameters()
            self.W_vq.reset_parameters()

            for w in self.W_vk:
                w.reset_parameters()

            if self.update_info_int["lin_w"] > 0:
                self.W_edp.reset_parameters()

    @staticmethod
    def vector_rejection(rep: Tensor, rl_ij: Tensor) -> Tensor:
        """
        Compute the vector rejection of vec onto rl_ij.

        Args:
            rep: Input tensor representation [num_edges, (L_max ** 2) - 1, hidden_dims]
            rl_ij: High-degree steerable feature tensor [num_edges, (L_max ** 2) - 1, 1]

        Returns:
            The component of vec orthogonal to rl_ij
        """
        vec_proj = (rep * rl_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return rep - vec_proj * rl_ij.unsqueeze(2)

    def forward(
        self,
        edge_index: Tensor,
        h: Tensor,
        X: Tensor,
        rl_ij: Tensor,
        t_ij: Tensor,
        r_ij: Tensor,
        phi_r0_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute interaction output for the Interaction layer.

        This method processes node and edge features through the attention mechanism
        and updates both scalar and high-degree steerable features.

        Args:
            edge_index: Tensor describing graph connectivity [2, num_edges]
            h: Scalar input values [num_nodes, 1, hidden_dims]
            X: High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
            rl_ij: Edge tensor representation [num_nodes, (L_max ** 2) - 1, 1]
            t_ij: Edge scalar features [num_nodes, 1, hidden_dims]
            r_ij: Edge scalar distance [num_nodes, 1]

        Returns:
            Tuple containing:
                - Updated scalar values [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
                - Updated edge features [num_edges, 1, hidden_dims]
        """
        h, X = self.tensor_layernorm(h, X)

        q = self.W_q(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.W_k(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        # inter-atomic
        x = self.gamma_s(h)
        v = self.gamma_v(h)
        t_ij_attn = self.W_re(t_ij)
        t_ij_filter = self.W_rs(t_ij)

        # propagate_type: (x: Tensor, q:Tensor, k:Tensor, v:Tensor, X: Tensor,
        #                  t_ij_filter: Tensor, t_ij_attn: Tensor, r_ij: Tensor,
        #                  rl_ij: Tensor, t_ij: Tensor, phi_r0_ij: Tensor)
        d_h, d_X, density = self.propagate(
            edge_index=edge_index,
            x=x,
            q=q,
            k=k,
            v=v,
            X=X,
            t_ij_filter=t_ij_filter,
            t_ij_attn=t_ij_attn,
            r_ij=r_ij,
            rl_ij=rl_ij,
            t_ij=t_ij,
            phi_r0_ij=phi_r0_ij,
        )

        h = h + d_h / (density + 1)
        X = X + d_X / (density + 1)

        if self.edge_updates:
            X_htr = X

            EQ = self.W_vq(X_htr)
            X_split = torch.split(X_htr, get_split_sizes_from_lmax(self.lmax), dim=1)
            EK = torch.concat([w(X_split[i]) for i, w in enumerate(self.W_vk)], dim=1)

            # edge_updater_type: (EQ: Tensor, EK:Tensor, rl_ij: Tensor, t_ij: Tensor)
            dt_ij = self.edge_updater(edge_index, EQ=EQ, EK=EK, rl_ij=rl_ij, t_ij=t_ij)
            t_ij = t_ij + dt_ij

        return h, X, t_ij

    def message(
        self,
        edge_index: Tensor,
        x_j: Tensor,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        X_j: Tensor,
        t_ij_filter: Tensor,
        t_ij_attn: Tensor,
        r_ij: Tensor,
        rl_ij: Tensor,
        t_ij: Tensor,
        phi_r0_ij: Tensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute messages from source nodes to target nodes.

        This method implements the message passing mechanism for the Interaction layer,
        combining attention-based and spatial filtering approaches.

        Args:
            edge_index: Edge connectivity tensor [2, num_edges]
            x_j: Source node features [num_edges, 1, hidden_dims]
            q_i: Target node query features [num_edges, num_heads, hidden_dims // num_heads]
            k_j: Source node key features [num_edges, num_heads, hidden_dims // num_heads]
            v_j: Source node value features [num_edges, num_heads, hidden_dims * multiplier // num_heads]
            X_j: Source node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            t_ij_filter: Edge scalar filter features [num_edges, 1, hidden_dims]
            t_ij_attn: Edge attention filter features [num_edges, 1, hidden_dims]
            r_ij: Edge scalar distance [num_edges, 1]
            rl_ij: Edge tensor representation [num_edges, (L_max ** 2) - 1, 1]
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation

        Returns:
            Tuple containing:
                - Scalar updates dh [num_edges, 1, hidden_dims]
                - High-degree steerable updates dX [num_edges, (L_max ** 2) - 1, hidden_dims]
        """
        # Reshape attention features
        t_ij_attn = t_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        logits = (q_i * k_j * t_ij_attn).sum(dim=-1, keepdim=True)
        attn = nn.functional.silu(logits)

        attn = attn / math.sqrt(self.n_atom_basis)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        sea_ij = attn * v_j.reshape(-1, self.num_heads, (self.n_atom_basis * self.multiplier) // self.num_heads)
        sea_ij = sea_ij.reshape(-1, 1, self.n_atom_basis * self.multiplier)

        # Apply spatial filter
        spatial_attn = t_ij_filter.unsqueeze(1) * x_j

        density = torch.tanh(self.density_fn(phi_r0_ij.unsqueeze(1) / self.n_rbf) ** 2)

        outputs = sea_ij + spatial_attn

        # Split outputs into components
        components = torch.split(outputs, self.n_atom_basis, dim=-1)

        o_s_ij = components[0]
        components = components[1:]

        o_d_ij, components = components[0], components[1:]
        dX_R = o_d_ij * rl_ij[..., None]

        o_t_ij = components[0]
        dX_X = o_t_ij * X_j

        # Combine components
        dX = dX_R + dX_X
        return o_s_ij, dX, density

    def edge_update(self, EQ_i: Tensor, EK_j: Tensor, rl_ij: Tensor, t_ij: Tensor) -> Tensor:
        """
        Update edge features based on node features.

        This method computes updates to edge features by combining information from
        source and target nodes' high-degree steerable features, potentially applying
        vector rejection.

        Args:
            EQ_i: Source node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            EK_j: Target node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            rl_ij: Edge tensor representation [num_edges, (L_max ** 2) - 1, 1]
            t_ij: Edge scalar features [num_edges, 1, hidden_dims]

        Returns:
            Updated edge features [num_edges, 1, hidden_dims]
        """
        EQ_i_split = split_to_components(EQ_i, self.lmax, dim=1)
        EK_j_split = split_to_components(EK_j, self.lmax, dim=1)
        rl_ij_split = split_to_components(rl_ij, self.lmax, dim=1)

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(len(EQ_i_split)):  # noqa: E741
            if self.update_info_bool["rej"]:
                EQ_i_l = self.vector_rejection(EQ_i_split[l], rl_ij_split[l])
                EK_j_l = self.vector_rejection(EK_j_split[l], -rl_ij_split[l])
            else:
                EQ_i_l = EQ_i_split[l]
                EK_j_l = EK_j_split[l]
            pairs.append((EQ_i_l, EK_j_l))

        # Compute edge weights
        w_ij = torch.zeros_like((pairs[0][0] * pairs[0][1]).sum(dim=1))
        for el in pairs:
            EQ_i_l, EK_j_l = el
            w_l = (EQ_i_l * EK_j_l).sum(dim=1)
            w_ij = w_ij + w_l

        return self.gamma_t(t_ij) * self.gamma_w(w_ij)

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Aggregate messages from source nodes to target nodes.

        This method implements the aggregation step of message passing, combining
        messages from neighboring nodes according to the specified aggregation method.

        Args:
            features: Tuple of scalar and vector features (h, X)
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation

        Returns:
            Tuple containing:
                - Aggregated scalar features [num_nodes, 1, hidden_dims]
                - Aggregated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        h, X, D = features
        h = scatter(h, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        X = scatter(X, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        D = scatter(D, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return h, X, D

    def update(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Update node features with aggregated messages.

        This method implements the update step of message passing. In this implementation,
        it simply passes through the aggregated features without additional processing.

        Args:
            inputs: Tuple of aggregated scalar and high-degree steerable features

        Returns:
            Tuple containing:
                - Updated scalar features [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        return inputs


class InteractionLast(MessagePassing):
    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        n_rbf: int,
        weight_init: Callable = nn.init.xavier_uniform_,
        aggr: str = "add",
        node_dim: int = 0,
        epsilon: float = 1e-7,
        num_heads: int = 8,
        dropout: float = 0.0,
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        lmax: int = 2,
    ):
        """
        Graph Attention Transformer Architecture.

        Args:
            n_atom_basis: Number of features to describe atomic environments.
            activation: Activation function to be used. If None, no activation function is used.
            weight_init: Weight initialization function.
            aggr: Aggregation method ('add', 'mean' or 'max').
            node_dim: The axis along which to aggregate.
            epsilon: Small constant for numerical stability.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            edge_updates: Whether to update edge features.
            last_layer: Whether this is the last layer.
            evec_dim: Dimension of edge vector features.
            emlp_dim: Dimension of edge MLP features.
            lmax: Maximum angular momentum.
        """
        super(InteractionLast, self).__init__(aggr=aggr, node_dim=node_dim)
        self.epsilon = epsilon
        self.activation = activation
        self.n_rbf = n_rbf

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis
        self.lmax = lmax

        multiplier = 3
        self.multiplier = multiplier

        # Initialize layers
        InitDense = partial(Dense, weight_init=weight_init, bias=False)

        # Implementation of gamma_s function
        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation, bias=False),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None, bias=False),
        )

        self.num_heads = num_heads

        # Query and key transformations
        self.W_q = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.W_k = InitDense(n_atom_basis, n_atom_basis, activation=None)

        # Value transformation
        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, multiplier * n_atom_basis, activation=None),
        )

        # Edge feature transformations
        self.W_re = InitDense(
            n_atom_basis,
            n_atom_basis,
            activation=activation,
            bias=False,
        )

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim

        # Spatial filter
        self.W_rs = InitDense(
            n_atom_basis,
            n_atom_basis * self.multiplier,
            activation=None,
            bias=False,
        )

        # Normalization layers
        self.tensor_layernorm = TensorLayerNorm(num_channels=n_atom_basis, lmax=self.lmax, eps=self.epsilon)

        self.density_fn = Dense(
            n_rbf,
            1,
            bias=False,
            activation=activation,
            norm=None,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        self.tensor_layernorm.reset_parameters()

        self.density_fn.reset_parameters()

        for l in self.gamma_s:  # noqa: E741
            l.reset_parameters()

        self.W_q.reset_parameters()
        self.W_k.reset_parameters()

        for l in self.gamma_v:  # noqa: E741
            l.reset_parameters()

        self.W_rs.reset_parameters()

    def forward(
        self,
        edge_index: Tensor,
        h: Tensor,
        X: Tensor,
        rl_ij: Tensor,
        t_ij: Tensor,
        r_ij: Tensor,
        phi_r0_ij: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute interaction output for the Interaction layer.

        This method processes node and edge features through the attention mechanism
        and updates both scalar and high-degree steerable features.

        Args:
            edge_index: Tensor describing graph connectivity [2, num_edges]
            h: Scalar input values [num_nodes, 1, hidden_dims]
            X: High-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
            rl_ij: Edge tensor representation [num_nodes, (L_max ** 2) - 1, 1]
            t_ij: Edge scalar features [num_nodes, 1, hidden_dims]
            r_ij: Edge scalar distance [num_nodes, 1]

        Returns:
            Tuple containing:
                - Updated scalar values [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
                - Updated edge features [num_edges, 1, hidden_dims]
        """
        h, X = self.tensor_layernorm(h, X)

        q = self.W_q(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.W_k(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        # inter-atomic
        x = self.gamma_s(h)
        v = self.gamma_v(h)
        t_ij_attn = self.W_re(t_ij)
        t_ij_filter = self.W_rs(t_ij)

        # propagate_type: (x: Tensor, q:Tensor, k:Tensor, v:Tensor, X: Tensor,
        #                  t_ij_filter: Tensor, t_ij_attn: Tensor, r_ij: Tensor,
        #                  rl_ij: Tensor, t_ij: Tensor, phi_r0_ij: Tensor)
        d_h, d_X, density = self.propagate(
            edge_index=edge_index,
            x=x,
            q=q,
            k=k,
            v=v,
            X=X,
            t_ij_filter=t_ij_filter,
            t_ij_attn=t_ij_attn,
            r_ij=r_ij,
            rl_ij=rl_ij,
            t_ij=t_ij,
            phi_r0_ij=phi_r0_ij,
        )

        h = h + d_h / (density + 1)
        X = X + d_X / (density + 1)

        return h, X, t_ij

    def message(
        self,
        edge_index: Tensor,
        x_j: Tensor,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        X_j: Tensor,
        t_ij_filter: Tensor,
        t_ij_attn: Tensor,
        r_ij: Tensor,
        rl_ij: Tensor,
        t_ij: Tensor,
        phi_r0_ij: Tensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute messages from source nodes to target nodes.

        This method implements the message passing mechanism for the Interaction layer,
        combining attention-based and spatial filtering approaches.

        Args:
            edge_index: Edge connectivity tensor [2, num_edges]
            x_j: Source node features [num_edges, 1, hidden_dims]
            q_i: Target node query features [num_edges, num_heads, hidden_dims // num_heads]
            k_j: Source node key features [num_edges, num_heads, hidden_dims // num_heads]
            v_j: Source node value features [num_edges, num_heads, hidden_dims * multiplier // num_heads]
            X_j: Source node high-degree steerable features [num_edges, (L_max ** 2) - 1, hidden_dims]
            t_ij_filter: Edge scalar filter features [num_edges, 1, hidden_dims]
            t_ij_attn: Edge attention filter features [num_edges, 1, hidden_dims]
            r_ij: Edge scalar distance [num_edges, 1]
            rl_ij: Edge tensor representation [num_edges, (L_max ** 2) - 1, 1]
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation

        Returns:
            Tuple containing:
                - Scalar updates dh [num_edges, 1, hidden_dims]
                - High-degree steerable updates dX [num_edges, (L_max ** 2) - 1, hidden_dims]
        """
        # Reshape attention features
        t_ij_attn = t_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        logits = (q_i * k_j * t_ij_attn).sum(dim=-1, keepdim=True)
        attn = nn.functional.silu(logits)

        attn = attn / math.sqrt(self.n_atom_basis)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Apply attention to values
        sea_ij = attn * v_j.reshape(-1, self.num_heads, (self.n_atom_basis * self.multiplier) // self.num_heads)
        sea_ij = sea_ij.reshape(-1, 1, self.n_atom_basis * self.multiplier)

        # Apply spatial filter
        spatial_attn = t_ij_filter.unsqueeze(1) * x_j

        density = torch.tanh(self.density_fn(phi_r0_ij.unsqueeze(1) / self.n_rbf) ** 2)

        outputs = spatial_attn + sea_ij

        # Split outputs into components
        components = torch.split(outputs, self.n_atom_basis, dim=-1)

        o_s_ij = components[0]
        components = components[1:]

        o_d_ij, components = components[0], components[1:]
        dX_R = o_d_ij * rl_ij[..., None]

        o_t_ij = components[0]
        dX_X = o_t_ij * X_j

        # Combine components
        dX = dX_R + dX_X
        return o_s_ij, dX, density

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Aggregate messages from source nodes to target nodes.

        This method implements the aggregation step of message passing, combining
        messages from neighboring nodes according to the specified aggregation method.

        Args:
            features: Tuple of scalar and vector features (h, X)
            index: Index tensor for scatter operation
            ptr: Pointer tensor for scatter operation
            dim_size: Dimension size for scatter operation

        Returns:
            Tuple containing:
                - Aggregated scalar features [num_nodes, 1, hidden_dims]
                - Aggregated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        h, X, density = features
        h = scatter(h, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        X = scatter(X, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        D = scatter(density, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return h, X, D

    def update(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Update node features with aggregated messages.

        This method implements the update step of message passing. In this implementation,
        it simply passes through the aggregated features without additional processing.

        Args:
            inputs: Tuple of aggregated scalar and high-degree steerable features

        Returns:
            Tuple containing:
                - Updated scalar features [num_nodes, 1, hidden_dims]
                - Updated high-degree steerable features [num_nodes, (L_max ** 2) - 1, hidden_dims]
        """
        return inputs


class EquivariantCoupling(nn.Module):
    """
    Equivariant Feed-Forward (EQFF) Network for mixing atom features.

    This module facilitates efficient channel-wise interaction while maintaining equivariance.
    It separates scalar and high-degree steerable features, allowing for specialized processing
    of each feature type before combining them with non-linear mappings as described in the paper:

    EQFF(h, X^(l)) = (h + m_1, X^(l) + m_2 * (X^(l)W_{vu}))
    where m_1, m_2 = split_2(gamma_{m}(||X^(l)W_{vu}||_2, h))
    """

    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        lmax: int,
        weight_init: Callable = nn.init.xavier_uniform_,
    ):
        """
        Initialize EQFF module.

        Args:
            n_atom_basis: Number of features to describe atomic environments.
            activation: Activation function. If None, no activation function is used.
            lmax: Maximum angular momentum.
            weight_init: Weight initialization function.
        """
        super(EquivariantCoupling, self).__init__()
        self.lmax = lmax
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias=False)

        context_dim = 2 * n_atom_basis
        out_size = 2

        self.gamma_m = nn.Sequential(
            InitDense(context_dim, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, out_size * n_atom_basis, activation=None),
        )

        self.W_vu = InitDense(n_atom_basis, n_atom_basis, activation=None)

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        self.W_vu.reset_parameters()
        for l in self.gamma_m:  # noqa: E741
            l.reset_parameters()

    def forward(self, h: Tensor, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute intraatomic mixing.

        Args:
            h: Scalar input values, [num_nodes, 1, hidden_dims].
            X: High-degree steerable features, [num_nodes, (L_max ** 2) - 1, hidden_dims].

        Returns:
            Tuple of updated scalar values and high-degree steerable features,
            each of shape [num_nodes, 1, hidden_dims] and [num_nodes, (L_max ** 2) - 1, hidden_dims].
        """
        X_p = self.W_vu(X)

        X_pn = torch.sum(X_p**2, dim=-2, keepdim=True)

        # Concatenate features for context
        channel_context = [h, X_pn]
        ctx = torch.cat(channel_context, dim=-1)

        # Apply gamma_m transformation
        x = self.gamma_m(ctx)

        # Split output into scalar and vector components
        m1, m2 = torch.split(x, self.n_atom_basis, dim=-1)

        # Update features with residual connections
        h = h + m1
        X = X + m2 * X_p

        return h, X


class Geodite(AbstractEncoder):
    """
    Graph Attention Transformer Network for atomic systems.

    GotenNet processes and updates two types of node features (invariant and steerable)
    and edge features (invariant) through three main mechanisms:

    1. Interaction (Graph Attention Transformer Architecture): A degree-wise attention-based
       message passing layer that updates both invariant and steerable features while
       preserving equivariance.
    2. HTR (Hierarchical Tensor Refinement): Updates edge features across degrees with
       inner products of steerable features.
    3. EQFF (Equivariant Feed-Forward): Further processes both types of node features
       while maintaining equivariance.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        radial_basis: Union[Callable, str] = "besselbasis",
        n_rbf: int = 12,
        cutoff: float = 5.0,
        activation: Optional[Union[Callable, str]] = F.silu,
        max_z: int = 100,
        epsilon: float = 1e-8,
        weight_init: Callable = nn.init.xavier_uniform_,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        edge_updates: Union[bool, str] = True,
        lmax: int = 1,
        aggr: str = "add",
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        edge_ln: str = "",
        distance_weighting: bool = False,
        distance_weighting_trainable: bool = False,
        **kwargs,
    ):
        """
        Initialize Geodite model.

        Args:
            hidden_channels: Number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            num_layers: Number of interaction blocks.
            radial_basis: Layer for expanding interatomic distances in a basis set.
            n_rbf: Number of radial basis functions.
            cutoff: Cutoff radius.
            activation: Activation function.
            max_z: Maximum atomic number.
            epsilon: Stability constant added in norm to prevent numerical instabilities.
            weight_init: Weight initialization function.
            max_num_neighbors: Maximum number of neighbors.
            num_heads: Number of attention heads.
            attn_dropout: Dropout probability for attention.
            edge_updates: Whether to update edge features.
            lmax: Maximum angular momentum.
            aggr: Aggregation method ('add', 'mean' or 'max').
            evec_dim: Dimension of edge vector features.
            emlp_dim: Dimension of edge MLP features.
        """
        super(Geodite, self).__init__()

        n_atom_basis = hidden_channels
        n_interactions = num_layers

        if type(weight_init) == str:  # noqa: E721
            weight_init = get_weight_init_by_string(weight_init)

        if type(activation) is str:
            activation = str2act(activation)

        self.n_atom_basis = self.hidden_dim = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.distance_weighting = distance_weighting

        self.node_init = NodeInit(
            [self.hidden_dim, self.hidden_dim],
            n_rbf,
            max_z=max_z,
            weight_init=weight_init,
            proj_ln="",
            activation=activation,
        )

        self.edge_init = EdgeInit(n_rbf, self.hidden_dim)

        radial_basis = str2basis(radial_basis)
        self.radial_basis = radial_basis(cutoff=self.cutoff, n_rbf=n_rbf)
        self.cutoff_fn = PolynomialCutoff(self.cutoff)

        self.A_na = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        self.sphere = TensorInit(l=lmax)

        self.interaction_list = nn.ModuleList(
            [
                Interaction(
                    n_atom_basis=self.n_atom_basis,
                    activation=activation,
                    aggr=aggr,
                    weight_init=weight_init,
                    epsilon=epsilon,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    edge_updates=edge_updates,
                    evec_dim=evec_dim,
                    emlp_dim=emlp_dim,
                    lmax=lmax,
                    edge_ln=edge_ln,
                    n_rbf=n_rbf,
                )
                for _ in range(self.n_interactions - 1)
            ]
            + [
                InteractionLast(
                    n_atom_basis=self.n_atom_basis,
                    activation=activation,
                    aggr=aggr,
                    weight_init=weight_init,
                    epsilon=epsilon,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    evec_dim=evec_dim,
                    emlp_dim=emlp_dim,
                    lmax=lmax,
                    n_rbf=n_rbf,
                )
            ]
        )

        self.ec_list = nn.ModuleList(
            [
                EquivariantCoupling(
                    n_atom_basis=self.n_atom_basis,
                    activation=activation,
                    lmax=lmax,
                    weight_init=weight_init,
                )
                for i in range(self.n_interactions)
            ]
        )

        self.distance_weighting_fn = (
            DistanceWeighting(trainable=distance_weighting_trainable) if self.distance_weighting else IdentityDistanceWeighting()
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.node_init.reset_parameters()
        self.edge_init.reset_parameters()
        for l in self.interaction_list:  # noqa: E741
            l.reset_parameters()
        for l in self.ec_list:  # noqa: E741
            l.reset_parameters()

    def _compute_embeddings(self, z, cutoff_edge_index, cutoff_edge_distance, cutoff_edge_vec):
        h = self.A_na(z)[:]

        r0_ij = cutoff_edge_distance.unsqueeze(1)

        w = self.distance_weighting_fn(z, cutoff_edge_distance, cutoff_edge_index).unsqueeze(-1) * self.cutoff_fn(r0_ij)
        phi_r0_ij = self.radial_basis(cutoff_edge_distance) * w

        h = self.node_init(z, h, cutoff_edge_index, phi_r0_ij)

        t_ij_init = self.edge_init(cutoff_edge_index, phi_r0_ij, h)

        rl_ij = self.sphere(cutoff_edge_vec / r0_ij)

        equi_dim = ((self.sphere.l + 1) ** 2) - 1

        hs = h.shape
        X = torch.zeros((int(hs[0]), int(equi_dim), int(hs[1])), device=h.device)
        h.unsqueeze_(1)
        t_ij = t_ij_init
        layer_outputs = []
        for interaction, ec in zip(self.interaction_list, self.ec_list, strict=False):
            h, X, t_ij = interaction(cutoff_edge_index, h, X, rl_ij=rl_ij, t_ij=t_ij, r_ij=r0_ij, phi_r0_ij=phi_r0_ij)
            h, X = ec(h, X)
            layer_outputs.append(h)

        return torch.stack(layer_outputs, dim=-1)

    def forward(self, data: DataInput):
        # Build vacuum-node list
        z = data.z
        unique_z, vac_idx_per_node = torch.unique(z, sorted=True, return_inverse=True)
        N = z.size(0)

        # Build concatenated inputs
        z_comb = torch.cat([z, unique_z], dim=0)

        idx_comb = data.cutoff_edge_index
        dist_comb = data.cutoff_edge_distance
        vec_comb = data.cutoff_edge_vec

        layer_all = self._compute_embeddings(
            z_comb,
            idx_comb,
            dist_comb,
            vec_comb,
        )

        # Split real vs. vacuum
        layer_real = layer_all[:N]
        layer_vac = layer_all[N:]

        # For each real node, pick its matching vac node by atomic number
        vac_layer_per_node = layer_vac[vac_idx_per_node]

        # Subtract to isolate environment
        out_layer = layer_real - vac_layer_per_node

        embedding_0 = out_layer.permute(0, 2, 1, 3)  # (N, F, 1, L)

        return {
            "embedding_0": embedding_0,
        }
