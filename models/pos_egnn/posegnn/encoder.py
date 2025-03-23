"""
This code was adapted from https://github.com/sarpaykent/GotenNet
Copyright (c) 2025 Sarp Aykent
MIT License

GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks
Sarp Aykent and Tian Xia
https://openreview.net/pdf?id=5wxCQDtbMo
"""

from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter, softmax

from .ops import (
    MLP,
    CosineCutoff,
    Dense,
    EdgeInit,
    NodeInit,
    TensorInit,
    TensorLayerNorm,
    get_weight_init_by_string,
    parse_update_info,
    str2act,
    str2basis,
)


def lmax_tensor_size(lmax):
    return ((lmax + 1) ** 2) - 1


def split_degree(tensor, lmax, dim=-1):  # default to last dim
    cumsum = 0
    tensors = []
    for i in range(1, lmax + 1):
        count = lmax_tensor_size(i) - lmax_tensor_size(i - 1)
        # Create slice object for the specified dimension
        slc = [slice(None)] * tensor.ndim  # Create list of slice(None) for all dims
        slc[dim] = slice(cumsum, cumsum + count)  # Replace desired dim with actual slice
        tensors.append(tensor[tuple(slc)])
        cumsum += count
    return tensors


class GATA(MessagePassing):
    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        aggr="add",
        node_dim=0,
        epsilon: float = 1e-7,
        layer_norm=False,
        vector_norm=False,
        cutoff=5.0,
        num_heads=8,
        dropout=0.0,
        edge_updates=True,
        last_layer=False,
        scale_edge=True,
        edge_ln="",
        evec_dim=None,
        emlp_dim=None,
        sep_vecj=True,
        lmax=1,
    ):
        """
        Args:
            n_atom_basis (int): Number of features to describe atomic environments.
            activation (Callable): Activation function to be used. If None, no activation function is used.
            weight_init (Callable): Weight initialization function.
            bias_init (Callable): Bias initialization function.
            aggr (str): Aggregation method ('add', 'mean' or 'max').
            node_dim (int): The axis along which to aggregate.
        """
        super(GATA, self).__init__(aggr=aggr, node_dim=node_dim)
        self.lmax = lmax
        self.sep_vecj = sep_vecj
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation

        self.update_info = parse_update_info(edge_updates)

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.num_heads = num_heads
        self.q_w = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.k_w = InitDense(n_atom_basis, n_atom_basis, activation=None)

        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.phik_w_ra = InitDense(
            n_atom_basis,
            n_atom_basis,
            activation=activation,
        )

        InitMLP = partial(MLP, weight_init=weight_init, bias_init=bias_init)

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim
        if not self.last_layer and self.edge_updates:
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
            else:
                dims = [n_atom_basis, n_atom_basis]
            self.edge_attr_up = InitMLP(
                dims, activation=activation, last_activation=None if self.update_info["mlp"] else self.activation, norm=edge_ln
            )
            self.vecq_w = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.sep_vecj:
                self.veck_w = nn.ModuleList(
                    [InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False) for i in range(self.lmax)]
                )
            else:
                self.veck_w = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.update_info["lin_w"] > 0:
                modules = []
                if self.update_info["lin_w"] % 10 == 2:
                    modules.append(self.activation)
                self.lin_w_linear = InitDense(
                    self.edge_vec_dim,
                    n_atom_basis,
                    activation=None,
                    norm="layer" if self.update_info["lin_w"] == 2 else "",  # lin_ln in original code but error
                )
                modules.append(self.lin_w_linear)
                self.lin_w = nn.Sequential(*modules)

        self.down_proj = nn.Identity()

        self.cutoff = CosineCutoff(cutoff)
        self._alpha = None

        self.w_re = InitDense(
            n_atom_basis,
            n_atom_basis * 3,
            None,
        )

        self.layernorm_ = layer_norm
        self.vector_norm_ = vector_norm

        if layer_norm:
            self.layernorm = nn.LayerNorm(n_atom_basis)
        else:
            self.layernorm = nn.Identity()
        if vector_norm:
            self.tln = TensorLayerNorm(n_atom_basis, trainable=False)
        else:
            self.tln = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        if self.layernorm_:
            self.layernorm.reset_parameters()
        if self.vector_norm_:
            self.tln.reset_parameters()
        for l in self.gamma_s:  # noqa: E741
            l.reset_parameters()

        self.q_w.reset_parameters()
        self.k_w.reset_parameters()
        for l in self.gamma_v:  # noqa: E741
            l.reset_parameters()
        # self.v_w.reset_parameters()
        # self.out_w.reset_parameters()
        self.w_re.reset_parameters()

        if not self.last_layer and self.edge_updates:
            self.edge_attr_up.reset_parameters()
            self.vecq_w.reset_parameters()

            if self.sep_vecj:
                for w in self.veck_w:
                    w.reset_parameters()
            else:
                self.veck_w.reset_parameters()

            if self.update_info["lin_w"] > 0:
                self.lin_w_linear.reset_parameters()

    def forward(
        self,
        edge_index,
        s: torch.Tensor,
        t: torch.Tensor,
        dir_ij: torch.Tensor,
        r_ij: torch.Tensor,
        d_ij: torch.Tensor,
        num_edges_expanded: torch.Tensor,
    ):
        """Compute interaction output."""
        s = self.layernorm(s)
        t = self.tln(t)

        q = self.q_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.k_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        x = self.gamma_s(s)
        val = self.gamma_v(s)
        f_ij = r_ij
        r_ij_attn = self.phik_w_ra(r_ij)
        r_ij = self.w_re(r_ij)

        # propagate_type: (x: Tensor, ten: Tensor, q:Tensor, k:Tensor, val:Tensor, r_ij: Tensor, r_ij_attn: Tensor, d_ij:Tensor, dir_ij: Tensor, num_edges_expanded: Tensor)
        su, tu = self.propagate(
            edge_index=edge_index,
            x=x,
            q=q,
            k=k,
            val=val,
            ten=t,
            r_ij=r_ij,
            r_ij_attn=r_ij_attn,
            d_ij=d_ij,
            dir_ij=dir_ij,
            num_edges_expanded=num_edges_expanded,
        )  # , f_ij=f_ij

        s = s + su
        t = t + tu

        if not self.last_layer and self.edge_updates:
            vec = t

            w1 = self.vecq_w(vec)
            if self.sep_vecj:
                vec_split = split_degree(vec, self.lmax, dim=1)
                w_out = torch.concat([w(vec_split[i]) for i, w in enumerate(self.veck_w)], dim=1)

            else:
                w_out = self.veck_w(vec)

            # edge_updater_type: (w1: Tensor, w2:Tensor,  d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, w1=w1, w2=w_out, d_ij=dir_ij, f_ij=f_ij)
            df_ij = f_ij + df_ij
            self._alpha = None
            return s, t, df_ij
        else:
            self._alpha = None
            return s, t, f_ij

        # return s, t

    def message(
        self,
        edge_index,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        val_j: torch.Tensor,
        ten_j: torch.Tensor,
        r_ij: torch.Tensor,
        r_ij_attn: torch.Tensor,
        d_ij: torch.Tensor,
        dir_ij: torch.Tensor,
        num_edges_expanded: torch.Tensor,
        index: torch.Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute message passing.
        """

        r_ij_attn = r_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        attn = (q_i * k_j * r_ij_attn).sum(dim=-1, keepdim=True)

        attn = softmax(attn, index, ptr, dim_size)

        # Normalize the attention scores
        if self.scale_edge:
            norm = torch.sqrt(num_edges_expanded.reshape(-1, 1, 1)) / np.sqrt(self.n_atom_basis)
        else:
            norm = 1.0 / np.sqrt(self.n_atom_basis)
        attn = attn * norm
        self._alpha = attn
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        self_attn = attn * val_j.reshape(-1, self.num_heads, (self.n_atom_basis * 3) // self.num_heads)
        SEA = self_attn.reshape(-1, 1, self.n_atom_basis * 3)

        x = SEA + (r_ij.unsqueeze(1) * x_j * self.cutoff(d_ij.unsqueeze(-1).unsqueeze(-1)))

        o_s, o_d, o_t = torch.split(x, self.n_atom_basis, dim=-1)
        dmu = o_d * dir_ij[..., None] + o_t * ten_j
        return o_s, dmu

    @staticmethod
    def rej(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def edge_update(self, w1_i, w2_j, w3_j, d_ij, f_ij):
        if self.sep_vecj:
            vi = w1_i
            vj = w2_j
            vi_split = split_degree(vi, self.lmax, dim=1)
            vj_split = split_degree(vj, self.lmax, dim=1)
            d_ij_split = split_degree(d_ij, self.lmax, dim=1)

            pairs = []
            for i in range(len(vi_split)):
                if self.update_info["rej"]:
                    w1 = self.rej(vi_split[i], d_ij_split[i])
                    w2 = self.rej(vj_split[i], -d_ij_split[i])
                    pairs.append((w1, w2))
                else:
                    w1 = vi_split[i]
                    w2 = vj_split[i]
                    pairs.append((w1, w2))
        elif not self.update_info["rej"]:
            w1 = w1_i
            w2 = w2_j
            pairs = [(w1, w2)]
        else:
            w1 = self.rej(w1_i, d_ij)
            w2 = self.rej(w2_j, -d_ij)
            pairs = [(w1, w2)]

        w_dot_sum = None
        for el in pairs:
            w1, w2 = el
            w_dot = (w1 * w2).sum(dim=1)
            if w_dot_sum is None:
                w_dot_sum = w_dot
            else:
                w_dot_sum = w_dot_sum + w_dot
        w_dot = w_dot_sum
        if self.update_info["lin_w"] > 0:
            w_dot = self.lin_w(w_dot)

        if self.update_info["gated"] == "gatedt":
            w_dot = torch.tanh(w_dot)
        elif self.update_info["gated"] == "gated":
            w_dot = torch.sigmoid(w_dot)
        elif self.update_info["gated"] == "act":
            w_dot = self.activation(w_dot)

        df_ij = self.edge_attr_up(f_ij) * w_dot
        return df_ij

    # noinspection PyMethodOverriding
    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EQFF(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        activation: Callable,
        epsilon: float = 1e-8,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        vec_dim=None,
    ):
        """Equiavariant Feed Forward layer."""
        super(EQFF, self).__init__()
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        vec_dim = n_atom_basis if vec_dim is None else vec_dim
        context_dim = 2 * n_atom_basis

        self.gamma_m = nn.Sequential(
            InitDense(context_dim, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 2 * n_atom_basis, activation=None),
        )
        self.w_vu = InitDense(n_atom_basis, vec_dim, activation=None, bias=False)

        self.epsilon = epsilon

    def reset_parameters(self):
        self.w_vu.reset_parameters()
        for l in self.gamma_m:  # noqa: E741
            l.reset_parameters()

    def forward(self, s, v):
        """Compute Equivariant Feed Forward output."""

        t_prime = self.w_vu(v)
        t_prime_mag = torch.sqrt(torch.sum(t_prime**2, dim=-2, keepdim=True) + self.epsilon)
        combined = [s, t_prime_mag]
        combined_tensor = torch.cat(combined, dim=-1)
        m12 = self.gamma_m(combined_tensor)

        m_1, m_2 = torch.split(m12, self.n_atom_basis, dim=-1)
        delta_v = m_2 * t_prime

        s = s + m_1
        v = v + delta_v

        return s, v


class GotenNet(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 8, 
        radial_basis: Union[Callable, str] = "BesselBasis",
        n_rbf: int = 20,
        cutoff: float = 5.0,
        activation: Optional[Union[Callable, str]] = F.silu,
        max_z: int = 100,
        epsilon: float = 1e-8,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        int_layer_norm=False,
        int_vector_norm=False,
        before_mixing_layer_norm=False,
        after_mixing_layer_norm=False,
        num_heads=8,
        attn_dropout=0.0,
        edge_updates=True,
        scale_edge=True,
        lmax=2,
        aggr="add",
        edge_ln="",
        evec_dim=None,
        emlp_dim=None,
        sep_int_vec=True,
    ):
        """
        Representation for GotenNet
        """
        super(GotenNet, self).__init__()

        self.scale_edge = scale_edge
        if type(weight_init) == str:  # noqa: E721
            # print(f"Using {weight_init} weight initialization")
            weight_init = get_weight_init_by_string(weight_init)

        if type(bias_init) == str:  # noqa: E721
            bias_init = get_weight_init_by_string(bias_init)

        if type(activation) is str:
            activation = str2act(activation)

        self.n_atom_basis = self.hidden_dim = hidden_channels
        self.n_interactions = num_layers
        self.cutoff = cutoff

        self.neighbor_embedding = NodeInit(
            [self.hidden_dim // 2, self.hidden_dim],
            n_rbf,
            self.cutoff,
            max_z=max_z,
            weight_init=weight_init,
            bias_init=bias_init,
            concat=False,
            proj_ln="layer",
            activation=activation,
        )
        self.edge_embedding = EdgeInit(
            n_rbf, [self.hidden_dim // 2, self.hidden_dim], weight_init=weight_init, bias_init=bias_init, proj_ln=""
        )

        radial_basis = str2basis(radial_basis)
        self.radial_basis = radial_basis(cutoff=self.cutoff, n_rbf=n_rbf)

        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        self.tensor_init = TensorInit(l=lmax)

        self.gata = nn.ModuleList(
            [
                GATA(
                    n_atom_basis=self.n_atom_basis,
                    activation=activation,
                    aggr=aggr,
                    weight_init=weight_init,
                    bias_init=bias_init,
                    layer_norm=int_layer_norm,
                    vector_norm=int_vector_norm,
                    cutoff=self.cutoff,
                    epsilon=epsilon,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    edge_updates=edge_updates,
                    last_layer=(i == self.n_interactions - 1),
                    scale_edge=scale_edge,
                    edge_ln=edge_ln,
                    evec_dim=evec_dim,
                    emlp_dim=emlp_dim,
                    sep_vecj=sep_int_vec,
                    lmax=lmax,
                )
                for i in range(self.n_interactions)
            ]
        )

        self.eqff = nn.ModuleList(
            [
                EQFF(n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon, weight_init=weight_init, bias_init=bias_init)
                for i in range(self.n_interactions)
            ]
        )

        # Extra layer norms for the scalar quantities
        if before_mixing_layer_norm:
            self.before_mixing_ln = nn.LayerNorm(self.n_atom_basis)
        else:
            self.before_mixing_ln = nn.Identity()

        if after_mixing_layer_norm:
            self.after_mixing_ln = nn.LayerNorm(self.n_atom_basis)
        else:
            self.after_mixing_ln = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        for l in self.gata:  # noqa: E741
            l.reset_parameters()
        for l in self.eqff:  # noqa: E741
            l.reset_parameters()

        if not isinstance(self.before_mixing_ln, nn.Identity):
            self.before_mixing_ln.reset_parameters()
        if not isinstance(self.after_mixing_ln, nn.Identity):
            self.after_mixing_ln.reset_parameters()

    def forward(self, z, pos, cutoff_edge_index, cutoff_edge_distance, cutoff_edge_vec):
        q = self.embedding(z)[:]

        edge_attr = self.radial_basis(cutoff_edge_distance)

        q = self.neighbor_embedding(z, q, cutoff_edge_index, cutoff_edge_distance, edge_attr)
        edge_attr = self.edge_embedding(cutoff_edge_index, edge_attr, q)
        mask = cutoff_edge_index[0] != cutoff_edge_index[1]
        # direction vector
        dist = torch.norm(cutoff_edge_vec[mask], dim=1).unsqueeze(1)
        cutoff_edge_vec[mask] = cutoff_edge_vec[mask] / dist

        cutoff_edge_vec = self.tensor_init(cutoff_edge_vec)
        equi_dim = ((self.tensor_init.l + 1) ** 2) - 1
        # count number of edges for each node
        num_edges = scatter(torch.ones_like(cutoff_edge_distance), cutoff_edge_index[0], dim=0, reduce="sum")
        # the shape of num edges is [num_nodes, 1], we want to expand this to [num_edges, 1]
        # Map num_edges back to the shape of attn using cutoff_edge_index
        num_edges_expanded = num_edges[cutoff_edge_index[0]]

        qs = q.shape
        mu = torch.zeros((qs[0], equi_dim, qs[1]), device=q.device)
        q.unsqueeze_(1)

        layer_outputs = []

        for i, (interaction, mixing) in enumerate(zip(self.gata, self.eqff)):
            q, mu, edge_attr = interaction(
                cutoff_edge_index,
                q,
                mu,
                dir_ij=cutoff_edge_vec,
                r_ij=edge_attr,
                d_ij=cutoff_edge_distance,
                num_edges_expanded=num_edges_expanded,
            )

            q = self.before_mixing_ln(q)
            q, mu = mixing(q, mu)
            q = self.after_mixing_ln(q)

            # Collect all scalars for inter-layer read-outs
            layer_outputs.append(q.squeeze(1))

        # q = q.squeeze(1)

        layer_outputs = torch.stack(layer_outputs, dim=-1)

        output_dict = {}
        output_dict["embedding_0"] = layer_outputs.unsqueeze(2)  # [n_nodes, n_features, dimension of irrep, n_layers]
        # This is a scalar so a single irrep

        return output_dict
