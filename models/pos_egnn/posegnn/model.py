from typing import Dict

import torch
from torch import nn, Tensor
from .encoder import GotenNet
from .utils import (
    get_symmetric_displacement,
    PeriodicDistance,
    ACT_CLASS_MAPPING,
)


class NodeInvariantReadout(nn.Module):
    def __init__(
        self, in_channels, num_residues, hidden_channels, out_channels, activation
    ):
        super().__init__()

        self.linears = nn.ModuleList(
            [nn.Linear(in_channels, out_channels) for _ in range(num_residues - 1)]
        )

        self.non_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            ACT_CLASS_MAPPING[activation](),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, embedding_0):
        layer_outputs = embedding_0.squeeze(2)

        outputs = torch.stack(
            [linear(layer_outputs[:, :, i]) for i, linear in enumerate(self.linears)],
            dim=0,
        )

        last_output = self.non_linear(layer_outputs[:, :, -1]).unsqueeze(0)
        processed_outputs = torch.cat([outputs, last_output], dim=0)
        output = processed_outputs.sum(dim=0).squeeze(-1)

        return output


class PosEGNN(nn.Module):
    def __init__(self, config: Dict, **kwargs):
        super().__init__()

        self.distance = PeriodicDistance(
            config["encoder"]["cutoff"], skin=kwargs.get("skin", None)
        )
        self.encoder = GotenNet(**config["encoder"])
        self.readout = NodeInvariantReadout(**config["decoder"])

        self.register_buffer("e0_mean", torch.tensor(config["e0_mean"]))
        self.register_buffer(
            "atomic_res_total_mean", torch.tensor(config["atomic_res_total_mean"])
        )
        self.register_buffer(
            "atomic_res_total_std", torch.tensor(config["atomic_res_total_std"])
        )

    def forward(self, z: Tensor, pos: Tensor, box: Tensor):
        pos_grad = pos.clone().requires_grad_(True)

        pos, box, displacements = get_symmetric_displacement(pos_grad, box)

        cutoff_edge_index, cutoff_edge_distance, cutoff_edge_vec, cutoff_shifts_idx = (
            self.distance(pos, box)
        )

        embedding_dict = self.encoder(
            z, pos, cutoff_edge_index, cutoff_edge_distance, cutoff_edge_vec
        )

        return embedding_dict, pos, displacements

    def compute_properties(
        self, z: Tensor, pos: Tensor, box: Tensor, compute_stress: float = True
    ):
        output = {}

        embedding_dict, pos, displacements = self.forward(z, pos, box)
        embedding_0 = embedding_dict["embedding_0"]

        node_e_res = self.readout(embedding_0)

        node_e_res = node_e_res * self.atomic_res_total_std + self.atomic_res_total_mean
        node_e0 = self.e0_mean[z]
        total_energy = node_e0.sum() + node_e_res.sum()

        output["total_energy"] = total_energy

        if compute_stress:
            inputs = [pos, displacements]
        else:
            inputs = [pos]

        grad_outputs = torch.autograd.grad(
            outputs=[total_energy],
            inputs=inputs,
            grad_outputs=[torch.ones_like(total_energy)],
            retain_graph=self.training,
            create_graph=self.training,
        )

        if compute_stress:
            force, virial = grad_outputs
            stress = virial / torch.det(box).abs().view(-1, 1, 1)
            output["force"] = -force
            output["stress"] = -stress
        else:
            force = grad_outputs[0]
            output["force"] = -force

        return output
