from torch import nn
import torch
from .encoder import GotenNet
from .utils import get_symmetric_displacement, BatchedPeriodicDistance, ACT_CLASS_MAPPING
from torch_scatter import scatter

class NodeInvariantReadout(nn.Module):
    def __init__(self, in_channels, num_residues, hidden_channels, out_channels, activation):
        super().__init__()

        self.linears = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(num_residues - 1)])

        # Define the nonlinear layer for the last layer's output
        self.non_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            ACT_CLASS_MAPPING[activation](),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, embedding_0):
        layer_outputs = embedding_0.squeeze(2)  # [n_nodes, in_channels, num_residues]

        processed_outputs = []
        for i, linear in enumerate(self.linears):
            processed_outputs.append(linear(layer_outputs[:, :, i]))

        processed_outputs.append(self.non_linear(layer_outputs[:, :, -1]))
        output = torch.stack(processed_outputs, dim=0).sum(dim=0).squeeze(-1)
        return output

class PosEGNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.distance = BatchedPeriodicDistance(config["encoder"]["cutoff"])
        self.encoder = GotenNet(**config["encoder"])
        self.readout = NodeInvariantReadout(**config["decoder"])
        self.register_buffer("e0_mean", torch.tensor(config["e0_mean"]))
        self.register_buffer("atomic_res_total_mean", torch.tensor(config["atomic_res_total_mean"]))
        self.register_buffer("atomic_res_total_std", torch.tensor(config["atomic_res_total_std"]))

    def forward(self, data):
        data.pos.requires_grad_(True)

        data.pos, data.box, data.displacements = get_symmetric_displacement(data.pos, data.box, data.num_graphs, data.batch)

        data.cutoff_edge_index, data.cutoff_edge_distance, data.cutoff_edge_vec, data.cutoff_shifts_idx = self.distance(
            data.pos, data.box, data.batch
        )

        embedding_dict = self.encoder(data.z, data.pos, data.cutoff_edge_index, data.cutoff_edge_distance, data.cutoff_edge_vec)

        return embedding_dict

    def compute_properties(self, data, compute_stress = True):
        output = {}
        
        embedding_dict = self.forward(data)
        embedding_0 = embedding_dict["embedding_0"]

        # Compute energy
        node_e_res = self.readout(embedding_0)

        node_e_res = node_e_res * self.atomic_res_total_std + self.atomic_res_total_mean
        total_e_res = scatter(src=node_e_res, index=data["batch"], dim=0, reduce="sum")

        node_e0 = self.e0_mean[data.z]
        total_e0 = scatter(src=node_e0, index=data["batch"], dim=0, reduce="sum")

        total_energy = total_e0 + total_e_res
        output["total_energy"] = total_energy

        # Compute gradients
        if compute_stress:
            inputs = [data.pos, data.displacements]
            compute_stress = True
        else:
            inputs = [data.pos]

        grad_outputs = torch.autograd.grad(
            outputs=[total_energy],
            inputs=inputs,
            grad_outputs=[torch.ones_like(total_energy)],
            retain_graph=self.training,
            create_graph=self.training,
        )

        # Get forces and stresses
        if compute_stress:
            force, virial = grad_outputs
            stress = virial / torch.det(data.box).abs().view(-1, 1, 1)
            stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
            output["force"] = -force
            output["stress"] = -stress
        else:
            force = grad_outputs[0]
            output["force"] = -force        
        
        return output