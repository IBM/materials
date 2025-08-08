from typing import Dict

import torch
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter_sum

from .._base_decoder import AbstractDecoder
from ..readouts import NodeInvariantReadout


class SnapshotDecoder(AbstractDecoder):
    def __init__(self, in_channels, num_residues, hidden_channels, activation, **kwargs):
        super().__init__(in_channels, num_residues, hidden_channels, activation, **kwargs)

        self.weights = torch.tensor(kwargs.get("snapshot_weights", [1, 1, 1]))

        self.readout = NodeInvariantReadout(in_channels, num_residues, hidden_channels, 1, activation)

        self.mae = nn.L1Loss()
        self.loss_fn = nn.HuberLoss(delta=1.0)

        print(f"Snapshot decoder:\nUsing weights E={self.weights[0]}, F={self.weights[1]}, S={self.weights[2]}\nUsing loss {self.loss_fn}")

    def forward(self, data: Data, compute_stress=False) -> torch.Tensor:
        node_e_res = self.readout(data.embedding_0) * self.atomic_res_total_std + self.atomic_res_total_mean
        node_e0 = self.e0_mean[data.z]
        per_atom = node_e0 + node_e_res
        total_energy = scatter_sum(src=per_atom, index=data.batch, dim=0)

        output = self._compute_properties(total_energy, data, compute_stress)
        output.update({"total_energy": total_energy})

        return output

    def _compute_properties(self, total_energy: torch.Tensor, data, compute_stress=False):
        inputs = [data.pos]
        if hasattr(data, "stress") or compute_stress:
            inputs += [data.displacements]
            compute_stress = True

        outputs = torch.autograd.grad(
            outputs=[total_energy],
            inputs=inputs,
            grad_outputs=[torch.ones_like(total_energy)],
            retain_graph=self.training,
            create_graph=self.training,
        )

        result = {"force": -outputs[0]}
        if compute_stress:
            virial = outputs[1]
            stress = virial / torch.det(data.box).abs().view(-1, 1, 1)
            stress = -torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
            result.update({"stress": stress})

        return result

    def loss(self, pred_data: Data, target_data: Data) -> Dict[str, torch.Tensor]:
        if self.normalize_after_forward:  # This is already done in self.normalize
            loss = {
                "Loss Energy": self.loss_fn(pred_data.total_energy, target_data.total_energy) * self.weights[0],
            }
        else:
            total_e0 = scatter_sum(src=self.e0_mean[pred_data.z], index=pred_data.batch, dim=0)
            na = pred_data.ptr.diff()
            loss = {
                "Loss Energy": self.loss_fn((pred_data.total_energy - total_e0) / na, (target_data.total_energy - total_e0) / na)
                * self.weights[0],
            }

        loss.update({"Loss Forces": self.loss_fn(pred_data.force, target_data.force) * self.weights[1]})

        if hasattr(target_data, "stress"):
            loss.update({"Loss Stress": self.loss_fn(pred_data.stress, target_data.stress) * self.weights[2]})

        return loss

    def metric(self, pred_data: Data, target_data: Data) -> Dict[str, torch.Tensor]:
        na = pred_data.ptr.diff()
        metrics = {
            "MAE Energy": self.mae(pred_data.total_energy / na, target_data.total_energy / na) * 1000,  # meV/atom
        }

        metrics.update(
            {"MAE Forces": self.mae(pred_data.force, target_data.force) * 1000},  # meV/A
        )

        if hasattr(target_data, "stress"):
            metrics.update(
                {"MAE Stress": self.mae(pred_data.stress / na[:, None, None], target_data.stress / na[:, None, None]) * 1000}
            )  # meV/A^3/atom

        return metrics

    def normalize(self, data):
        node_e0 = self.e0_mean[data.z]
        total_e0 = scatter_sum(src=node_e0, index=data.batch, dim=0)
        na = data.ptr.diff()
        data.total_energy = (((data.total_energy - total_e0) / na) - self.atomic_res_total_mean) / self.atomic_res_total_std
        data.force = data.force / self.force_std
        if hasattr(data, "stress"):
            data.stress = (data.stress - self.stress_mean) / self.stress_std
        return data

    def store_constants(self, data):
        dtype = data.total_energy

        z_unique = data.z.unique().sort()[0]
        num_z_types = len(z_unique)
        num_molecules = data.num_graphs

        # Initialize
        one_hot = torch.zeros(len(data.z), num_z_types).to(dtype)
        z_indices = torch.searchsorted(z_unique, data.z)
        one_hot.scatter_(1, z_indices.unsqueeze(1), 1)

        A = torch.zeros(num_molecules, num_z_types).to(dtype)
        A.index_add_(0, data.batch, one_hot)

        e0_mean = torch.zeros(119).to(dtype)

        # Scale sides by number of atoms
        num_atoms_per_molecule = data.ptr.diff()
        mean_energy = data.total_energy / num_atoms_per_molecule
        scaled_A = A / num_atoms_per_molecule.unsqueeze(1)

        # Scale by norm
        column_norms = torch.linalg.norm(scaled_A, dim=0, keepdim=True)
        scaled_A_normalized = scaled_A / column_norms

        # Use gelsd for ill-conditioned matrix
        result = torch.linalg.lstsq(scaled_A_normalized, mean_energy.unsqueeze(1), driver="gelsd")

        e0_mean[z_unique] = result.solution.squeeze() / column_norms.squeeze()

        baseline_node_energy = e0_mean[data.z]
        baseline_total_energy = scatter_sum(baseline_node_energy, index=data.batch, dim=0)
        residual_total_energy = data.total_energy - baseline_total_energy

        return {
            "e0_mean": e0_mean,
            "atomic_res_total_mean": (residual_total_energy / num_atoms_per_molecule).mean(),
            "atomic_res_total_std": (residual_total_energy / num_atoms_per_molecule).std(),
            "force_std": data.force.std(),
            "stress_mean": data.stress.mean(axis=0) if hasattr(data, "stress") else torch.zeros(1, 3, 3),
            "stress_std": data.stress.std(axis=0) if hasattr(data, "stress") else torch.zeros(1, 3, 3),
        }

    @property
    def target_keys(self):
        return ["total_energy", "force", "stress"]

    @property
    def loss_keys(self):
        return ["Loss Energy", "Loss Forces", "Loss Stress"]

    @property
    def metric_keys(self):
        return ["MAE Energy", "MAE Forces", "MAE Stress"]

    @property
    def constants_keys(self):
        return ["z", "total_energy", "force", "stress"]
