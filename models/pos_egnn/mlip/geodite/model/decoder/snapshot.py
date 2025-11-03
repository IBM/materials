from typing import Dict, List, Optional

import torch
from ase.data import covalent_radii
from torch import Tensor, nn
from torch.jit import annotate
from torch_geometric.data import Data

from geodite.utils import DataInput

from ...utils.graph import scatter
from ._base_decoder import AbstractDecoder


# NOTE: Adapted from MACE's ZBLBasis
class ZBLPotential(nn.Module):
    def __init__(self, p: int = 6):
        super().__init__()
        self.p = p

        self.a_factor = nn.Parameter(torch.tensor(0.4546, dtype=torch.get_default_dtype()))
        self.Z_power = nn.Parameter(torch.tensor(0.0842, dtype=torch.get_default_dtype()))
        self.screen_coefs = nn.Parameter(torch.tensor([0.1663, 0.1663, 0.1663, 0.1663], dtype=torch.get_default_dtype()))
        self.screen_exps = nn.Parameter(torch.tensor([0.0354, 0.0354, 0.0354, 0.0354], dtype=torch.get_default_dtype()))

        self.register_buffer("covalent_radii", torch.tensor(covalent_radii, dtype=torch.get_default_dtype()))

    def forward(self, z: torch.Tensor, edge_distance: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index[0], edge_index[1]
        Z_u = z[sender]
        Z_v = z[receiver]

        a = self.a_factor * 0.529 / (Z_u**self.Z_power + Z_v**self.Z_power)
        r_over_a = edge_distance / a

        c = self.screen_coefs.view(1, 4)
        d = self.screen_exps.view(1, 4)
        x = r_over_a.unsqueeze(-1)
        phi = torch.sum(c * torch.exp(-d * x), dim=-1)

        v_edges = (14.3996 * Z_u * Z_v) / edge_distance * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        r_over_r_max = edge_distance / r_max
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * r_over_r_max**self.p
            + self.p * (self.p + 2.0) * r_over_r_max ** (self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * r_over_r_max ** (self.p + 2)
        ) * (r_over_r_max < 1)
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter(v_edges, receiver, dim=0, dim_size=z.size(0))
        return V_ZBL.squeeze(-1)


class NodeInvariantReadout(nn.Module):
    def __init__(self, in_channels: int, num_residues: int, out_channels: int, **kwargs):
        super().__init__()

        self.linears = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(num_residues)])

    def forward(self, embedding_0: Tensor) -> Tensor:
        layer_outputs = embedding_0.squeeze(2)

        processed = []
        for i, lin in enumerate(self.linears):
            processed.append(lin(layer_outputs[:, :, i]))

        result = torch.stack(processed, dim=0)
        return result


class SnapshotDecoder(AbstractDecoder):
    def __init__(self, in_channels: int, num_residues: int, **kwargs):
        super().__init__(in_channels, num_residues, **kwargs)

        self.weights = torch.tensor(kwargs.get("snapshot_weights", [1.0, 1.0, 1.0]))
        self.normalize_after_forward = True

        self.readout = NodeInvariantReadout(in_channels, num_residues, 1)

        self.repulsion_term = ZBLPotential(p=5)

        self.mae = nn.L1Loss()
        self.loss_fn = nn.HuberLoss(delta=0.01)

        self.e0 = torch.tensor(0)
        self.e_scale = torch.tensor(0)
        self.e_shift = torch.tensor(0)
        self.force_std = torch.tensor(0)
        self.stress_mean = torch.tensor(0)
        self.stress_std = torch.tensor(0)

    def forward(self, data: DataInput, compute_stress: bool = False):
        node_e0 = self.e0[data.z]

        node_e_res_per_layer = self.readout(data.embedding_0)
        node_e_res = node_e_res_per_layer.sum(dim=0).squeeze(-1)

        node_e_repulsion = self.repulsion_term(data.z, data.cutoff_edge_distance, data.cutoff_edge_index)

        per_atom = node_e0 + self.shift_scale(node_e_res + node_e_repulsion)
        total_energy = scatter(src=per_atom, index=data.batch, dim=0)

        output = self._compute_properties(total_energy, data, compute_stress)
        output.update({"total_energy": total_energy})

        return output

    def shift_scale(self, node_energy):
        return node_energy * self.e_scale + self.e_shift

    def _compute_properties(self, total_energy: torch.Tensor, data: DataInput, compute_stress: bool = False):
        inputs = [data.pos]
        if hasattr(data, "stress") or compute_stress:
            inputs += [data.displacements]
            compute_stress = True

        grad_outputs = annotate(Optional[List[Optional[Tensor]]], [torch.ones_like(total_energy)])

        outputs = torch.autograd.grad(
            outputs=[total_energy],
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=self.training,
            create_graph=self.training,
        )

        force = -outputs[0]
        assert force is not None

        result = {"force": force}
        if compute_stress:
            virial = outputs[1]
            assert virial is not None
            stress = virial / torch.det(data.box).abs().view(-1, 1, 1)
            stress = -torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
            result.update({"stress": stress})

        return result

    def loss(self, pred_data: Data, target_data: Data) -> Dict[str, torch.Tensor]:
        loss = {
            "Loss Energy": self.loss_fn(pred_data.total_energy, target_data.total_energy) * self.weights[0],
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
        node_e0 = self.e0[data.z]
        total_e0 = scatter(src=node_e0, index=data.batch, dim=0)
        na = data.ptr.diff()
        data.total_energy = (((data.total_energy - total_e0) / na) - self.e_shift) / self.e_scale
        data.force = data.force / self.force_std
        if hasattr(data, "stress"):
            data.stress = (data.stress - self.stress_mean) / self.stress_std
        return data

    def store_constants(self, data):
        # NOTE: Using this is not ideal. It is recommended to provide E0s through a yaml
        # since linear regression might not be a good approximation
        # NOT USED IN GEODITE-MP.
        dtype = torch.float64
        device = data.z.device

        z_unique, _ = torch.sort(data.z.unique())
        num_z_types = z_unique.numel()
        num_molecules = data.num_graphs

        one_hot = torch.zeros(len(data.z), num_z_types, device=device, dtype=dtype)
        z_indices = torch.searchsorted(z_unique, data.z)
        one_hot.scatter_(1, z_indices.unsqueeze(1), 1.0)

        A = torch.zeros(num_molecules, num_z_types, device=device, dtype=dtype)
        A.index_add_(0, data.batch, one_hot)

        e0 = torch.zeros(119, device=device, dtype=dtype)

        num_atoms_per_molecule = data.ptr.diff().to(device=device, dtype=dtype)
        total_energy = data.total_energy.to(device=device, dtype=dtype)
        mean_energy = total_energy / num_atoms_per_molecule

        scaled_A = A / num_atoms_per_molecule.unsqueeze(1)

        column_norms = torch.linalg.norm(scaled_A, dim=0, keepdim=True)
        scaled_A_normalized = scaled_A / column_norms

        result = torch.linalg.lstsq(scaled_A_normalized, mean_energy.unsqueeze(1), driver="gelsd")

        e0[z_unique] = result.solution.squeeze() / column_norms.squeeze()

        baseline_node_energy = e0[data.z]
        baseline_total_energy = scatter(baseline_node_energy, index=data.batch, dim=0)
        residual_total_energy = total_energy - baseline_total_energy

        return {
            "e0": e0,
            "e_shift": (residual_total_energy / num_atoms_per_molecule).mean(),
            "e_scale": (residual_total_energy / num_atoms_per_molecule).std(),
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
