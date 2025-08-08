from typing import Dict

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_scatter import scatter
from torchmetrics import R2Score

from .._base_decoder import AbstractDecoder
from ..readouts import MultipleAggregationGlobalReadout, NodeInvariantReadout


class ScalarRegressionDecoder(AbstractDecoder):
    def __init__(self, in_channels, num_residues, hidden_channels, activation, **kwargs):
        super().__init__(in_channels, num_residues, hidden_channels, activation, **kwargs)

        self.normalize_after_forward = True

        self.readout = MultipleAggregationGlobalReadout(in_channels, num_residues, hidden_channels, 1, activation)

        self.loss_fn = nn.HuberLoss(delta=0.01)
        self.mae = nn.L1Loss()
        self.r2 = R2Score()

    def forward(self, data: Data) -> Dict[str, Tensor]:
        pred = self.readout(data.embedding_0, data.batch) * self.target_scale + self.target_shift
        return {
            "target": pred,
        }

    def loss(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        target = target_data.target
        pred = pred_data.target

        loss = {
            "Loss": self.loss_fn(pred, target),
        }

        return loss

    def metric(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        target = target_data.target
        pred = pred_data.target

        return {
            "MAE": self.mae(pred, target),
            "R2": self.r2(pred, target) if len(pred) > 1 else 0,
        }

    @property
    def target_keys(self):
        return ["target"]

    @property
    def loss_keys(self):
        return ["Loss"]

    @property
    def metric_keys(self):
        return ["MAE", "R2"]

    def normalize(self, data):
        data.target = (data.target - self.target_shift) / self.target_scale
        return data

    def unnormalize(self, data):
        data.target = data.target * self.target_scale + self.target_shift
        return data

    def store_constants(self, data):
        prop = data.target

        shift = prop.median()

        # Robust version of std
        scale = prop.quantile(1 - 0.1587) - prop.quantile(0.1587)

        return {
            "target_shift": shift,
            "target_scale": scale,
        }

    @property
    def constants_keys(self):
        return ["target"]


class AtomicContributionDecoder(AbstractDecoder):
    def __init__(self, in_channels, num_residues, hidden_channels, activation, **kwargs):
        super().__init__(in_channels, num_residues, hidden_channels, activation, **kwargs)

        self.normalize_after_forward = True

        self.readout = NodeInvariantReadout(in_channels, num_residues, hidden_channels, 1, activation)

        self.mae = nn.L1Loss()
        self.loss_fn = nn.HuberLoss(delta=0.01)

    def forward(self, data: Data) -> torch.Tensor:
        embedding = data.embedding_0

        atomic_contributions = self.target_atomic_contributions
        total_residual_mean_per_atom = self.target_total_residual_mean_per_atom
        total_residual_std_per_atom = self.target_total_residual_std_per_atom
        atomic_residual_contribution = self.readout(embedding)

        atomic_residual_contribution = atomic_residual_contribution * total_residual_std_per_atom + total_residual_mean_per_atom
        total_residual_contribution = scatter(src=atomic_residual_contribution, index=data.batch, dim=0, reduce="sum")

        atomic_0_contribution = atomic_contributions[data.z]
        total_0_contribution = scatter(src=atomic_0_contribution, index=data.batch, dim=0, reduce="sum")

        total_contribution = total_0_contribution + total_residual_contribution

        output = {"target": total_contribution, "z": data.z, "batch": data.batch, "ptr": data.ptr}

        return output

    def loss(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        target = target_data.target
        pred = pred_data.target

        na = pred_data.ptr.diff()

        loss = {
            "Loss ": self.loss_fn(pred / na, target / na),
        }

        return loss

    def metric(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        target = target_data.target
        pred = pred_data.target

        na = pred_data.ptr.diff()

        return {
            "MAE": self.mae(pred, target),
            "MAE (per atom)": self.mae(pred / na, target / na),
        }

    @property
    def loss_keys(self):
        return ["Loss"]

    @property
    def metric_keys(self):
        return ["MAE", "R2"]

    def normalize(self, data):
        data.target = (data.target - self.target_mean) / self.target_std
        return data

    def unnormalize(self, data):
        data.target = data.target * self.target_std + self.target_mean
        return data

    @property
    def constants_keys(self):
        return ["target"]

    @property
    def target_keys(self):
        return ["target"]

    def store_constants(self, data):
        prop = data.target

        z_unique = data.z.unique().sort()[0]
        num_z_types = len(z_unique)
        num_molecules = data.num_graphs

        # Initialize
        one_hot = torch.zeros(len(data.z), num_z_types)
        z_indices = torch.searchsorted(z_unique, data.z)
        one_hot.scatter_(1, z_indices.unsqueeze(1), 1)

        A = torch.zeros(num_molecules, num_z_types)
        A.index_add_(0, data.batch, one_hot)

        atomic_contributions = torch.zeros(119)

        # Scale sides by number of atoms
        num_atoms_per_molecule = data.ptr.diff()
        mean_prop = prop / num_atoms_per_molecule
        scaled_A = A / num_atoms_per_molecule.unsqueeze(1)

        # Scale by norm
        column_norms = torch.linalg.norm(scaled_A, dim=0, keepdim=True)
        scaled_A_normalized = scaled_A / column_norms

        # Use gelsd for ill-conditioned matrix
        result = torch.linalg.lstsq(scaled_A_normalized, mean_prop.unsqueeze(1), driver="gelsd")

        atomic_contributions[z_unique] = result.solution.squeeze() / column_norms.squeeze()

        baseline_node_prop = atomic_contributions[data.z]
        baseline_total_prop = scatter(baseline_node_prop, index=data.batch, dim=0, reduce="sum")
        residual_total_prop = prop - baseline_total_prop

        return {
            "target_atomic_contributions": atomic_contributions,
            "target_total_residual_mean_per_atom": (residual_total_prop / num_atoms_per_molecule).mean(),
            "target_total_residual_std_per_atom": (residual_total_prop / num_atoms_per_molecule).std(),
            "target_mean": prop.mean(),
            "target_std": prop.std(),
        }
