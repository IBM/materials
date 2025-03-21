from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_nl import compute_neighborlist
from torch_nl.geometry import compute_distances
from torch_nl.neighbor_list import compute_cell_shifts


ACT_CLASS_MAPPING = {"silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}

class BatchedPeriodicDistance(nn.Module):
    """
    Wraps the `torch_nl` package to calculate Periodic Distance using
    PyTorch operations efficiently. Compute the neighbor list for a given cutoff.
    Reference: https://github.com/felixmusil/torch_nl
    """

    def __init__(self, cutoff: float = 5.0) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.self_interactions = False

    def forward(
        self, pos: Tensor, box: Tensor, batch: Optional[Tensor] = None, precomputed_edge_index=None, precomputed_shifts_idx=None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # No batch, single sample
        if batch is None:
            n_atoms = pos.shape[0]
            batch = torch.zeros(n_atoms, device=pos.device, dtype=torch.int64)

        is_zero = torch.eq(box, 0)
        is_not_all_zero = ~is_zero.all(dim=-1).all(dim=-1)
        pbc = is_not_all_zero.unsqueeze(-1).repeat(1, 3)  # We need to change this when dealing with interfaces

        if (precomputed_edge_index is None) or (precomputed_shifts_idx is None):
            edge_index, batch_mapping, shifts_idx = compute_neighborlist(self.cutoff, pos, box, pbc, batch, self.self_interactions)
        else:
            edge_index = precomputed_edge_index
            shifts_idx = precomputed_shifts_idx
            batch_mapping = batch[edge_index[0]]  # NOTE: should be same as edge_index[1]

        cell_shifts = compute_cell_shifts(box, shifts_idx, batch_mapping)
        edge_weight = compute_distances(pos, edge_index, cell_shifts)

        edge_vec = -(pos[edge_index[1]] - pos[edge_index[0]] + cell_shifts)

        # edge_weight and edge_vec should have grad_fn
        return edge_index, edge_weight, edge_vec, shifts_idx


def get_symmetric_displacement(
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.einsum("be,bec->bc", positions, symmetric_displacement[batch])
    box = box.view(-1, 3, 3)
    box = box + torch.matmul(box, symmetric_displacement)

    return positions, box, displacement
