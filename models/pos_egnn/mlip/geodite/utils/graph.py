from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_nl import compute_neighborlist
from torch_nl.geometry import compute_distances
from torch_nl.neighbor_list import compute_cell_shifts


class BatchedPeriodicDistance(nn.Module):
    """
    Wraps the `torch_nl` package to calculate Periodic Distance using
    PyTorch operations efficiently. Compute the neighbor list for a given cutoff.
    Reference: https://github.com/felixmusil/torch_nl
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(
        self,
        pos: Tensor,
        box: Tensor,
        batch: Optional[Tensor] = None,
        precomputed_edge_index: Optional[Tensor] = None,
        precomputed_shifts_idx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # No batch, single sample
        if batch is None:
            n_atoms = pos.shape[0]
            batch = torch.zeros(n_atoms, device=pos.device, dtype=torch.int64)

        pbc = box.any(-1).any(-1).unsqueeze(-1).repeat(1, 3)
        pos_wrapped = self._wrap_positions(pos, box, batch, pbc)

        if precomputed_edge_index is None or precomputed_shifts_idx is None:
            edge_index, batch_mapping, shifts_idx = compute_neighborlist(self.cutoff, pos_wrapped, box, pbc, batch, self_interaction=False)
        else:
            edge_index = precomputed_edge_index
            shifts_idx = precomputed_shifts_idx
            batch_mapping = batch[edge_index[0]]

        cell_shifts = compute_cell_shifts(box, shifts_idx, batch_mapping)
        edge_weight = compute_distances(pos_wrapped, edge_index, cell_shifts)
        edge_vec = -(pos_wrapped[edge_index[1]] - pos_wrapped[edge_index[0]] + cell_shifts)

        # edge_weight and edge_vec should have grad_fn
        return edge_index, edge_weight, edge_vec, shifts_idx

    @staticmethod
    def _wrap_positions(pos: Tensor, box: Tensor, batch: Tensor, pbc: Tensor) -> Tensor:
        cell = box[batch]
        cell_is_zero = (cell == 0).all(dim=(-2, -1))

        cell_inv = torch.linalg.pinv(cell)
        # Convert to fractional coordinates
        frac_coords = torch.where(cell_is_zero.unsqueeze(-1), pos, torch.einsum("ni,nij->nj", pos, cell_inv))

        pbc_mask = pbc[batch]
        frac_coords_wrapped = torch.where(
            pbc_mask,
            frac_coords % 1.0,  # Wrap to [0, 1) for periodic dims
            frac_coords,
        )

        # Convert back to Cartesian coordinates
        pos_wrapped = torch.where(cell_is_zero.unsqueeze(-1), frac_coords_wrapped, torch.einsum("ni,nij->nj", frac_coords_wrapped, cell))
        return pos_wrapped


def get_symmetric_displacement(
    positions: torch.Tensor,
    box: torch.Tensor,
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


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum" or reduce == "add"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
