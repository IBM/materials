from typing import Optional, Tuple

import torch
from torch import nn
from torch_nl import compute_neighborlist
from torch_nl.geometry import compute_distances
from torch_nl.neighbor_list import compute_cell_shifts


ACT_CLASS_MAPPING = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


class PeriodicDistance(nn.Module):
    """
    Wraps the `torch_nl` package to calculate Periodic Distance using
    PyTorch operations efficiently. Compute the neighbor list for a given cutoff
    with an optional 'skin' buffer to cache results until atoms move
    beyond a threshold.
    Reference: https://github.com/felixmusil/torch_nl
    """

    def __init__(self, cutoff: float = 6.0, skin: Optional[float] = None) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.skin = skin
        self.self_interactions = False

        if skin is not None:
            self.register_buffer("_prev_pos", None)
            self.register_buffer("_prev_box", None)
            self._cached_topo = None

    def forward(self, pos: torch.Tensor, box: torch.Tensor):
        # This method supports one structure only
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        rebuild = True
        if self.skin is not None:
            rebuild = self._cached_topo is None or self._moved_beyond_skin(pos, box)

        if rebuild:
            is_zero = torch.eq(box, 0.0)
            pbc_mask = ~is_zero.all(dim=-1).all(dim=-1)
            pbc = pbc_mask.unsqueeze(-1).repeat(1, 3)

            edge_index, batch_map, shifts_idx = compute_neighborlist(
                self.cutoff, pos, box, pbc, batch, self.self_interactions
            )
            self._cached_topo = (edge_index, shifts_idx, batch_map)
            self._prev_pos = pos.clone()
            self._prev_box = box.clone()

        edge_index, shifts_idx, batch_map = self._cached_topo

        cell_shifts = compute_cell_shifts(box, shifts_idx, batch_map)
        edge_weight = compute_distances(pos, edge_index, cell_shifts)
        edge_vec = -(pos[edge_index[1]] - pos[edge_index[0]] + cell_shifts)

        return edge_index, edge_weight, edge_vec, shifts_idx

    def _moved_beyond_skin(self, pos: torch.Tensor, box: torch.Tensor):
        prev_box = self._prev_box[0]

        if not torch.allclose(box, prev_box, rtol=1e-5, atol=1e-8):  # Box deformation
            return True

        if torch.allclose(box, torch.zeros_like(box)):
            delta = pos - self._prev_pos
            max_disp = torch.linalg.norm(delta, dim=1).max()
            return max_disp.item() > self.skin

        inv_box, _ = torch.linalg.inv_ex(prev_box, check_errors=True)
        delta = pos.unsqueeze(0) - self._prev_pos.unsqueeze(0)
        frac = torch.einsum("ij,bkj->bki", inv_box, delta)
        frac = frac - torch.round(frac)
        delta = torch.einsum("bij,bkj->bki", self._prev_box, frac)
        max_disp = torch.linalg.norm(delta.squeeze(0), dim=1).max()

        return max_disp.item() > self.skin


def get_symmetric_displacement(
    positions: torch.Tensor, box: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # This method supports one structure only
    batch = torch.zeros(positions.size(0), dtype=torch.long, device=positions.device)
    num_graphs = 1

    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    box = box.view(-1, 3, 3)
    box = box + torch.matmul(box, symmetric_displacement)

    return positions, box, displacement
