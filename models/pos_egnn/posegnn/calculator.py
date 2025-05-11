from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers

from .model import PosEGNN


class PosEGNNCalculator(Calculator):
    def __init__(
        self,
        checkpoint: str,
        device: str,
        compute_stress: bool = True,
        compile: bool = True,
        skin: Optional[float] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        checkpoint_dict = torch.load(checkpoint, weights_only=True, map_location=device)

        self.model = PosEGNN(checkpoint_dict["config"], skin=skin)
        self.model.load_state_dict(checkpoint_dict["state_dict"], strict=True)
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.compute_stress = compute_stress

        self.implemented_properties = ["energy", "forces"] + (
            ["stress"] if compute_stress else []
        )

        if compile:
            print("Using torch.compile")
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=True
            )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)

        z, pos, box = self._build_tensors(atoms)

        out = self.model.compute_properties(
            z, pos, box, compute_stress=self.compute_stress
        )

        self.results = {
            "energy": out["total_energy"].cpu().detach().numpy(),
            "forces": out["force"].cpu().detach().numpy(),
        }

        if self.compute_stress:
            self.results.update(
                {"stress": -out["stress"].squeeze().cpu().detach().numpy()}
            )

    def _build_tensors(self, atoms: Atoms):
        atomic_nums = np.array([atomic_numbers[symbol] for symbol in atoms.symbols])

        z = torch.tensor(atomic_nums, device=self.device)
        box = (
            torch.tensor(atoms.get_cell().tolist(), device=self.device)
            .unsqueeze(0)
            .float()
        )
        pos = torch.tensor(atoms.get_positions().tolist(), device=self.device).float()

        return z, pos, box


def get_invariant_embeddings(self):
    if self.calc is None:
        raise RuntimeError("No calculator is set.")
    else:
        z, pos, box = self.calc._build_tensors(self)
        with torch.no_grad():
            embeddings, _, _ = self.calc.model(z, pos, box)["embedding_0"][
                ..., -1
            ].squeeze(2)
        return embeddings


Atoms.get_invariant_embeddings = get_invariant_embeddings
