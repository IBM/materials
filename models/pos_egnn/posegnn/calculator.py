import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
from ase.stress import full_3x3_to_voigt_6_stress
from torch_geometric.data.data import Data

from .model import PosEGNN


class PosEGNNCalculator(Calculator):
    def __init__(self, checkpoint: str, device: str, compute_stress: bool = True, **kwargs):
        Calculator.__init__(self, **kwargs)

        checkpoint_dict = torch.load(checkpoint, weights_only=True, map_location=device)

        self.model = PosEGNN(checkpoint_dict["config"])
        self.model.load_state_dict(checkpoint_dict["state_dict"], strict=True)
        self.model.eval()

        self.model.to(device)
        self.model.eval()

        self.implemented_properties = ["energy", "forces"]
        self.implemented_properties += ["stress"] if compute_stress else []
        self.device = device
        self.compute_stress = compute_stress

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        self.results = {}
        data = self._build_data(atoms)
        out = self.model.compute_properties(data, compute_stress=self.compute_stress)

        # Decoder Forward
        self.results = {
            "energy": out["total_energy"].cpu().detach().numpy(),
            "forces": out["force"].cpu().detach().numpy()
        }
        if self.compute_stress:
            self.results.update({
                "stress": full_3x3_to_voigt_6_stress(out["stress"].cpu().detach().numpy())
                })

    def _build_data(self, atoms):
        z = torch.tensor(np.array([atomic_numbers[symbol] for symbol in atoms.symbols]), device=self.device)
        box = torch.tensor(atoms.get_cell().tolist(), device=self.device).unsqueeze(0).float()
        pos = torch.tensor(atoms.get_positions().tolist(), device=self.device).float()
        batch = torch.zeros(len(z), device=self.device).long()
        ptr = torch.zeros(1, device=self.device).long()
        return Data(z=z, pos=pos, box=box, batch=batch, num_graphs=1, ptr=ptr)


def get_invariant_embeddings(self):
    if self.calc is None:
        raise RuntimeError("No calculator is set.")
    else:
        data = self.calc._build_data(self)
        with torch.no_grad():
            embeddings = self.calc.model(data)["embedding_0"][..., 1].squeeze(2)
        return embeddings


Atoms.get_invariant_embeddings = get_invariant_embeddings
