import logging
import warnings
from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
from torch_geometric.data.data import Data
import requests
from pathlib import Path

from .model import GeoditeModule

torch.set_float32_matmul_precision("highest")


class GeoditeCalculator(Calculator):
    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        fidelity: str = "MPtrj",
        precision: str = "32",
        compute_stress: bool = True,
        download_path = ".",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.implemented_properties = ["energy", "forces"]
        self.implemented_properties += ["stress"] if compute_stress else []

        self.device = device = torch.device(device)
        self.fidelity = f"{fidelity}_"
        self.compute_stress = compute_stress

        if checkpoint == "MP":
            url = "https://huggingface.co/ibm-research/materials.geodite/resolve/main/Geodite-MP.ckpt"

            download_path = Path(download_path)  # change this to your folder
            download_path.mkdir(parents=True, exist_ok=True)

            file_path = download_path / "Geodite-MP.ckpt"

            if file_path.exists():
                print(f"File already exists: {file_path}")
            else:
                print(f"Downloading {url} to {file_path}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"Download complete: {file_path}")

            checkpoint = file_path

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GeoditeModule.load_from_checkpoint(checkpoint, strict=False, map_location=device, weights_only=True)
        model.SnapshotDecoder.set_context_state(self.fidelity)
        try:
            self.model = torch.compile(model, mode="default")
            logging.info("Loading compiled model.")
        except Exception as e:
            logging.warning(f"Compiling failed for {checkpoint!r}: {e}. \nLoading checkpoint normally.")
            self.model = model

        if precision == "64":
            self.model.double()
            self.np_dtype = np.float64
            self.torch_dtype = torch.float64
        elif precision == "32":
            self.model.float()
            self.np_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            assert False

        self.decoder_key = "SnapshotDecoder"

        self.model.encoder.to(device)
        self.model.SnapshotDecoder.to(device)
        self.model.SnapshotDecoder.e0 = self.model.SnapshotDecoder.e0.to(device)
        self.model.to(device)
        self.model.eval()

        self.default_attrs = {
            att: torch.tensor(-1, dtype=torch.long)
            for att in [
                "z",
                "cutoff_edge_index",
                "cutoff_edge_distance",
                "cutoff_edge_vec",
                "batch",
                "embedding_0",
                "pos",
                "displacements",
                "box",
                "num_graphs",
                "cutoff_shifts_idx",
                "embedding_1",
            ]
        }

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        data = self._build_data(atoms)

        out = self.model(data)

        # Decoder Forward
        out = self.model.SnapshotDecoder(out, compute_stress=self.compute_stress)

        self.results = {"energy": out["total_energy"].cpu().detach().item(), "forces": out["force"].cpu().detach().numpy()}

        if self.compute_stress:
            self.results.update({"stress": -out["stress"].squeeze().cpu().detach().numpy()})

    def _build_data(self, atoms):
        np_dtype, torch_dtype = self.np_dtype, self.torch_dtype
        z = torch.tensor([atomic_numbers[symbol] for symbol in atoms.symbols], dtype=torch.long)
        if atoms.pbc.all():
            cell = atoms.get_cell().array.astype(np_dtype)
            box = torch.from_numpy(cell).to(dtype=torch_dtype).unsqueeze(0)
        else:
            box = torch.zeros(3, 3, dtype=torch_dtype)

        positions = atoms.get_positions().astype(np_dtype)
        pos = torch.from_numpy(positions).to(dtype=torch_dtype)

        batch = torch.zeros(len(z), dtype=torch.long)
        num_graphs = 1
        ptr = torch.zeros(1, dtype=torch.long)

        attrs = {
            "z": z,
            "pos": pos,
            "box": box,
            "batch": batch,
            "num_graphs": num_graphs,
            "ptr": ptr,
            "fidelity": [self.fidelity],
        }

        attrs = self.default_attrs.copy() | attrs

        return Data(**attrs).to(self.device)
