from typing import Any, Dict, Generator

import ijson
from torch import float64, tensor
from torch_geometric.data import Data

from ...constants import SYMBOL_TO_Z
from ..datahandler import TemplateDataHandler


class MPtrj(TemplateDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder=root_folder,
            license="MIT",
            download_link="https://figshare.com/ndownloader/files/41619375",
            n_entries=1580395,
        )

    @staticmethod
    def _read_json_keys_values(filename):
        # Read a JSON file and return a list of key-value pairs
        # MPtrj is too big to load it all at once.
        with open(filename, "r") as file:
            # 'kvitems' will iterate over key-value pairs at the first level
            for key, value in ijson.kvitems(file, "", use_float=True):
                yield key, value

    @staticmethod
    def _parse_frame(frame_data: Dict[str, Any]) -> Dict[str, Any]:
        # Parse a single frame from the JSON file and return a dictionary
        # >> frame_data.keys()
        # dict_keys(['structure', 'uncorrected_total_energy', 'corrected_total_energy', 'energy_per_atom', 'ef_per_atom',
        # 'e_per_atom_relaxed', 'ef_per_atom_relaxed', 'force', 'stress', 'magmom', 'bandgap', 'mp_id'])

        structure = frame_data["structure"]

        sites = structure["sites"]
        z = [SYMBOL_TO_Z[site["label"]] for site in sites]
        occupancy = [site["species"][0]["occu"] for site in sites]
        pos = [site["xyz"] for site in sites]

        lattice = structure["lattice"]
        box = lattice["matrix"]

        total_energy = frame_data["uncorrected_total_energy"]
        force = frame_data["force"]
        magmom = frame_data["magmom"]
        stress = frame_data["stress"]
        bandgap = frame_data["bandgap"]

        parsed_frame = {
            "z": tensor(z),
            "occupancy": tensor(occupancy),
            "pos": tensor(pos, dtype=float64),
            "box": tensor(box, dtype=float64).unsqueeze(0),
            "total_energy": tensor(total_energy, dtype=float64),
            "force": tensor(force, dtype=float64),  # ev/A
            "stress": tensor(stress, dtype=float64).unsqueeze(0) * 6.2415091258833e-4,  # kbar to ev/A3
            "bandgap": tensor(bandgap, dtype=float64) if bandgap is not None else None,
        }

        if magmom is not None:
            parsed_frame["magmom"] = tensor(magmom)

        return parsed_frame

    def _stream_data(self, json_path: str, n_files_to_skip: int = 0) -> Generator[Dict[str, Any], None, None]:
        for mp_id, frames in self._read_json_keys_values(json_path):
            for frame_id, frame_data in frames.items():
                if n_files_to_skip > 0:
                    n_files_to_skip -= 1
                    continue
                data_dict = self._parse_frame(frame_data)
                data_dict.update({"id": f"{mp_id}-{frame_id}"})
                yield Data(**data_dict)
