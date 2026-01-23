import gzip
from typing import Any, Dict, Generator

import ijson
from torch import float64, tensor
from torch_geometric.data import Data

from ...constants import SYMBOL_TO_Z
from ..datahandler import TemplateDataHandler


class MatPES_r2SCAN(TemplateDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder=root_folder,
            license="BSD-3-Clause",
            download_link="https://s3.us-east-1.amazonaws.com/materialsproject-contribs/MatPES_2025_1/MatPES-R2SCAN-2025.1.json.gz",
            n_entries=387897,
        )

    @staticmethod
    def _parse_frame(frame_data: Dict[str, Any]) -> Dict[str, Any]:
        # {'builder_meta': {'emmet_version': '0.84.6rc3.dev21+g7a6aab3b', 'pymatgen_version': '2025.1.9', 'run_id': None,
        # 'batch_id': None, 'database_version': None, 'build_date': '2025-03-20 14:56:01.976913+00:00', 'license': None},
        # 'nsites': 2, 'elements': ['N', 'Zn'], 'nelements': 2, 'composition': {'Zn': 1.0, 'N': 1.0}, 'composition_reduced':
        # {'Zn': 1.0, 'N': 1.0}, 'formula_pretty': 'ZnN', 'formula_anonymous': 'AB', 'chemsys': 'N-Zn', 'volume':
        # 22.207063878330768, 'density': 5.938329950242052, 'density_atomic': 11.103531939165384, 'symmetry': {'crystal_system':
        # 'Triclinic', 'symbol': 'P1', 'number': 1, 'point_group': '1', 'symprec': 0.1, 'angle_tolerance': 5.0, 'version':
        # '2.5.0'}, 'structure': {'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0.0, 'lattice': {'matrix':
        # [[2.908514648819814, 0.0, 0.0], [1.018584372515761, 2.888569425880174, 0.0], [1.7708643492698883, 1.6736874405713007,
        # 2.6432429113961335]], 'pbc': [True, True, True], 'a': 2.908514648819814, 'b': 3.06289853767033, 'c': 3.5949858526685587,
        # 'alpha': 52.923694372643624, 'beta': 60.488854571291554, 'gamma': 70.57603020593486, 'volume': 22.207063878330768},
        # 'properties': {}, 'sites': [{'species': [{'element': 'Zn', 'occu': 1}], 'abc': [0.0573317012748689, 0.9788136382207782,
        # 0.0008728250415118], 'properties': {'magmom': 0.0}, 'label': 'Zn', 'xyz': [1.1653000232458615, 2.8288319853088724,
        # 0.0023070886038651013]}, {'species': [{'element': 'N', 'occu': 1}], 'abc': [0.8235724038271997, 0.7566069221567204,
        # 0.7231268801513266], 'properties': {'magmom': 0.001}, 'label': 'N', 'xyz': [4.446599999999986, 3.395799999999987,
        # 1.9113999999999955]}]}, 'energy': -16.83287187, 'forces': [[0.52931582, 0.11320229, 0.33051735], [-0.52931582,
        # -0.11320229, -0.33051735]], 'stress': [51.59958795, 46.26469169, -41.31292119, -68.39966876, -8.26649824, 44.22463946],
        # 'matpes_id': 'matpes-20240214_999485_2', 'bandgap': 0.0, 'functional': 'r2SCAN', 'formation_energy_per_atom': None,
        # 'cohesive_energy_per_atom': -2.4813380500000015, 'abs_forces': [0.6342174031154766, 0.6342174031154766], 'bader_charges':
        # None, 'bader_magmoms': None, 'provenance': {'original_mp_id': 'mp-999485', 'materials_project_version': 'v2022.10.28',
        # 'md_ensemble': 'NpT', 'md_temperature': 300.0, 'md_pressure': 1.0, 'md_step': 8, 'mlip_name': 'M3GNet-MP-2021.2.8-DIRECT'
        # }}

        structure = frame_data["structure"]

        sites = structure["sites"]
        z = [SYMBOL_TO_Z[site["label"]] for site in sites]
        occupancy = [site["species"][0]["occu"] for site in sites]
        pos = [site["xyz"] for site in sites]

        lattice = structure["lattice"]
        box = lattice["matrix"]

        total_energy = frame_data["energy"]
        force = frame_data["forces"]
        stress_voigt = frame_data["stress"]
        bandgap = frame_data["bandgap"]

        stress = tensor(
            [
                [
                    [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                    [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                    [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
                ]
            ],
            dtype=float64,
        )

        parsed_frame = {
            "z": tensor(z),
            "occupancy": tensor(occupancy),
            "pos": tensor(pos, dtype=float64),
            "box": tensor(box, dtype=float64).unsqueeze(0),
            "total_energy": tensor(total_energy, dtype=float64),  # ev
            "force": tensor(force, dtype=float64),  # ev/A
            "stress": stress * 6.2415091258833e-3,  # GPa to ev/A3
            "bandgap": tensor(bandgap, dtype=float64) if bandgap is not None else None,
            "label": "TRAJECTORY",
        }

        return parsed_frame

    def _stream_data(self, json_path: str, n_files_to_skip: int = 0) -> Generator[Dict[str, Any], None, None]:
        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            for idx, frame in enumerate(ijson.items(f, "item")):
                if idx < n_files_to_skip:
                    continue
                data_dict = self._parse_frame(frame)
                data_dict.update({"id": frame["matpes_id"]})
                yield Data(**data_dict)
