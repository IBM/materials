import csv
import tarfile
from os.path import join, split
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator

import numpy as np
import torch
from torch_geometric.data import Data

from ..datahandler import TemplateDataHandler


class RevMD17(TemplateDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder=root_folder,
            license="CC0",
            download_link="https://archive.materialscloud.org/record/file?record_id=466&filename=rmd17.tar.bz2",
            n_entries=20000,
        )

    @staticmethod
    def extract_and_read_indices(split_path, tar_file, dir):
        tmp_split_path = join(dir, split(split_path)[-1])
        with open(tmp_split_path, "wb") as f:
            f.write(tar_file.extractfile(split_path).read())
        with open(tmp_split_path) as f:
            return [int(idx[0]) for idx in csv.reader(f)]

    def _stream_data(self, tar_path: str, n_files_to_skip: int = 0) -> Generator[Dict[str, Any], None, None]:
        with TemporaryDirectory(dir=self.root_folder) as tmp_dir, tarfile.open(tar_path, "r:bz2") as tar_file:
            # NOTE: This dataset comes with 5 different splits. I was not sure which one to select.
            split_id = 1
            # NOTE: If there is no validation set from the original dataset, what should we use?
            train_idxs = self.extract_and_read_indices(f"rmd17/splits/index_train_0{split_id}.csv", tar_file, tmp_dir)
            val_idxs = self.extract_and_read_indices(f"rmd17/splits/index_test_0{split_id}.csv", tar_file, tmp_dir)
            idxs_map = {"train": train_idxs, "val": val_idxs}

            trajectories = [x for x in tar_file.getnames() if x.endswith("npz")]
            trajectories, n_files_to_skip = self._skip_processed_trajectories(trajectories, n_files_to_skip, idxs_map)
            for traj_path in trajectories:
                tmp_traj_path = join(tmp_dir, split(traj_path)[-1])

                with open(tmp_traj_path, "wb") as f:
                    f.write(tar_file.extractfile(traj_path).read())
                data = np.load(tmp_traj_path)

                name = traj_path.split("_")[-1][:-4]
                # if "benzene" in name:  # Select one molecule only
                z = torch.from_numpy(data["nuclear_charges"]).long()
                box = torch.zeros(1, 3, 3, dtype=torch.float64)  # Zero box means that it does not have a box at all.
                positions = torch.from_numpy(data["coords"].astype(np.float64))
                forces = torch.from_numpy(data["forces"].astype(np.float64))
                for split_type, idxs in idxs_map.items():
                    for idx in idxs:
                        if n_files_to_skip > 0:
                            n_files_to_skip -= 1
                            continue
                        attrs = {
                            "z": z,
                            "pos": positions[idx],
                            "total_energy": torch.tensor([data["energies"][idx]], dtype=torch.float64) * 0.043364103900593226,
                            "force": forces[idx] * 0.043364103900593226,
                            "split": split_type,
                            "box": box,
                            "id": f"revised {name} - {idx}",
                        }
                        yield Data(**attrs)

    @staticmethod
    def _skip_processed_trajectories(trajectories, n_files_to_skip, idxs_map):
        trajectories_to_skip = n_files_to_skip // (len(idxs_map["train"]) + len(idxs_map["val"]))
        n_files_to_skip -= trajectories_to_skip * (len(idxs_map["train"]) + len(idxs_map["val"]))
        trajectories = trajectories[trajectories_to_skip:]
        return trajectories, n_files_to_skip
