import os
import tarfile
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from ..datahandler import TemplateDataHandler


class ANI2X(TemplateDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder=root_folder,
            license="CC0",
            download_link="https://zenodo.org/records/10108942/files/ANI-2x-B973c-def2mTZVP.tar.gz",
            n_entries=9643594,
        )

    def _stream_data(self, tar_path: str, n_files_to_skip: int = 0):
        with TemporaryDirectory(dir=self.root_folder) as tmp_dir:
            # Open the tar.gz file with gzip compression mode
            with tarfile.open(tar_path, "r:gz") as tar_file:
                # Extract all files to the temporary directory
                tar_file.extractall(path=tmp_dir)

                hdf5_path = os.path.join(tmp_dir, tar_file.getmembers()[0].name)

                # Open and process the HDF5 file
                with h5py.File(hdf5_path, "r") as h5:
                    counter = 0
                    for num_atoms_group, properties in h5.items():  # Iterate through like a dictionary
                        species = properties["species"][:]
                        coordinates = properties["coordinates"][:]
                        energies = properties["energies"][:]
                        forces = properties["forces"][:]

                        for s, c, e, f in zip(species, coordinates, energies, forces):
                            if counter < n_files_to_skip:
                                counter += 1
                                continue

                            attrs = {
                                "z": torch.from_numpy(s),
                                "pos": torch.from_numpy(c.astype(np.float64)),
                                "total_energy": torch.tensor([e], dtype=torch.float64) * 27.211386246,  # Ha to eV
                                "force": torch.from_numpy(f.astype(np.float64)) * 27.211386246,  # Ha/A to eV/a
                                "box": torch.zeros(1, 3, 3, dtype=torch.float64),
                                "id": f"ANI-2X {num_atoms_group} {counter}",
                            }

                            yield Data(**attrs)
                            counter += 1
