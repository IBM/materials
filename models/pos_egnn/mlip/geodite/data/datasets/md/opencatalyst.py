import platform
import tarfile
import zlib
from abc import abstractmethod
from io import BytesIO
from json import loads
from os import makedirs
from os.path import basename, dirname, exists, isdir, join, split
from pickle import load
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, List

import lmdb
import torch
import wget
from torch import tensor
from torch_geometric.data import Data

from ..datahandler import TemplateDataHandler


class OpenCatalystDataHandler(TemplateDataHandler):
    def _stream_data(self, tar_paths: List[str] | str, n_files_to_skip: int = 0) -> Generator[Dict[str, Any], None, None]:
        lmdb_kwargs = {"subdir": False, "readonly": True, "lock": False, "readahead": True, "meminit": False, "max_readers": 10}

        if isinstance(tar_paths, str):
            tar_paths = [tar_paths]

        temp_dir_location = self._set_ram_disk_if_available()
        total_keys = 0

        # Process each tar file in sequence
        for tar_path in tar_paths:
            print(f"Reading {tar_path}")
            with TemporaryDirectory(dir=temp_dir_location) as tmp_dir, tarfile.open(tar_path, "r|gz") as tar_file:
                for tar_member in tar_file:
                    if not tar_member.name.endswith("lmdb"):
                        continue

                    db_path = tar_member.name

                    if hasattr(self, "file_to_split"):
                        split_type = self.file_to_split[basename(tar_path)]
                    else:
                        split_type = basename(dirname(db_path))

                    # ODAC has its own LMDB for each trajectory
                    extracted_db_path = join(tmp_dir, split(db_path)[-1])
                    with open(extracted_db_path, "wb") as f:
                        f.write(tar_file.extractfile(tar_member).read())

                    with lmdb.open(extracted_db_path, **lmdb_kwargs) as env:
                        with env.begin() as txn:
                            n_entries = txn.stat()["entries"]
                            # Check if we need to skip this entire LMDB
                            if total_keys + n_entries <= n_files_to_skip:
                                total_keys += n_entries
                                continue
                            # If we need to start processing from somewhere in this LMDB
                            elif total_keys < n_files_to_skip < total_keys + n_entries:
                                snapshot_keys = [key for key, _ in txn.cursor()]
                                remaining_skip = n_files_to_skip - total_keys
                                snapshot_keys = snapshot_keys[remaining_skip:]
                                total_keys += n_entries
                            # If we're past the skip point, process all entries
                            else:
                                snapshot_keys = [key for key, _ in txn.cursor()]
                                total_keys += n_entries

                        # Remove the b'length' key from snapshot_keys
                        if b"length" in snapshot_keys:
                            snapshot_keys.remove(b"length")

                        # Remove the b'nextid' key from snapshot_keys
                        if b"nextid" in snapshot_keys:
                            snapshot_keys.remove(b"nextid")

                        for key in snapshot_keys:
                            with env.begin() as txn:
                                bin_data = txn.get(key)

                                # Check if data is compressed
                                if any(bin_data.startswith(header) for header in [b"x\x01", b"x\x5e", b"x\x9c", b"x\xda"]):
                                    ocp_data = loads(zlib.decompress(txn.get(key)))
                                else:
                                    ocp_data = load(BytesIO(bin_data))

                            data_dict, id = self.data_to_dict(ocp_data, split_type)
                            data_dict.update({"id": id})
                            yield Data(**data_dict)

    def _set_ram_disk_if_available(self):
        ram_disk_folder = None
        if platform.system() == "Linux" and isdir("/dev/shm"):
            ram_disk_folder = "/dev/shm"
        elif platform.system() == "Darwin" and isdir("/Volumes/ramdisk"):
            ram_disk_folder = "/Volumes/ramdisk"
        if ram_disk_folder:
            print("Using ramdisk at " + ram_disk_folder)
            return ram_disk_folder
        return self.root_folder

    @abstractmethod
    def data_to_dict(self, data, split_type):
        pass


class OC22_S2EF(OpenCatalystDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder,
            license="CC-BY-4.0",
            download_link="https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/s2ef_total_train_val_test_lmdbs.tar.gz",
            n_entries=9854709,
        )

    def data_to_dict(self, data, split_type):
        # Example of train ocp_data from S2EF:
        # Data(y=-695.13171908, pos=[97, 3], cell=[1, 3, 3], atomic_numbers=[97], natoms=97, force=[97, 3], fixed=[97], tags=[97],
        # nads=1, sid=11385, fid=146, id='29_27578', oc22=1)

        # Split can be `train`, `val_id`, `val_ood`, `test_id`, `test_ood`

        # OC22 uses an older version of PyG, so we have to do the following, according to
        # https://github.com/pyg-team/pytorch_geometric/discussions/7241
        data = Data.from_dict(data.__dict__)

        data_kwargs = {
            "z": data.atomic_numbers,
            "pos": data.pos,
            "box": data.cell,
            "fixed": data.fixed,
            "n_adsorbate": data.nads,
            "tags": data.tags,
            "split": split_type,
            "label": "TRAJECTORY",
        }

        if "test" in split_type:
            # `test` comes without `y` and `force`
            data_kwargs["split"] = "evaluation"
        else:
            data_kwargs["adsorption_energy"] = data.y
            data_kwargs["force"] = data.force

        return data_kwargs, f"{data.sid}-{data.fid}-{data.id}-{data.oc22}"


class ODAC23_S2EF(OpenCatalystDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder,
            license="CC-BY-4.0",
            download_link="https://dl.fbaipublicfiles.com/large_objects/dac/datasets/odac23_s2ef.tar.gz",
            n_entries=38983748,
        )

    def data_to_dict(self, data, split_type):
        # Example of train ocp_data from S2EF:
        # Data(pos=[261, 3], cell=[1, 3, 3], atomic_numbers=[261], natoms=261, tags=[261], y=-0.712270970000418, force=[261, 3],
        # fixed=[261], raw_y=-1750.6496627800002, nco2=1, nh2o=2, nads=3, sid=[1], name='AMILUE_0.08_0_w_CO2_2H2O_1', fid=[1],
        # supercell=[3], oms=False, defective=True)

        data_kwargs = {
            "z": data.atomic_numbers,
            "pos": data.pos,
            "box": data.cell,
            "fixed": data.fixed,
            "total_energy": data.raw_y,
            "n_CO2": data.nco2,
            "n_H2O": data.nh2o,
            "n_adsorbate": data.nads,
            "tags": data.tags,
            "oms": data.oms,
            "supercell": data.supercell,
            "defective": data.defective,
            "split": split_type,
            "label": "TRAJECTORY",
        }

        if split_type in ["ood_big_no_targets", "ood_linker_no_targets", "ood_linker_topology_no_targets", "ood_topology_no_targets"]:
            # `test` comes without `pos_relaxed`, `raw_y` and `y_relaxed`
            data_kwargs["split"] = "evaluation"
        else:
            data_kwargs["adsorption_energy"] = data.y
            data_kwargs["force"] = data.force

        return data_kwargs, f"{data.sid.item()}-{data.fid.item()}-{data.name}"


class sAlexandria(OpenCatalystDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder,
            license="CC-BY-4.0",
            download_link="",
            n_entries=11000983,
        )

    def data_to_dict(self, data, split_type):
        # {'numbers': [58, 58, 22, 22, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
        # 'positions': [[2.733677827940298, 2.399301348172859, 2.137917224979261], [5.5827124720597014, 4.89984936182714,
        # 4.366051125020738], [0.7773680965080659, 0.6822824192796962, 0.607953368420937], [7.539022203491933, 6.616868290720303,
        # 5.896014981579063], [3.41547895, -0.072360595, -0.0644775], [0.3713581, 0.325934525, 3.380939175], [0.3713581,
        # 3.396001425, -0.0644775], [0.7427162, 5.024350001177634, 1.8548130413230846], [5.0782440120185806, 1.852258095826083,
        # -0.128955], [2.495430087981419, 0.42254261299628315, 4.778110308676916], [2.495430087981419, 4.795023564173918,
        # -0.128955], [5.0782440120185806, 0.08460524700371679, 1.8548130413230846], [0.7427162, 2.4195218988223663,
        # 4.778110308676916], [2.017699942897146, 4.767478996912451, 4.248101417289083], [4.988953915640364, 4.378717835336282,
        # 0.8851574056052249], [4.988953915640364, 1.3821388811162292, 4.248101417289083], [6.298690357102854, 2.531671713087549,
        # .2558669327109158], [3.3274363843596357, 2.9204328746637183, 5.618810944394775], [3.3274363843596357, 5.917011828883771,
        # 2.2558669327109158]], 'unique_id': '3a987f7ca3c783b2d77f4ed3db75a4e8', 'pbc': [True, True, True], 'cell': [[6.8309579,
        # -0.14472119, -0.128955], [0.7427162, 6.79200285, -0.128955], [0.7427162, 0.65186905, 6.76187835]], 'calculator':
        # 'unknown', 'calculator_parameters': {}, 'energy': -50.5057429, 'forces': [[0.00046851, 0.00041121, 0.00036641],
        # [-0.00046851, -0.00041121, -0.00036641], [9.716e-05, 8.528e-05, 7.599e-05], [-9.716e-05, -8.528e-05, -7.599e-05],
        #  [-0.0, -0.0, 0.0], [-0.0, -0.0, 0.0], [-0.0, -0.0, -0.0], [-0.0, -0.00120113, 0.00134798], [-0.00119098, 0.00135696,
        # -0.0], [0.00119098, -0.00015583, -0.00134798], [0.00119098, -0.00135696, 0.0], [-0.00119098, 0.00015583, 0.00134798],
        # [-0.0, 0.00120113, -0.00134798], [-0.00030292, 0.00240572, 0.00214363], [0.00234609, 0.00205912, -0.00085458],
        # [0.00234609, -0.00061247, 0.00214363], [0.00030292, -0.00240572, -0.00214363], [-0.00234609, -0.00205912, 0.00085458],
        # [-0.00234609, 0.00061247, -0.00214363]], 'stress': [4.786414244498755e-05, 4.9116519975623406e-05, 4.998188024989062e-05,
        #  -3.742864502045791e-06, -4.264479902714107e-06, -4.785864367544764e-06], 'ctime': 24.758530868976667, 'user': 'lbluque',
        # 'mtime': 24.758530868976667, 'data': {'sid': 'agm003300702_0_0', 'prototype_label': 'A2B2C15_hR57_166_c_c_dfh:Ce-Ti-Zn'}}

        assert all(data["pbc"]), "PBCs should apply to all directions"

        stress_voigt = data["stress"]
        stress = tensor(
            [
                [
                    [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                    [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                    [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
                ]
            ],
            dtype=torch.float64,
        )

        data_kwargs = {
            "z": tensor(data["numbers"]),
            "pos": tensor(data["positions"], dtype=torch.float64),
            "box": tensor(data["cell"], dtype=torch.float64).unsqueeze(0),
            "total_energy": tensor([data["energy"]], dtype=torch.float64),  # eV
            "force": tensor(data["forces"], dtype=torch.float64),  # eV/A
            "stress": stress,  # eV/A^3
            "split": split_type,
        }

        return data_kwargs, data["unique_id"]

    def _download(self) -> str:
        folder_path = join(self.root_folder, "Alexandria")

        links = {
            join(folder_path, "train.tar.gz"): "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz",
            join(folder_path, "val.tar.gz"): "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/val.tar.gz",
        }

        if not exists(folder_path):
            makedirs(folder_path)

        for file_path, download_link in links.items():
            if not exists(file_path):
                wget.download(download_link, out=file_path, bar=self.bar_in_MB)
                print(f"\n{file_path} downloaded.")

        return list(links.keys())


class OMAT24(OpenCatalystDataHandler):
    def __init__(self, root_folder):
        super().__init__(
            root_folder,
            license="CC-BY-4.0",
            download_link="",
            n_entries=101849946,
        )

    def data_to_dict(self, data, split_type):
        # {'numbers': [12, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 'positions': [[2.2797634666261444, 2.0571831438901182,
        # 2.5065354416770873], [1.07361636988187, 4.227351134921758, 2.0637412019778263], [2.9282035136699727, 4.51352252263954,
        # 2.8096104801709867], [4.099768245872722, 2.2203470740700406, 1.0797996416189222], [0.12896817345950262,
        # 2.8001591589053074, 4.687697696799841], [2.737170913442615, 1.7228009149383112, 0.20307571044489628],
        # [2.7914929114024956, 3.3785663225574782, 4.494842295518068], [4.830048729380251, 2.9036332099487714, 2.5886555457441562],
        #  [0.6733389003504636, 1.6694495876898197, 1.9536405941842354], [3.629249176119807, 2.0258575550330487, 4.551012476168204],
        #  [1.497463605460619, 2.009679162799493, 0.721161766715337], [2.1197128750200833, 4.548151489291896, 4.288034515617653],
        # [3.0968220567686027, 4.652878504187409, 1.0970059965031955]], 'unique_id': '5b24a0c7ecd989e90e69ae169f6f6c77', 'pbc':
        # [True, True, True], 'cell': [[4.895352047627077, 0.0, 0.0], [0.0, 4.727098351107126, 0.0], [0.0, 0.0, 5.136330238020333]],
        #  'initial_magmoms': [-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0], 'calculator': 'unknown',
        # 'calculator_parameters': {}, 'energy': -59.33341297, 'forces': [[6.91655291, 1.18709092, 5.44989599], [-0.10549774,
        #  1.97810818, 1.66801744], [-1.57180157, -1.03373953, -0.80727408], [2.01749882, -0.2132175, -1.12247722], [1.2511966,
        # -0.28992558, -0.35243906], [-7.3328522, -7.51981219, 8.36360538], [4.41488278, -5.34130643, 2.14167417], [0.16439032,
        #  1.96156377, 1.6750757], [-7.46300923, -4.85506184, -2.24044133], [15.87026434, 4.87605416, -11.66880074], [-9.37076495,
        #  1.68703238, -1.66516961], [-3.42398415, 8.00393265, -1.57477912], [-1.36687592, -0.440719, 0.13311248]], 'stress':
        # [-0.3438417788264359, -0.08534068871990372, -0.07392910008314608, 0.022804080081231456, 0.13011420597056814,
        # 0.035154582268761565], 'ctime': 24.70459938637797, 'user': 'lbluque', 'mtime': 24.70459938637797, 'data': {'sid':
        #  'agm000999967_AB12_12_spg221_1_0_rattled-1000_77w3vd', 'calc_id': 'rattled-1000', 'task_type': 'Static',
        # 'composition_reduced': 'Mg1 B12', 'prototype_label': 'A12B_aP13_1_12a_a:B-Mg', 'prototype_error': '',
        #  'energy_corrected_mp2020': -59.33341297, 'energy_correction_uncertainty_mp2020': 0.0, 'energy_adjustments_mp2020': [],
        # 'correction_warnings': ['Failed to guess oxidation states for Entry None (MgB12). Assigning anion correction to only
        # the most electronegative atom.'], 'elements': 'MgB', 'parent_id': 'agm000999967_AB12_12_spg221',
        # 'parent_prototype_label': 'A12B_cP13_221_h_a:B-Mg'}}

        assert all(data["pbc"]), "PBCs should apply to all directions"

        stress_voigt = data["stress"]
        stress = tensor(
            [
                [
                    [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
                    [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
                    [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
                ]
            ],
            dtype=torch.float64,
        )

        data_kwargs = {
            "z": tensor(data["numbers"]),
            "pos": tensor(data["positions"], dtype=torch.float64),
            "box": tensor(data["cell"], dtype=torch.float64).unsqueeze(0),
            "total_energy": tensor([data["energy"]], dtype=torch.float64),
            "force": tensor(data["forces"], dtype=torch.float64),
            "stress": stress,
            "split": split_type,
        }

        return data_kwargs, data["unique_id"]

    def _download(self) -> str:
        folder_path = join(self.root_folder, "OMAT24")

        links = {
            "train_rattled-1000.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000.tar.gz",
            "val_rattled-1000.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz",
            "train_rattled-1000-subsampled.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000-subsampled.tar.gz",
            "val_rattled-1000-subsampled.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz",
            "train_rattled-500.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz",
            "val_rattled-500.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz",
            "train_rattled-500-subsampled.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500-subsampled.tar.gz",
            "val_rattled-500-subsampled.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500-subsampled.tar.gz",
            "train_rattled-300.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz",
            "val_rattled-300.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300.tar.gz",
            "train_rattled-300-subsampled.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz",
            "val_rattled-300-subsampled.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300-subsampled.tar.gz",
            "train_aimd-from-PBE-1000-npt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-npt.tar.gz",
            "val_aimd-from-PBE-1000-npt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-npt.tar.gz",
            "train_aimd-from-PBE-1000-nvt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-nvt.tar.gz",
            "val_aimd-from-PBE-1000-nvt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-nvt.tar.gz",
            "train_aimd-from-PBE-3000-npt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-npt.tar.gz",
            "val_aimd-from-PBE-3000-npt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-npt.tar.gz",
            "train_aimd-from-PBE-3000-nvt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-nvt.tar.gz",
            "val_aimd-from-PBE-3000-nvt.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-nvt.tar.gz",
            "train_rattled-relax.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-relax.tar.gz",
            "val_rattled-relax.tar.gz": "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-relax.tar.gz",
        }

        self.file_to_split = {key: key.split("_")[0] for key in links.keys()}

        if not exists(folder_path):
            makedirs(folder_path)

        file_paths = []
        for file_path, download_link in links.items():
            file_path = join(folder_path, file_path)
            file_paths.append(file_path)
            if not exists(file_path):
                wget.download(download_link, out=file_path, bar=self.bar_in_MB)
                print(f"\n{file_path} downloaded.")

        return file_paths
