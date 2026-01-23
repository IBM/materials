from typing import List

from .md.ani import ANI2X
from .md.matpes import MatPES_r2SCAN
from .md.mptrj import MPtrj
from .md.opencatalyst import OC22_S2EF, ODAC23_S2EF, OMAT24, sAlexandria
from .md.revmd17 import RevMD17


class DataHandlerManager:
    def __init__(self, datasets_names: List[str], root_folder: str):
        initialized_datasets = []
        for dataset_name in datasets_names:
            dataset_class = str_to_class[dataset_name]
            dataset_class = dataset_class(root_folder=root_folder)
            initialized_datasets.append(dataset_class)
        self.datasets = initialized_datasets


str_to_class = {
    "ANI2X": ANI2X,
    "MatPES_r2SCAN": MatPES_r2SCAN,
    "MPtrj": MPtrj,
    "ODAC23_S2EF": ODAC23_S2EF,
    "OC22_S2EF": OC22_S2EF,
    "OMAT24": OMAT24,
    "RevMD17": RevMD17,
    "sAlexandria": sAlexandria,
}
