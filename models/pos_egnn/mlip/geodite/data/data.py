from typing import List, Optional

from joblib import Parallel, delayed
from torch import get_num_threads
from tqdm import tqdm

from .database import LmdbDataset, LmdbManager
from .datasets import DataHandlerManager
from .parser import Parser


def initialize_datasets(
    datasets: List[str],
    root_folder: str,
    pre_transforms: Optional[List[str]] = None,
    cutoff: float = 5.0,
    reprocess: bool = False,
    n_jobs: int = -1,
    max_elements: Optional[int] = None,
    check_integrity: Optional[bool] = False,
):
    datahandler_m = DataHandlerManager(datasets, root_folder=f"{root_folder}/files")
    lmdbs = []
    for dataset in datahandler_m.datasets:
        print("Processing ", dataset.name)
        lmdb = LmdbDataset(
            root_folder=f"{root_folder}/databases",
            database_name=dataset.name,
            cutoff=cutoff,
            reprocess=reprocess,
        )

        if max_elements:
            total_elements = min(max_elements, dataset.n_entries if dataset.n_entries is not None else float("inf"))
        else:
            total_elements = dataset.n_entries if dataset.n_entries is not None else None

        print(f"skipping {len(lmdb)} files...")

        # NOTE: These changes dont allow to lower max_elements once the dataset is generated.
        if len(lmdb) < total_elements:  # Skip if LMDB contains the necessary Datas
            data_generator = dataset.get_data_stream(n_files_to_skip=len(lmdb))
            parser = Parser(dataset, cutoff, pre_transforms)
            parallel_generator = get_parallel_generator(n_jobs, parser, data_generator)
            for data in tqdm(parallel_generator, desc="Inserting into DB", total=total_elements, initial=len(lmdb)):
                if max_elements and len(lmdb) >= total_elements:
                    break
                if not lmdb.write:  # Open read and write environment
                    lmdb.open(write=True)
                lmdb.append(data)
            parallel_generator.close()

        if lmdb.write:
            lmdb.open(write=False)

        lmdbs.append(lmdb)

    lmdb_manager = LmdbManager(lmdbs)

    if check_integrity:
        for lmdb in lmdb_manager:
            assert lmdb.check_integrity(), f"LMDB ({lmdb.database_name}) is corrupted."

    return lmdb_manager


def get_parallel_generator(n_jobs, parser, data_generator):
    if n_jobs == -1:
        n_jobs = get_num_threads()
    return Parallel(
        n_jobs=n_jobs,
        return_as="generator",
        verbose=0,
        timeout=6000,
    )(delayed(parser.parse)(d) for d in data_generator)
