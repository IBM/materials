from itertools import chain
from random import shuffle

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch import get_num_threads
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from ..data import initialize_datasets
from ..model.decoder import TASK_TO_DECODER
from .sampler import CustomDynamicBatchSampler, DistributedDynamicBatchSampler


class GeoditeDataLoader(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()

        # Set the attributes of the class from the config
        for key, value in config["dataset"].items():
            setattr(self, key, value)

        self.dataset_names = list(self.datasets.keys())
        self.db_manager = initialize_datasets(
            datasets=self.dataset_names,
            root_folder=self.dataset_path,
            reprocess=self.reprocess,
            n_jobs=self.n_jobs,
            cutoff=self.cutoff,
            pre_transforms=self.pre_transforms if hasattr(self, "pre_transforms") else None,
            max_elements=self.max_elements if hasattr(self, "max_elements") else None,
            check_integrity=self.check_integrity if hasattr(self, "check_integrity") else False,
        )

        if not hasattr(self, "max_elements_for_constants"):
            self.max_elements_for_constants = float("inf")

        unique_tasks = sorted(set(chain(*list(self.datasets.values()))))  # Remove duplicates and sort
        self.decoder_target_keys = {t: TASK_TO_DECODER[t].target_keys.fget(None) for t in unique_tasks}
        self.decoder_constants_keys = {t: TASK_TO_DECODER[t].constants_keys.fget(None) for t in unique_tasks}

        if self.dataloader_n_workers == -1:
            self.dataloader_n_workers = get_num_threads()

        if int(self.dataloader_n_workers) > 1:
            self.dataloader_n_workers_train = self.dataloader_n_workers // 2 + self.dataloader_n_workers % 2
            self.dataloader_n_workers_val = self.dataloader_n_workers // 2
        elif int(self.dataloader_n_workers) == 1:
            self.dataloader_n_workers_train = 1
            self.dataloader_n_workers_val = 1
        else:
            self.dataloader_n_workers_train = 0
            self.dataloader_n_workers_val = 0

        if hasattr(self, "max_edges"):
            self.sampler = True
        else:
            self.sampler = False

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_datasets = {}
            self.val_datasets = {}
            self.test_datasets = {}

            for db in self.db_manager:
                name = db.database_name

                # decide which splitting strategy to use
                if getattr(db[0], "split", None) is None:
                    train_ids, val_ids, test_ids = self._random_split_indices(len(db))
                else:
                    train_ids, val_ids, test_ids = self._by_point_split_indices(db)

                # build the Subsets
                self.train_datasets[name] = Subset(db, train_ids)
                self.val_datasets[name] = Subset(db, val_ids)
                if test_ids:
                    self.test_datasets[name] = Subset(db, test_ids)

    def _random_split_indices(self, total_elements):
        """
        Randomly split indices based on self.split["train"] and self.split["val"],
        each of which may be either:
          - an int > 1   → treated as an absolute count
          - a float ≤ 1  → treated as a fraction of the dataset
        The remainder becomes test. Respects self.max_elements if set.
        """
        n = min(getattr(self, "max_elements", total_elements), total_elements)

        tr = self.split["train"]
        va = self.split["val"]
        sd = self.split["seed"]

        # compute absolute counts
        n_train = int(tr if tr > 1 else n * tr)
        n_val = int(va if va > 1 else n * va)
        n_test = max(0, n - n_train - n_val)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sd)
            if n_test > 0:
                train_ids, val_ids, test_ids = random_split(range(n), [n_train, n_val, n_test])
            else:
                train_ids, val_ids = random_split(range(n), [n_train, n_val])
                test_ids = []

        return train_ids, val_ids, test_ids

    def _by_point_split_indices(self, db):
        train_ids, val_ids = [], []

        for idx, point in tqdm(enumerate(db), desc=f"Splitting {db.database_name}"):
            split_lbl = getattr(point, "split", "").lower()
            if split_lbl == "train":
                train_ids.append(idx)
            elif split_lbl in ("val", "validation"):
                val_ids.append(idx)

        shuffle(train_ids)
        shuffle(val_ids)

        return train_ids, val_ids, []

    def teardown(self, stage=None):
        # Datasets have an open read transaction for an LMDB, so good to clean up
        # Otherwise transactions might hit maximum possible open transactions
        if stage == "fit" or stage is None:
            self.db_manager.close()

    def batch_for_registering_task(self, task: str):
        # Get datasets with task
        datasets = [key for key, values in self.datasets.items() if task in values]
        # Get list of Datas from each dataset.
        # NOTE: This might be a problem when dealing with huge datasets.

        # complete_list = [item for key in datasets for item in tqdm(self.train_datasets[key])]

        complete_list = []
        for key in datasets:
            n = 0
            for item in tqdm(self.train_datasets[key]):
                if n < self.max_elements_for_constants:
                    complete_list.append(item)
                else:
                    break
                n += 1

        complete_batch = self._heterogeneous_collate(complete_list, select_task=task)
        return complete_batch[task]

    def _make_dataloader(self, datasets: dict, shuffle: bool, num_workers: int, prefetch_factor: int = 5):
        loaders = {}
        n_datasets = len(datasets)
        for name, ds in datasets.items():
            if self.sampler:
                if self.devices == 1:
                    sampler = CustomDynamicBatchSampler(
                        dataset=ds,
                        max_num=self.max_edges // n_datasets,
                        mode="edge",
                        shuffle=shuffle,
                    )
                else:
                    sampler = DistributedDynamicBatchSampler(
                        dataset=ds,
                        max_num=self.max_edges // n_datasets,
                        mode="edge",
                        shuffle=shuffle,
                    )
                loader_kwargs = {"batch_sampler": sampler}
            else:
                loader_kwargs = {"batch_size": self.batch_size // n_datasets, "shuffle": shuffle}

            loaders[name] = DataLoader(
                ds,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                collate_fn=self._heterogeneous_collate,
                prefetch_factor=prefetch_factor,
                **loader_kwargs,
            )

        return CombinedLoader(loaders, mode="max_size_cycle")

    def train_dataloader(self):
        return self._make_dataloader(self.train_datasets, shuffle=True, num_workers=self.dataloader_n_workers_train)

    def val_dataloader(self):
        return self._make_dataloader(self.val_datasets, shuffle=False, num_workers=self.dataloader_n_workers_val)

    def test_dataloader(self):
        return self._make_dataloader(self.test_datasets, shuffle=False, num_workers=self.dataloader_n_workers_val)

    def _heterogeneous_collate(self, data_list, select_task=None):
        # Initialize stratified_batch as a nested dictionary
        stratified_batch = {}

        if len(data_list) > 10000:
            progress_bar = tqdm(total=len(data_list))
        else:
            progress_bar = None

        while data_list:
            data = data_list.pop(0)
            all_keys = data.keys()
            tasks = [select_task] if select_task else self.datasets[data.dataset_name]

            for task in tasks:
                attr_list = self.decoder_target_keys[task]  # Get all target keys for the task
                for attr in attr_list:
                    matching_keys = [key for key in all_keys if key.startswith(attr)]
                    for key in matching_keys:
                        fidelity = f"{data.dataset_name}_{key[len(attr) + 1 :]}"
                        value = getattr(data, key)
                        # Skip data items without valid targets
                        if torch.isnan(torch.as_tensor(value)).any():
                            continue

                        # Initialize nested dictionaries as needed
                        if task not in stratified_batch:
                            stratified_batch[task] = {}
                        if fidelity not in stratified_batch[task]:
                            stratified_batch[task][fidelity] = {}

                        # Check if data.id is already in the dict for this task/dataset/fidelity
                        filtered_data = Data(num_nodes=data.num_nodes)
                        setattr(data, attr, value)
                        for keep_attr in self.decoder_constants_keys[task] + ATTRS_IN_EVERY_DATA + attr_list:
                            if hasattr(data, keep_attr):
                                setattr(filtered_data, keep_attr, getattr(data, keep_attr))

                        # Store the filtered_data in the dict with data.id as the key
                        setattr(filtered_data, "fidelity", fidelity)
                        stratified_batch[task][fidelity][data.id] = filtered_data

            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        # Convert the dictionaries to Batches
        for task, fidelity_to_data_dict in stratified_batch.items():
            for fidelity, data_dict in fidelity_to_data_dict.items():
                data_list = list(data_dict.values())
                batch = Batch.from_data_list(data_list)
                batch = self.convert_batch_precision(batch)
                stratified_batch[task][fidelity] = batch

        return stratified_batch

    def convert_batch_precision(self, batch):
        precision = self.trainer.precision
        if isinstance(precision, str):
            if precision.startswith("16"):
                target_dtype = torch.float16
            elif precision.startswith("bf16"):
                target_dtype = torch.bfloat16
            elif precision.startswith("32"):
                target_dtype = torch.float32
            elif precision.startswith("64"):
                target_dtype = torch.float64
            else:
                target_dtype = torch.float32
        else:
            if precision == 16:
                target_dtype = torch.float16
            elif precision == 32:
                target_dtype = torch.float32
            elif precision == 64:
                target_dtype = torch.float64
            else:
                target_dtype = torch.float32

        batch = batch.apply(lambda x: x.to(target_dtype) if isinstance(x, torch.Tensor) and x.dtype.is_floating_point else x)
        return batch


ATTRS_IN_EVERY_DATA = ["pos", "z", "box", "id", "cutoff_edge_index", "cutoff_shifts_idx"]
