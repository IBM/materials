import pickle
from os import makedirs
from os.path import exists, getsize, join
from shutil import rmtree
from typing import List

import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm


class LmdbDataset(Dataset):
    def __init__(self, root_folder: str, database_name: str, reprocess: bool = False, cutoff: float = 6.0):
        self.database_name = database_name
        file_path = join(root_folder, database_name)
        self.cutoff = cutoff
        self.file_path = file_path
        if not exists(file_path):
            makedirs(file_path, exist_ok=True)
            write = True
        elif not exists(join(file_path, "data.mdb")):
            write = True
        elif reprocess:
            self.drop()
            write = True
        else:
            write = False

        self.open(write=write)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == len(self):
            raise StopIteration
        entry = self.__getitem__(self.idx)
        self.idx += 1
        return entry

    def __getitem__(self, key: int | slice) -> Data:
        if isinstance(key, slice):
            values = []
            start = key.start if key.start is not None else 0  # slices can be None
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            for i in range(start, stop, step):
                values.append(self.__getitem__(i))
            return values
        byte_value = self.txn.get(encode_key(key))
        value = decode_value(byte_value)
        return value

    def __setitem__(self, key: int, data: Data) -> None:
        self.txn.put(encode_key(key), encode_value(data), overwrite=True)
        self.txn.commit()
        self.txn = self.env.begin(write=True)

    def __len__(self):
        return self.txn.stat()["entries"] - 1  # -1 because of cutoff

    def __str__(self):
        return self.database_name

    def __contains__(self, key: int):
        return self.txn.get(encode_key(key)) is not None

    def append(self, data: Data):
        idx = self.__len__()
        return self.__setitem__(idx, data)

    def close(self):
        self.env.close()
        self.env = None
        self.txn = None

    def open(self, write=False):
        self.write = write

        if write:
            # Write-optimized configuration
            self.env = lmdb.open(
                self.file_path,
                map_size=int(3 * 1024 * 1024 * 1024 * 1024),
                sync=False,
                lock=True,
                readonly=False,
            )
        else:
            # Read-only optimized configuration
            self.env = lmdb.open(self.file_path, readonly=True, lock=False, map_size=int(3 * 1024 * 1024 * 1024 * 1024), readahead=True)

        # self._warmup_file()

        self.txn = self.env.begin(write=write)

        if write:
            self._check_cutoff(self.cutoff)

    def drop(self):
        rmtree(self.file_path)

    def _check_cutoff(self, cutoff: float):
        current_cutoff = self.txn.get(encode_key("cutoff"))
        if current_cutoff and decode_value(current_cutoff) != cutoff:
            raise ValueError(f"Dataset {str(self)} has cutoff={decode_value(current_cutoff)} which is incompatible with cutoff={cutoff}")
        self["cutoff"] = self.cutoff

    def _warmup_file(self):
        file_path = join(self.file_path, "data.mdb")
        total_size = getsize(file_path)
        chunk_size = 1 << 20  # 1 MiB

        with open(file_path, "rb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="Warming up LMDB") as pbar:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                pbar.update(len(data))

    def check_integrity(self) -> bool:
        try:
            with self.env.begin(write=False) as txn:
                total_entries = self.__len__()
                for idx in tqdm(range(total_entries), desc="Checking DB integrity"):
                    _ = txn.get(encode_key(idx))
            return True

        except (lmdb.CorruptedError, lmdb.Error):
            return False


class LmdbManager:
    def __init__(self, lmdbs: List[LmdbDataset]):
        database_dict = {}
        for db in lmdbs:
            db_name = str(db)
            database_dict[db_name] = db
        self.database_dict = database_dict
        self.lmdbs = lmdbs

    def __getitem__(self, idx_or_dataset_name: str | int) -> None:
        if isinstance(idx_or_dataset_name, str):
            return self.database_dict[idx_or_dataset_name]  # returns the actual LMDB object, making it possible to pass an index right away

        cur_idx = 0
        for db in self.lmdbs:
            if cur_idx <= idx_or_dataset_name < cur_idx + len(db):  # check the current idx of the db
                return db[idx_or_dataset_name - cur_idx]
            cur_idx += len(db)
        raise ValueError(f"index {idx_or_dataset_name} does not exist inside db, please insert index between 0 and {cur_idx - 1}")

    def __len__(self) -> int:
        total_len = 0
        for db in self.lmdbs:
            total_len += len(db)
        return total_len

    def __iter__(self):
        for db in self.lmdbs:
            yield db

    def close(self):
        for db in self.lmdbs:
            db.close()


# Specifies how we encode the keys and values in LMDB
def encode_key(key):
    return str(key).encode()


def encode_value(value):
    return pickle.dumps(value)


def decode_value(value):
    return pickle.loads(value)
