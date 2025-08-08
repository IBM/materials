from abc import ABC

from torch import nn


class AbstractEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.context_to_index = {}

    def register_datasets(self, dataset_batch=None, precomputed_constants=None, file_to_store_constants=None):
        pass

    def setup(self):
        pass
