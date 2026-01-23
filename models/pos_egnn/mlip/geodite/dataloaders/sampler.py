from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
from torch_geometric.data import Dataset
from torch_geometric.loader import DynamicBatchSampler


class CustomDynamicBatchSampler(DynamicBatchSampler):
    def __len__(self) -> int:
        if self.num_steps is None:
            raise NotImplementedError()
        return self.num_steps


class DistributedDynamicBatchSampler(CustomDynamicBatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        max_num: int,
        mode: str = "node",
        shuffle: bool = False,
        skip_too_big: bool = False,
        num_steps: Optional[int] = None,
        num_replicas=None,
        rank=None,
        drop_last=False,
        seed=123,
    ):
        if max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num})")
        if mode not in ["node", "edge"]:
            raise ValueError(f"`mode` choice should be either 'node' or 'edge' (got '{mode}')")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.rank = rank
        self.num_replicas = num_replicas
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last

        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps

        self.max_steps = num_steps or len(dataset)

        self.cached_batches = []

    def __iter__(self) -> Iterator[List[int]]:
        if self.epoch > 0:
            assert len(self.cached_batches) != 0
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.cached_batches), generator=g).tolist()
            else:
                indices = list(range(len(self.cached_batches)))

            for i in indices:
                yield self.cached_batches[i]

        else:
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            batch_of_batches = []
            samples: List[int] = []
            current_num: int = 0
            num_steps: int = 0
            num_processed: int = 0

            while num_processed < len(self.dataset) and num_steps < self.max_steps:
                for i in indices[num_processed:]:
                    data = self.dataset[i]
                    num = data.num_nodes if self.mode == "node" else data.num_edges

                    if current_num + num > self.max_num:
                        if current_num == 0:
                            if self.skip_too_big:
                                continue
                        else:  # Mini-batch filled:
                            break

                    samples.append(i)
                    num_processed += 1
                    current_num += num

                batch_of_batches.append(samples)

                if len(batch_of_batches) == self.num_replicas:
                    self.cached_batches.append(batch_of_batches[self.rank])
                    yield batch_of_batches[self.rank]
                    batch_of_batches = []

                samples: List[int] = []
                current_num = 0
                num_steps += 1

            if len(batch_of_batches) != 0:
                if not self.drop_last:
                    self.cached_batches.append(batch_of_batches[self.rank % len(batch_of_batches)])
                    yield batch_of_batches[self.rank % len(batch_of_batches)]

        self.epoch += 1
