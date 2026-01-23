import os
from collections import defaultdict
from itertools import chain
from os.path import join
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import yaml
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch_geometric.data import Batch
from tqdm import tqdm

from geodite.utils import DataInput

from ..utils.graph import BatchedPeriodicDistance, get_symmetric_displacement
from .decoder import TASK_TO_DECODER
from .encoder import ENCODER_CLASS_MAP

OPTIMIZER_CLASS_MAP = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "LBFGS": torch.optim.LBFGS,
}


class GeoditeModule(LightningModule):
    def __init__(self, config: Dict[str, Dict[str, Any]]) -> None:
        super().__init__()

        self.constants_file_path = None

        encoder_args = config["encoder"]["args"]
        self.save_hyperparameters(config)

        hp = self.hparams

        self.distance = BatchedPeriodicDistance(hp.encoder["args"]["cutoff"])
        self.encoder = ENCODER_CLASS_MAP[hp.encoder["arch"]](**encoder_args)

        combine = hp.decoder.get("combine_decoders", True)
        dataset_tasks = hp.dataset["datasets"]
        self.decoders = {}
        if combine:
            # One decoder per unique task
            for task in sorted(set(chain.from_iterable(dataset_tasks.values()))):
                dec = TASK_TO_DECODER[task](**hp.decoder)
                self.add_module(dec.id, dec)
                self.decoders[dec.id] = dec

        else:
            # One decoder per (dataset, task)
            for ds_name, tasks in dataset_tasks.items():
                for task in tasks:
                    dec = TASK_TO_DECODER[task](dataset_name=ds_name, **hp.decoder)
                    self.add_module(dec.id, dec)
                    self.decoders[dec.id] = dec

        self.synced_buffers = False

    def forward(self, data: DataInput):
        data.pos.requires_grad_(True)

        data.pos, data.box, data.displacements = get_symmetric_displacement(data.pos, data.box, data.num_graphs, data.batch)

        if hasattr(data, "cutoff_edge_index") and (data.cutoff_edge_index > 0).all():
            data.cutoff_edge_index, data.cutoff_edge_distance, data.cutoff_edge_vec, _ = self.distance(
                data.pos, data.box, data.batch, data.cutoff_edge_index, data.cutoff_shifts_idx
            )
        else:
            data.cutoff_edge_index, data.cutoff_edge_distance, data.cutoff_edge_vec, data.cutoff_shifts_idx = self.distance(
                data.pos, data.box, data.batch
            )

        embedding_dict = self.encoder(data)

        data.embedding_0 = embedding_dict["embedding_0"]
        if "embedding_1" in embedding_dict.keys():
            data.embedding_1 = embedding_dict["embedding_1"]

        return data

    def sync_buffers(self):
        if dist.is_initialized():
            dist.barrier()

        for buffer_name in ["context_to_index"]:
            if self.global_rank == 0:
                buffer_in_cpu = self.recursive_move_to_cpu(getattr(self.encoder, buffer_name))
            else:
                buffer_in_cpu = None

            setattr(self.encoder, buffer_name, self.trainer.strategy.broadcast(buffer_in_cpu, 0))

        if self.global_rank != 0:
            self.encoder.setup()

        self.encoder.to(self.device)

        for decoder in self.decoders.values():
            for buffer_name in ["constants", "context"]:
                if self.global_rank == 0:
                    buffer_in_cpu = self.recursive_move_to_cpu(getattr(decoder, buffer_name))
                else:
                    buffer_in_cpu = None

                setattr(decoder, buffer_name, self.trainer.strategy.broadcast(buffer_in_cpu, 0))
                decoder.to(self.device)

        self.synced_buffers = True

    @staticmethod
    def recursive_move_to_cpu(var_in_gpu):
        if isinstance(var_in_gpu, dict):
            for k in var_in_gpu.keys():
                var_in_gpu[k] = GeoditeModule.recursive_move_to_cpu(var_in_gpu[k])
            return var_in_gpu
        elif isinstance(var_in_gpu, Tensor):
            return var_in_gpu.cpu()
        else:
            return var_in_gpu

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict", checkpoint)

        encoder_keys = {}
        decoder_keys = {}
        scalarizer_keys = {}
        main_model_keys = {}

        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_keys[key] = value
            elif key.startswith("scalarizer."):
                scalarizer_keys[key] = value
            elif any(key.startswith(f"{d.id}.") for d in self.decoders.values()):
                decoder_keys[key] = value
            else:
                main_model_keys[key] = value

        main_model_checkpoint = checkpoint.copy()
        main_model_checkpoint["state_dict"] = main_model_keys

        super().on_load_checkpoint(main_model_checkpoint)

        # Load encoder
        encoder_state_dict = {k[len("encoder.") :]: v for k, v in encoder_keys.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state_dict, strict=False)

        # Load decoders
        for decoder in self.decoders.values():
            decoder_state_dict = {k[len(decoder.id) + 1 :]: v for k, v in decoder_keys.items() if k.startswith(decoder.id + ".")}
            decoder.load_state_dict(decoder_state_dict, strict=False, device=self.device)
            decoder.to(self.device)

        # Load scalarization method

        self.starting_from_checkpoint = True
        self.strict_loading = False

    def get_constants_file_path(self):
        return self.constants_file_path

