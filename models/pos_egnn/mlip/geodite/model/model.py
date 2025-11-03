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
from .scalarizer import SCALARIZER_CLASS_MAP
from .scheduler import SCHEDULER_CLASS_MAP

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

    def training_step(self, data: Dict[str, Dict[str, Dict[str, Batch]]]) -> Tensor:
        if hasattr(self, "frozen_encoder"):
            self.encoder.eval()
        loss = self._compute_loss_and_log_metrics(data)
        return loss

    def validation_step(self, data: Dict[str, Dict[str, Dict[str, Batch]]]) -> Tensor:
        loss = self._compute_loss_and_log_metrics(data)
        return loss

    def test_step(self, data: Dict[str, Dict[str, Dict[str, Batch]]]) -> Tensor:
        loss = self._compute_loss_and_log_metrics(data)
        return loss

    def _compute_loss_and_log_metrics(self, data: Dict[str, Dict[str, Dict[str, Batch]]]):
        losses_dict, metrics = self._get_losses(data)

        losses = torch.stack(list(losses_dict.values()))

        if self.scalarizer.dynamic:
            self.trainer.strategy.reduce(losses, reduce_op="mean")
            self.latest_task_losses = {name: loss.detach().item() for name, loss in losses_dict.items()}

        loss = self.scalarizer(losses)

        # Batch size independent of sampler
        total_graphs = sum(batch.num_graphs for sub in data.values() for batch in sub.values())

        self._log_metrics(loss, metrics, total_graphs)

        return loss

    def _get_losses(self, data: Dict[str, Dict[str, Dict[str, Batch]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        losses = self.losses_template.copy()  # A dict mapping the loss key to 0 tensor
        metrics = []

        new_data = {}
        for ds in data.values():
            for task, batches in ds.items():
                new_data.setdefault(task, {}).update(batches)

        data.clear()
        data.update(new_data)

        # Process each decoder only if its *declared* task is in the batch
        for decoder in self.decoders.values():
            task_name = decoder.task
            if task_name not in data:
                continue
            task_batches = data[task_name]
            if decoder._dataset_name:
                # split‐mode: only handle the one fidelity that matches this decoder
                fid = f"{decoder._dataset_name}_"
                if fid not in task_batches:
                    continue
                selection = {fid: task_batches[fid]}
            else:
                # combine‐mode: handle all fidelities
                selection = task_batches

            for fidelity, batch in selection.items():
                out = self.forward(batch)
                d_loss, d_metrics = decoder.forward_loss(out, fidelity)
                losses.update(d_loss)
                metrics.append(d_metrics)

        return losses, metrics

    def _log_metrics(self, total_loss: Tensor, metrics: List[Dict[str, Tensor]], batch_size: int) -> None:
        step_type = "Test" if self.trainer.testing else "Validation" if self.trainer.validating else "Train"
        combined_dict = {f"Total loss/{step_type}": total_loss.detach()}
        for dictionary in metrics:
            for key, value in dictionary.items():
                new_key = f"{key}{step_type}"
                combined_dict[new_key] = value

        self.log("Batch size", batch_size, batch_size=batch_size, sync_dist=True)
        self.log_dict(combined_dict, sync_dist=True, batch_size=batch_size)

    def configure_optimizers(self):
        # Calling this here because it was the only hook between configure_model and on_fit_start
        if not hasattr(self, "starting_from_checkpoint"):
            self._register_datasets()

        optimizer_class = OPTIMIZER_CLASS_MAP[self.hparams.optimizer["method"]]
        optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer["args"])

        # Define scalarizer
        if not hasattr(self, "scalarizer"):
            n_tasks = len(self._init_decoder_losses())
            scalarizer_method = SCALARIZER_CLASS_MAP[self.hparams.optimizer["scalarization"]["method"]]
            self.scalarizer = scalarizer_method(n_tasks=n_tasks, device=self.device, **self.hparams.optimizer["scalarization"]["args"])

        if "schedulers" in self.hparams["optimizer"]:
            schedulers_list = []

            for scheduler_config in self.hparams["optimizer"]["schedulers"]:
                lr_method = SCHEDULER_CLASS_MAP[scheduler_config["method"]]
                schedulers_list.append(
                    {
                        "scheduler": lr_method(optimizer, **scheduler_config["method_args"]),
                    }
                    | scheduler_config["monitor_args"]
                )

                print(f"Using {lr_method} as scheduler")

            return [optimizer], schedulers_list
        else:
            return optimizer

    def _init_decoder_losses(self):
        datasets = self.hparams["dataset"]["datasets"]
        task_to_datasets = defaultdict(list)
        for dataset, tasks in datasets.items():
            for task in tasks:
                task_to_datasets[task].append(dataset)

        losses = {}
        for task, decoder in self.decoders.items():
            for fidelity in decoder.context:
                for loss in decoder.loss_keys:
                    key = f"{fidelity}/{decoder.name}/{loss}/"
                    losses[key] = torch.tensor(0.0, device=self.device)

        sorted_losses = dict(sorted(losses.items(), key=lambda item: item[0]))
        return sorted_losses

    def on_fit_start(self):
        self.losses_template = self._init_decoder_losses()
        if not self.synced_buffers:  # This should only run when loading from checkpoint
            self.sync_buffers()

    def on_validation_model_eval(self) -> None:
        torch.set_grad_enabled(True)

    def _register_datasets(self):
        if self.global_rank == 0:
            if self.hparams.dataset.get("constants_path") is not None:
                with open(self.hparams.dataset["constants_path"], "r") as file:
                    constants = yaml.safe_load(file)
            else:
                for callback in self.trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint):
                        checkpoint_callback = callback
                        break
                os.makedirs(checkpoint_callback.dirpath, exist_ok=True)
                new_constants_path = join(checkpoint_callback.dirpath, "constants.yaml")
                open(new_constants_path, "a+").close()
                self.constants_file_path = new_constants_path

            for _, decoder in (pbar := tqdm(self.decoders.items())):
                task_name = decoder.task
                pbar.set_description(f"Registering datasets for {decoder.id}")

                if "max_elements_for_constants" in self.hparams.dataset:
                    whole_batch = self.trainer.datamodule.batch_for_registering_task(task_name)
                    self.encoder.register_datasets(dataset_batch=whole_batch, file_to_store_constants=new_constants_path)
                    decoder.register_datasets(dataset_batch=whole_batch, file_to_store_constants=new_constants_path)
                elif self.hparams.dataset.get("constants_path") is not None:
                    assert task_name in constants, f"{task_name} is not included in constants file"
                    self.encoder.register_datasets(precomputed_constants=constants[task_name])
                    decoder.register_datasets(precomputed_constants=constants[task_name])
                else:
                    raise Exception("Please specify either constants_path or max_elements_for_constants")

                decoder.to(self.device)
            self.encoder.setup()
            self.encoder.to(self.device)

            if "max_elements_for_constants" in self.hparams.dataset:
                with open(new_constants_path, "r") as file:
                    existing_content = file.read()

                with open(new_constants_path, "w") as file:
                    file.write(
                        f"# Constant file created with seed {self.hparams.general['seed']} and maximum dataset size of {self.hparams.dataset['max_elements_for_constants']} \n"
                    )
                    file.write(existing_content)

        self.sync_buffers()

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
        n_tasks = len(self._init_decoder_losses())
        scalarizer_method = SCALARIZER_CLASS_MAP[self.hparams.optimizer["scalarization"]["method"]]
        self.scalarizer = scalarizer_method(n_tasks=n_tasks, device=self.device, **self.hparams.optimizer["scalarization"]["args"])
        scalarizer_state_dict = {k[len("scalarizer.") :]: v for k, v in scalarizer_keys.items() if k.startswith("scalarizer.")}
        self.scalarizer.load_state_dict(scalarizer_state_dict, strict=False)

        self.starting_from_checkpoint = True
        self.strict_loading = False

    def get_constants_file_path(self):
        return self.constants_file_path

    def on_train_batch_end(self, *args, **kwargs):
        if self.scalarizer.dynamic:
            with torch.no_grad():
                losses = self.latest_task_losses
                self.scalarizer.update(losses)

                if hasattr(self.scalarizer, "weights"):
                    for n, w in zip(self.latest_task_losses.keys(), self.scalarizer.weights):
                        self.log(f"{n}/Weight", w, on_step=True, sync_dist=True)

    def on_train_epoch_start(self) -> None:
        freeze_epochs = self.hparams.get("fine-tuning", {}).get("freeze_epochs")
        if hasattr(self, "frozen_encoder") and freeze_epochs == self.current_epoch:  # unfreeze model
            self.configure_optimizers = self.unfreeze_configure_optimizers
            self.trainer.strategy.setup_optimizers(self.trainer)

    def unfreeze_configure_optimizers(self):
        optimizer_class = OPTIMIZER_CLASS_MAP[self.hparams.optimizer["method"]]
        new_lr = self.hparams.get("fine-tuning", {}).get("unfreeze_lr")
        if new_lr is not None:
            kwargs = {k: v for k, v in self.hparams.optimizer["args"].items() if k != "lr"}
            optimizer = optimizer_class(self.parameters(), lr=new_lr, **kwargs)
            print(f"unfreezing encoder with lr: {new_lr}")
        else:
            optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer["args"])
            print(f"unfreezing encoder with lr: {self.hparams.optimizer['args']['lr']}")

        return optimizer
