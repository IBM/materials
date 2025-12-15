# Data
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from .utils import get_lr, get_order, get_tqdm_eta, LoadCheckpointMode, AvgMeter

# Torch
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(
        self,
        init_datetime,
        model: torch.nn.Module,
        datasets: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        save_every,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        load_checkpoint_mode: str,
        load_checkpoint_filename: str,
        config,
        total_gpus=0
    ) -> None:
        self.init_datetime = init_datetime
        self.config = config
        self.total_gpus = total_gpus
        self.local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)

        # data
        self.train_data = datasets[0]
        self.valid_data = datasets[1]

        # optimizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # checkpoint
        self.save_every = save_every
        self.start_epoch = 1
        self.last_batch_idx = -1
        self.save_checkpoint_path = save_checkpoint_path

        if not os.path.exists(self.save_checkpoint_path):
            os.mkdir(self.save_checkpoint_path)

        ckpt_mode = LoadCheckpointMode[load_checkpoint_mode.upper()]
        if ckpt_mode != LoadCheckpointMode.SKIP:
            self._load_checkpoint(ckpt_mode, load_checkpoint_path, load_checkpoint_filename)

        self.model = DDP(self.model, find_unused_parameters=False)

    def _load_checkpoint(self, checkpoint_mode, checkpoint_path, checkpoint_filename):
        print("Loading checkpoint...")
        loc = f"cuda:{self.local_rank}"

        if checkpoint_mode == LoadCheckpointMode.LAST:
            last_ckpt = sorted([filename for filename in os.listdir(checkpoint_path) if filename.endswith(".pt")], key=get_order)[-1]
            checkpoint_filename = last_ckpt
            ckpt_dict = torch.load(os.path.join(checkpoint_path, checkpoint_filename), map_location=loc)
        else:
            ckpt_dict = torch.load(os.path.join(checkpoint_path, checkpoint_filename), map_location=loc)

        # load keys
        self.model.load_state_dict(ckpt_dict["MODEL_STATE"])
        self.optimizer.load_state_dict(ckpt_dict["optimizer"])
        self.last_batch_idx = ckpt_dict["last_batch_idx"] if 'last_batch_idx' in ckpt_dict else -1
        self.start_epoch = ckpt_dict["EPOCHS_RUN"] + 1 if self.last_batch_idx == -1 else ckpt_dict["EPOCHS_RUN"]

        # load RNG states each time the model and states are loaded from checkpoint
        if 'rng' in ckpt_dict:
            rng = ckpt_dict['rng']
            for key, value in rng.items():
                if key =='torch_state':
                    torch.set_rng_state(value.cpu())
                elif key =='cuda_state':
                    torch.cuda.set_rng_state(value.cpu())
                elif key =='numpy_state':
                    np.random.set_state(value)
                elif key =='python_state':
                    random.setstate(value)
                else:
                    print('unrecognized state')

        print(f"Sucessfully restored checkpoint {os.path.join(checkpoint_path, checkpoint_filename)} at epoch {self.start_epoch}.")

    def _save_checkpoint(self, epoch, last_idx, metadata):
        # save RNG states each time the model and states are saved
        rng_dict = dict()
        rng_dict['torch_state'] = torch.get_rng_state()
        rng_dict['cuda_state'] = torch.cuda.get_rng_state()
        if np:
            rng_dict['numpy_state'] = np.random.get_state()
        if random:
            rng_dict['python_state'] = random.getstate()

        # hparams
        hparams = {
            'model': dict(self.config),
            'text_encoder': self.model.module.text_encoder.get_hparams(),
            'image_encoder': self.model.module.image_encoder.get_hparams(),
        }

        # save checkpoint dict
        checkpoint = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "optimizer": self.optimizer.state_dict(),
            "hparams": hparams,
            "last_batch_idx": last_idx,
            "rng": rng_dict
        }

        if last_idx == -1:
            filename = f'{str(self.model.module)}_{epoch}_{self.init_datetime}.pt'
        else:
            filename = f'{str(self.model.module)}_{last_idx}_{epoch}_{self.init_datetime}.pt'

        torch.save(checkpoint, os.path.join(self.save_checkpoint_path, filename))

        # save metadata
        epoch_metrics = pd.DataFrame({
            'train_avg_loss': [metadata['train_avg_loss']],
            'valid_avg_loss': [metadata['valid_avg_loss']],
            'train_eta': [metadata['train_eta']],
            'valid_eta': [metadata['valid_eta']],
            'TOTAL_GPUs': [self.total_gpus],
        })
        epoch_metrics.to_csv(os.path.join(self.save_checkpoint_path, os.path.splitext(filename)[0]+'.log'), index=False)
        
        print(f"Epoch {epoch} | Training checkpoint saved at {os.path.join(self.save_checkpoint_path, filename)}.")

    def train(self, max_epochs: int):
        for epoch in range(self.start_epoch, max_epochs+1):
            metadata = self._run_epoch(epoch)
            if self.global_rank == 0:
                self._save_checkpoint(epoch, last_idx=-1, metadata=metadata)

    def _run_epoch(self, epoch):
        metadata = {
            'train_avg_loss': 0.,
            'valid_avg_loss': 0.,
            'train_eta': 0.,
            'valid_eta': 0.
        }
        valid_losses = None

        self.train_data.sampler.set_epoch(epoch)
        self.valid_data.sampler.set_epoch(epoch)

        if self.global_rank == 0:
            train_pbar = tqdm(total=len(self.train_data))

        # training data
        train_loss_meter = AvgMeter()
        train_losses = pd.DataFrame()
        for step, data in enumerate(self.train_data):
            # skip batches
            if step <= self.last_batch_idx:
                continue

            self.model.train()
            loss = self._run_batch(data, step=self.config.train_step)

            # update loss meter
            count = data["image"].size(0)
            train_loss_meter.update(loss, count)
            metadata['train_avg_loss'] = train_loss_meter.avg

            if self.global_rank == 0:
                # track loss
                df = pd.DataFrame({'loss': [loss]})
                train_losses = pd.concat([train_losses, df], axis=0)

                # update training ETA
                train_eta = get_tqdm_eta(train_pbar)
                metadata['train_eta'] = train_eta

                # progress bar
                train_pbar.update(1)
                train_pbar.set_description(f'[TRAINING] Epoch:{epoch} | Batch size:{self.config.batch_size}')
                train_pbar.set_postfix(
                    lr=get_lr(self.optimizer),
                    loss=train_loss_meter.avg
                )
                train_pbar.refresh()

            # intermediate checkpoint
            if self.global_rank == 0 and step % self.save_every == 0 and step != 0:
                self._save_checkpoint(epoch, step, metadata)
                # WARN: due to job limit time - save loss for each iter
                train_losses.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_{step}_epoch{epoch}.csv'), index=False)
                train_losses = pd.Series()

        self.last_batch_idx = -1

        # validation data
        if len(self.valid_data) > 0:
            if self.global_rank == 0:
                valid_pbar = tqdm(total=len(self.valid_data))
            valid_loss_meter = AvgMeter()
            valid_losses = pd.DataFrame()
            for data in self.valid_data:
                # inference
                with torch.no_grad():
                    self.model.eval()
                    loss = self.model(data, self.local_rank).item()

                # update loss meter
                count = data["image"].size(0)
                valid_loss_meter.update(loss, count)
                metadata['valid_avg_loss'] = valid_loss_meter.avg

                if self.global_rank == 0:
                    # track loss
                    df = pd.DataFrame({'loss': [loss]})
                    valid_losses = pd.concat([valid_losses, df], axis=0)

                    # update validation ETA
                    valid_eta = get_tqdm_eta(valid_pbar)
                    metadata['valid_eta'] = valid_eta

                    # progress bar
                    valid_pbar.update(1)
                    valid_pbar.set_description(f'[VALIDATION] Epoch:{epoch} | Batch size:{self.config.valid_batch_size}')
                    valid_pbar.set_postfix(loss=valid_loss_meter.avg)
                    valid_pbar.refresh()

        # save logs
        if self.global_rank == 0:
            train_pbar.close()
            train_losses.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_epoch{epoch}.csv'), index=False)
            if valid_losses is not None:
                valid_pbar.close()
                valid_losses.to_csv(os.path.join(self.save_checkpoint_path, f'validation_losses_epoch{epoch}.csv'), index=False)

        return metadata

    def _run_batch(self, batch, step):
        self.optimizer.zero_grad()
        
        # Forward pass and loss computation
        loss = self.model(batch, self.local_rank)
        
        # Backward pass
        loss.backward()
        
        # Update optimizer parameters
        self.optimizer.step()
        
        # Update learning rate scheduler if step is "batch"
        if step == "batch":
            self.lr_scheduler.step()

        return loss.item()