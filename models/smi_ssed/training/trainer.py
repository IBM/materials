# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP

# Data
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Standard library
import os
import random
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer

        self.save_every = save_every
        self.epochs_run = 0
        self.last_batch_idx = -1
        self.padding_idx = 2
        
        self.save_checkpoint_path = save_checkpoint_path
        self.load_checkpoint_path = load_checkpoint_path
        self.config = config

        if os.path.exists(load_checkpoint_path):
            print("Loading checkpoint...")
            self._load_checkpoint(load_checkpoint_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_checkpoint(self, checkpoint_path):
        loc = f"cuda:{self.local_rank}"
        ckpt_dict = torch.load(checkpoint_path, map_location=loc)
        if os.path.exists(os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt')):
            opt_dict = torch.load(os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt'), map_location=loc)
            self.optimizer.load_state_dict(opt_dict["OPTIMIZER_STATE"])
            print('Optimizer states restored!')

        if 'state_dict' in ckpt_dict:
            self.model.encoder.load_state_dict(ckpt_dict["state_dict"][0])
            self.model.decoder.load_state_dict(ckpt_dict["state_dict"][1])
        else:
            self.model.load_state_dict(ckpt_dict["MODEL_STATE"])

        self.last_batch_idx = ckpt_dict["last_batch_idx"] if 'last_batch_idx' in ckpt_dict else -1

        if 'epoch' in ckpt_dict:
            self.epochs_run = ckpt_dict["epoch"] + 1 if self.last_batch_idx == -1 else ckpt_dict["epoch"]
        else:
            self.epochs_run = ckpt_dict["EPOCHS_RUN"] + 1 if self.last_batch_idx == -1 else ckpt_dict["EPOCHS_RUN"]

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
        
        print(f"Resuming training from checkpoint at Epoch {self.epochs_run}.")

    def _save_checkpoint(self, epoch, config, last_idx=-1):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state'] = torch.get_rng_state()
        out_dict['cuda_state'] = torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state'] = np.random.get_state()
        if random:
            out_dict['python_state'] = random.getstate()

        # model states
        ckpt_dict = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "hparams": vars(config),
            "last_batch_idx": last_idx,
            "rng": out_dict
        }

        # optimizer states
        opt_dict = {
            "OPTIMIZER_STATE": self.optimizer.state_dict()
        }

        if last_idx == -1:
            filename = f'{str(self.model.module)}_{epoch}.pt'
        else:
            filename = f'{str(self.model.module)}_{last_idx}_{epoch}.pt'

        torch.save(ckpt_dict, os.path.join(self.save_checkpoint_path, filename))
        torch.save(opt_dict, os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt'))

        print(f"Epoch {epoch} | Training checkpoint saved at {os.path.join(self.save_checkpoint_path, filename)}.")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0:
                self._save_checkpoint(epoch, self.config)

    def _run_epoch(self, epoch):
        pass

    def _run_batch(self, batch_idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked):
        pass


class TrainerEncoderDecoder(Trainer):

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
    ) -> None:
        super().__init__(model, train_data, optimizer, save_every, save_checkpoint_path, load_checkpoint_path, config)
        self.ngpus_per_node = torch.cuda.device_count()
        self.total_batches = len(self.train_data)
        self.batch_thresh = int(self.total_batches - (self.total_batches * 0.01 * self.ngpus_per_node))
        print('batch_thresh:', self.batch_thresh)

        self.criterionC = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterionR = nn.MSELoss()

        self.optimE = self.optimizer[0]
        self.optimD = self.optimizer[1]

    def _load_checkpoint(self, checkpoint_path):
        loc = f"cuda:{self.local_rank}"
        ckpt_dict = torch.load(checkpoint_path, map_location=loc)
        if os.path.exists(os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt')):
            opt_dict = torch.load(os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt'), map_location=loc)
            self.optimizer[0].load_state_dict(opt_dict["OPTIMIZER_STATE_ENCODER"])
            self.optimizer[1].load_state_dict(opt_dict["OPTIMIZER_STATE_DECODER"])
            print('Optimizer states restored!')

        if 'state_dict' in ckpt_dict:
            self.model.encoder.load_state_dict(ckpt_dict["state_dict"][0])
            self.model.decoder.load_state_dict(ckpt_dict["state_dict"][1])
        else:
            self.model.load_state_dict(ckpt_dict["MODEL_STATE"])

        self.last_batch_idx = ckpt_dict["last_batch_idx"] if 'last_batch_idx' in ckpt_dict else -1

        if 'epoch' in ckpt_dict:
            self.epochs_run = ckpt_dict["epoch"] + 1 if self.last_batch_idx == -1 else ckpt_dict["epoch"]
        else:
            self.epochs_run = ckpt_dict["EPOCHS_RUN"] + 1 if self.last_batch_idx == -1 else ckpt_dict["EPOCHS_RUN"]

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
        
        print(f"Resuming training from checkpoint at Epoch {self.epochs_run}.")

    def _save_checkpoint(self, epoch, config, last_idx=-1):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state'] = torch.get_rng_state()
        out_dict['cuda_state'] = torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state'] = np.random.get_state()
        if random:
            out_dict['python_state'] = random.getstate()

        # model states
        ckpt_dict = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "hparams": vars(config),
            "last_batch_idx": last_idx,
            "rng": out_dict
        }

        # optimizer states
        opt_dict = {
            "OPTIMIZER_STATE_ENCODER": self.optimizer[0].state_dict(),
            "OPTIMIZER_STATE_DECODER": self.optimizer[1].state_dict(),
        }

        if last_idx == -1:
            filename = f'{str(self.model.module)}_{epoch}.pt'
        else:
            filename = f'{str(self.model.module)}_{last_idx}_{epoch}.pt'

        torch.save(ckpt_dict, os.path.join(self.save_checkpoint_path, filename))
        torch.save(opt_dict, os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt'))

        print(f"Epoch {epoch} | Training checkpoint saved at {os.path.join(self.save_checkpoint_path, filename)}.")

    def _run_epoch(self, epoch):
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {self.config.n_batch} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        loss_list = pd.DataFrame()

        for idx, data in enumerate(tqdm(self.train_data)):
            bucket_idx_masked       = data[0]
            bucket_targets          = data[1]
            bucket_idx_not_masked   = data[2]
            self.model.train()
            lossE, lossD = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked)

            # track loss
            if self.local_rank == 0:
                df = pd.DataFrame({
                    'lossE': [lossE.cpu().item()],
                    'lossD': [lossD.cpu().item()],
                })
                loss_list = pd.concat([loss_list, df], axis=0)

        if self.local_rank == 0:
            loss_list.to_csv(os.path.join(self.config.save_checkpoint_path, f'training_loss_epoch{epoch}.csv'), index=False)

    def _run_batch(self, batch_idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked):
        self.optimE.zero_grad(set_to_none=True)
        self.optimD.zero_grad(set_to_none=True)

        can_train_encoder = (batch_idx + 1) <= self.batch_thresh
        can_train_decoder = (batch_idx + 1) > self.batch_thresh

        errorE = torch.zeros(1).to(self.local_rank)
        errorD = torch.zeros(1).to(self.local_rank)
        errorE_tmp = .0
        errorD_tmp = .0

        for chunk in range(len(bucket_idx_masked)):
            idx_masked = bucket_idx_masked[chunk].to(self.local_rank)
            targets = bucket_targets[chunk].to(self.local_rank)
            idx_not_masked = bucket_idx_not_masked[chunk]
            idx_not_masked = list(map(lambda x: F.pad(x, pad=(0, self.config.max_len - x.shape[0]), value=2).unsqueeze(0), idx_not_masked))
            idx_not_masked = torch.cat(idx_not_masked, dim=0).to(self.local_rank)
            mask = (idx_masked != self.padding_idx)

            ###########
            # Encoder #
            ###########
            if can_train_encoder:
                for param in self.model.module.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.module.decoder.parameters():
                    param.requires_grad = False

                logits = self.model.module.encoder(idx_masked)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                errorE_tmp = self.criterionC(logits, targets) / len(bucket_idx_masked)

                if chunk < len(bucket_idx_masked)-1:
                    errorE_tmp.backward()
                    errorE += errorE_tmp.detach()
                else:
                    errorE += errorE_tmp
                
            ###########
            # Decoder #
            ###########
            if can_train_decoder:
                for param in self.model.module.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.module.decoder.parameters():
                    param.requires_grad = True
                    
                self.model.module.encoder.eval()

                # encoder forward
                with torch.no_grad():
                    true_set, true_cte = self.model.module.encoder(idx_masked, mask=mask, inference=True)

                # add padding
                input_mask_expanded = mask.unsqueeze(-1).expand(true_cte.size()).float()
                mask_embeddings = (true_cte * input_mask_expanded)
                true_cte = F.pad(mask_embeddings, pad=(0, 0, 0, self.config.max_len - mask_embeddings.shape[1]), value=0)
                true_cte = true_cte.view(-1, self.config.max_len*self.config.n_embd)

                # decoder forward
                pred_set, pred_ids = self.model.module.decoder(true_cte)

                # losses
                pred_ids = pred_ids.view(-1, pred_ids.size(-1))
                true_ids = idx_not_masked.view(-1)

                error_ids = self.criterionC(pred_ids, true_ids) / len(bucket_idx_masked)
                error_set = self.criterionR(pred_set, true_set) / len(bucket_idx_masked)
                errorD_tmp = error_ids + error_set

                if chunk < len(bucket_idx_masked)-1:
                    errorD_tmp.backward()
                    errorD += errorD_tmp.detach()
                else:
                    errorD += errorD_tmp

        if can_train_decoder:
            errorD.backward()
            self.optimD.step()
        elif can_train_encoder:
            errorE.backward()
            self.optimE.step()

        if self.local_rank == 0:
            print(f'LossE: {errorE.item()} | LossD: {errorD.item()}')
        return errorE, errorD


class TrainerDirectDecoder(Trainer):

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
    ) -> None:
        super().__init__(model, train_data, optimizer, save_every, save_checkpoint_path, load_checkpoint_path, config)
        self.criterionC = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterionR = nn.MSELoss()

    def _run_epoch(self, epoch):
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {self.config.n_batch} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        loss_list = pd.DataFrame()

        for idx, data in enumerate(tqdm(self.train_data)):
            bucket_idx_masked       = data[0]
            bucket_targets          = data[1]
            bucket_idx_not_masked   = data[2]
            self.model.train()
            loss = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked)

            # track loss
            if self.local_rank == 0:
                df = pd.DataFrame({
                    'lossD': [loss.cpu().item()],
                })
                loss_list = pd.concat([loss_list, df], axis=0)

        if self.local_rank == 0:
            loss_list.to_csv(os.path.join(self.config.save_checkpoint_path, f'training_loss_epoch{epoch}.csv'), index=False)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def _run_batch(self, batch_idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked):
        self.optimizer.zero_grad(set_to_none=True)
        error = torch.zeros(1).to(self.local_rank)
        error_tmp = .0

        for chunk in range(len(bucket_idx_masked)):
            idx_masked = bucket_idx_masked[chunk].to(self.local_rank)
            targets = bucket_targets[chunk].to(self.local_rank)
            idx_not_masked = bucket_idx_not_masked[chunk]
            idx_not_masked = list(map(lambda x: F.pad(x, pad=(0, self.config.max_len - x.shape[0]), value=2).unsqueeze(0), idx_not_masked))
            idx_not_masked = torch.cat(idx_not_masked, dim=0).to(self.local_rank)
            mask = (idx_masked != self.padding_idx)

            # encoder forward
            token_embeddings = checkpoint.checkpoint(self.custom(self.model.module.encoder.mamba), idx_masked)
            # mean pooling
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            true_set = sum_embeddings / sum_mask
            true_cte = token_embeddings
            del token_embeddings

            # add padding
            input_mask_expanded = mask.unsqueeze(-1).expand(true_cte.size()).float()
            mask_embeddings = (true_cte * input_mask_expanded)
            true_cte = F.pad(mask_embeddings, pad=(0, 0, 0, self.config.max_len - mask_embeddings.shape[1]), value=0)
            true_cte = true_cte.view(-1, self.config.max_len*self.config.n_embd)

            # decoder forward
            pred_set, pred_ids = self.model.module.decoder(true_cte)

            # losses
            pred_ids = pred_ids.view(-1, pred_ids.size(-1))
            true_ids = idx_not_masked.view(-1)

            error_ids = self.criterionC(pred_ids, true_ids) / len(bucket_idx_masked)
            error_set = self.criterionR(pred_set, true_set) / len(bucket_idx_masked)
            error_tmp = error_ids + error_set

            error += error_tmp

        error.backward()
        self.optimizer.step()

        if self.local_rank == 0:
            print(f'Loss: {error.item()}')

        return error