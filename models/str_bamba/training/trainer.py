# Deep learning
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader
from mamba_ssm.modules.block import Block
from str_bamba.bamba import BambaEncoder

# Standard library
import pandas as pd
import numpy as np
import functools
import random
import os
import gc
from tqdm import tqdm


class Trainer:
    
    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: float,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str = '',
        load_checkpoint_path: str = '',
        config = None,
    ) -> None:
        self.model = model
        self.local_device = get_accelerator().device_name(self.model.local_rank)
        self.local_rank = int(os.environ.get("LOCAL_RANK"))

        self.vocab_size = vocab_size

        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.last_batch_idx = -1
        self.save_checkpoint_path = save_checkpoint_path
        self.config = config

        self.last_representation = None

        # restore checkpoint
        if load_checkpoint_path != '':
            print(f"Loading checkpoint at {load_checkpoint_path}...")
            self._load_checkpoint(load_checkpoint_path)

        # create save checkpoint directory
        os.makedirs(self.save_checkpoint_path, exist_ok=True)

    def _load_checkpoint(self, checkpoint_path):
        # opt_dict = None
        # loc = f"cuda:{self.local_rank}"
        # ckpt_dict = torch.load(checkpoint_path, map_location=loc, weights_only=False)
        # if os.path.exists(os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt')):
        #     opt_dict = torch.load(os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt'), map_location=loc, weights_only=False)

        # self.model.load_state_dict(ckpt_dict["MODEL_STATE"])
        # if opt_dict is not None:
        #     self.optimizer.load_state_dict(opt_dict["OPTIMIZER_STATE"])
        #     print('Optimizer states restored!')

        pieces = checkpoint_path.split('_')
        if len(pieces) == 3:
            self.last_representation = pieces[1]

        _, ckpt_dict = self.model.load_checkpoint(self.save_checkpoint_path, checkpoint_path)

        self.last_batch_idx = ckpt_dict["last_batch_idx"] if 'last_batch_idx' in ckpt_dict else -1
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

    def _save_checkpoint(self, epoch, config, last_idx):
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
            # "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "hparams": vars(config),
            "last_batch_idx": last_idx,
            "rng": out_dict
        }

        # optimizer states
        # opt_dict = {
        #     "OPTIMIZER_STATE": self.optimizer.state_dict(),
        # }

        # checkpoint filename
        if last_idx == '':
            filename = f'Bamba_{epoch}'
        else:
            filename = f'Bamba_{last_idx}_{epoch}'

        # save weights and optimizer states
        # torch.save(ckpt_dict, os.path.join(self.save_checkpoint_path, filename))
        # torch.save(opt_dict, os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt'))
        self.model.save_checkpoint(
            self.save_checkpoint_path,
            filename,
            ckpt_dict,
        )

        # print(f"Epoch {epoch} | Training checkpoint saved at {os.path.join(self.save_checkpoint_path, filename)}.")
        print(f"Epoch {epoch} | Training checkpoint saved at {self.save_checkpoint_path}.")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # if self.local_rank == 0:
            self._save_checkpoint(epoch, self.config, last_idx='')

    def _run_epoch(self, epoch):
        raise NotImplementedError

    def _run_batch(self, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, mol_type=''):
        raise NotImplementedError


class TrainerEncoder(Trainer):
    
    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: float,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
    ) -> None:
        super().__init__(model, vocab_size, train_data, optimizer, save_every, save_checkpoint_path, load_checkpoint_path, config)

        self.mf_train_data = train_data[0]
        self.smiles_train_data = train_data[1]
        self.iupac_train_data = train_data[2]
        self.inchi_train_data = train_data[3]
        self.selfies_train_data = train_data[4]
        self.polymer_train_data = train_data[5]
        self.formulation_train_data = train_data[6]

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_nsp = nn.CrossEntropyLoss()

    def _run_epoch(self, epoch):
        print(f"[GPU{self.local_device}] Epoch {epoch} | Batchsize: {self.config.n_batch}")
        self.mf_train_data.sampler.set_epoch(epoch)
        self.smiles_train_data.sampler.set_epoch(epoch)
        self.iupac_train_data.sampler.set_epoch(epoch)
        self.inchi_train_data.sampler.set_epoch(epoch)
        self.selfies_train_data.sampler.set_epoch(epoch)
        self.polymer_train_data.sampler.set_epoch(epoch)
        self.formulation_train_data.sampler.set_epoch(epoch)
        self.model.train()
        loss_list = pd.DataFrame()

        if self.last_representation is None:
            for idx, data in enumerate(tqdm(self.mf_train_data)):
                # run batch
                bucket_idx_masked       = data[0]
                bucket_targets          = data[1]
                bucket_idx_not_masked   = data[2]
                attn_masks              = data[3]
                tokens_type_ids         = data[4]
                nsp_labels              = data[5]
                mf_lossE = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type='mf')
                if self.local_rank == 0:
                    df = pd.DataFrame({
                        'loss_encoder': [mf_lossE.cpu().item()],
                    })
                    loss_list = pd.concat([loss_list, df], axis=0)
                if idx % 500 == 0:
                    gc.collect()
                torch.cuda.empty_cache()

            if self.local_rank == 0:
                loss_list.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_encoder_epoch{epoch}_mf.csv'), index=False)
                loss_list = pd.DataFrame()
            self._save_checkpoint(epoch, self.config, last_idx='mf')

        if self.last_representation is None or self.last_representation == 'mf':
            for idx, data in enumerate(tqdm(self.smiles_train_data)):
                # run batch
                bucket_idx_masked       = data[0]
                bucket_targets          = data[1]
                bucket_idx_not_masked   = data[2]
                attn_masks              = data[3]
                tokens_type_ids         = data[4]
                nsp_labels              = data[5]
                smiles_lossE = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type='smiles')
                if self.local_rank == 0:
                    df = pd.DataFrame({
                        'loss_encoder': [smiles_lossE.cpu().item()],
                    })
                    loss_list = pd.concat([loss_list, df], axis=0)
                if idx % 500 == 0:
                    gc.collect()
                torch.cuda.empty_cache()

            if self.local_rank == 0:
                loss_list.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_encoder_epoch{epoch}_smiles.csv'), index=False)
                loss_list = pd.DataFrame()
            self._save_checkpoint(epoch, self.config, last_idx='smiles')
            self.last_representation = 'smiles'

        if self.last_representation is None or self.last_representation == 'smiles':
            for idx, data in enumerate(tqdm(self.iupac_train_data)):
                # run batch
                bucket_idx_masked       = data[0]
                bucket_targets          = data[1]
                bucket_idx_not_masked   = data[2]
                attn_masks              = data[3]
                tokens_type_ids         = data[4]
                nsp_labels              = data[5]
                iupac_lossE = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type='iupac')
                if self.local_rank == 0:
                    df = pd.DataFrame({
                        'loss_encoder': [iupac_lossE.cpu().item()],
                    })
                    loss_list = pd.concat([loss_list, df], axis=0)
                if idx % 500 == 0:
                    gc.collect()
                torch.cuda.empty_cache()

            if self.local_rank == 0:
                loss_list.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_encoder_epoch{epoch}_iupac.csv'), index=False)
                loss_list = pd.DataFrame()
            self._save_checkpoint(epoch, self.config, last_idx='iupac')
            self.last_representation = 'iupac'

        if self.last_representation is None or self.last_representation == 'iupac':
            for idx, data in enumerate(tqdm(self.inchi_train_data)):
                # run batch
                bucket_idx_masked       = data[0]
                bucket_targets          = data[1]
                bucket_idx_not_masked   = data[2]
                attn_masks              = data[3]
                tokens_type_ids         = data[4]
                nsp_labels              = data[5]
                inchi_lossE = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type='inchi')
                if self.local_rank == 0:
                    df = pd.DataFrame({
                        'loss_encoder': [inchi_lossE.cpu().item()],
                    })
                    loss_list = pd.concat([loss_list, df], axis=0)
                if idx % 500 == 0:
                    gc.collect()
                torch.cuda.empty_cache()

            if self.local_rank == 0:
                loss_list.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_encoder_epoch{epoch}_inchi.csv'), index=False)
                loss_list = pd.DataFrame()
            self._save_checkpoint(epoch, self.config, last_idx='inchi')
            self.last_representation = 'inchi'

        if self.last_representation is None or self.last_representation == 'inchi':
            for idx, data in enumerate(tqdm(self.selfies_train_data)):
                # run batch
                bucket_idx_masked       = data[0]
                bucket_targets          = data[1]
                bucket_idx_not_masked   = data[2]
                attn_masks              = data[3]
                tokens_type_ids         = data[4]
                nsp_labels              = data[5]
                selfies_lossE = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type='selfies')
                if self.local_rank == 0:
                    df = pd.DataFrame({
                        'loss_encoder': [selfies_lossE.cpu().item()],
                    })
                    loss_list = pd.concat([loss_list, df], axis=0)
                if idx % 500 == 0:
                    gc.collect()
                torch.cuda.empty_cache()

            if self.local_rank == 0:
                loss_list.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_encoder_epoch{epoch}_selfies.csv'), index=False)
                loss_list = pd.DataFrame()
            self._save_checkpoint(epoch, self.config, last_idx='selfies')

        for idx, data in enumerate(tqdm(self.polymer_train_data)):
            # run batch
            bucket_idx_masked       = data[0]
            bucket_targets          = data[1]
            bucket_idx_not_masked   = data[2]
            attn_masks              = data[3]
            tokens_type_ids         = data[4]
            nsp_labels              = data[5]
            polymer_lossE = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type='polymer')
            if self.local_rank == 0:
                df = pd.DataFrame({
                    'loss_encoder': [polymer_lossE.cpu().item()],
                })
                loss_list = pd.concat([loss_list, df], axis=0)
            if idx % 500 == 0:
                gc.collect()
            torch.cuda.empty_cache()

        for idx, data in enumerate(tqdm(self.formulation_train_data)):
            # run batch
            bucket_idx_masked       = data[0]
            bucket_targets          = data[1]
            bucket_idx_not_masked   = data[2]
            attn_masks              = data[3]
            tokens_type_ids         = data[4]
            nsp_labels              = data[5]
            formulation_lossE = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type='formulation')
            if self.local_rank == 0:
                df = pd.DataFrame({
                    'loss_encoder': [formulation_lossE.cpu().item()],
                })
                loss_list = pd.concat([loss_list, df], axis=0)
            if idx % 500 == 0:
                gc.collect()
            torch.cuda.empty_cache()

        self.last_representation = None
        
        if self.local_rank == 0:
            loss_list.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_encoder_epoch{epoch}_last.csv'), index=False)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def _run_batch(self, batch_idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, tokens_type_ids, nsp_labels, mol_type=''):
        self.model.zero_grad()

        errorE = torch.zeros(1).to(self.local_device)
        errorE_tmp = .0

        for chunk in range(len(bucket_idx_masked)):
            idx_masked = bucket_idx_masked[chunk].to(self.local_device)
            mask = attn_masks[chunk].to(self.local_device)
            token_type_ids = tokens_type_ids[chunk].to(self.local_device)
            labels = bucket_targets[chunk].to(self.local_device)
            next_sentence_label = nsp_labels[chunk].to(self.local_device)
            # idx_not_masked = bucket_idx_not_masked[chunk]

            # model forward
            output = self.model.encoder(idx_masked, token_type_ids)

            # word logits & NSP logits
            prediction_scores = output.logits
            seq_relationship_score = output.seq_relationship_logits

            # losses
            masked_lm_loss = self.criterion(prediction_scores.view(-1, self.vocab_size), labels.view(-1))
            next_sentence_loss = self.criterion_nsp(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

            errorE_tmp = total_loss / len(bucket_idx_masked)
            errorE += errorE_tmp

            torch.cuda.empty_cache()

        # Compute loss
        self.model.backward(errorE)
        self.model.step()

        if self.local_rank == 0:
            print(f'LossE {mol_type}: {errorE.item()}')
        return errorE


class TrainerDecoder(Trainer):
    
    def __init__(
        self,
        model: torch.nn.Module,
        vocab_size: float,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
    ) -> None:
        super().__init__(model, vocab_size, train_data, optimizer, save_every, save_checkpoint_path, load_checkpoint_path, config)
        self.train_data = train_data
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _run_epoch(self, epoch):
        print(f"[GPU{self.local_device}] Epoch {epoch} | Batchsize: {self.config.n_batch}")
        # self.mf_train_data.sampler.set_epoch(epoch)
        # self.smiles_train_data.sampler.set_epoch(epoch)
        # self.iupac_train_data.sampler.set_epoch(epoch)
        # self.inchi_train_data.sampler.set_epoch(epoch)
        # self.selfies_train_data.sampler.set_epoch(epoch)
        # self.polymer_train_data.sampler.set_epoch(epoch)
        # self.formulation_train_data.sampler.set_epoch(epoch)
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        loss_list = pd.DataFrame()

        for idx, data in enumerate(tqdm(self.train_data)):
            # skip batches
            if idx <= self.last_batch_idx:
                continue

            # run batch
            bucket_idx_masked       = data[0]
            bucket_targets          = data[1]
            bucket_idx_not_masked   = data[2]
            attn_masks              = data[3]
            bucket_alts             = data[4]
            formulation_lossD = self._run_batch(idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, bucket_alts, mol_type='all')
            if self.local_rank == 0:
                df = pd.DataFrame({
                    'loss_decoder': [formulation_lossD.cpu().item()],
                })
                loss_list = pd.concat([loss_list, df], axis=0)
            torch.cuda.empty_cache()

            # checkpoint
            if idx % self.save_every == 0 and idx != 0:
                self._save_checkpoint(epoch, self.config, idx)
                # WARN: due to job limit time - save loss for each iter
                loss_list.to_csv(os.path.join(self.config.save_checkpoint_path, f'training_loss_{idx}_epoch{epoch}.csv'), index=False)
                loss_list = pd.Series()

        self.last_representation = None
        self.last_batch_idx = -1
        
        if self.local_rank == 0:
            loss_list.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_decoder_epoch{epoch}.csv'), index=False)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def prepare_decoder_inputs(self, input_ids, padding_idx=2):
        labels = F.pad(input_ids[:, 1:], pad=(0, input_ids.shape[1] - (input_ids.shape[1]-1)), value=2).to(self.local_rank)
        eos_idx = (input_ids != padding_idx).sum(dim=1).add(-1).unsqueeze(-1)
        decoder_inputs = input_ids.clone().scatter_(1, eos_idx, padding_idx).to(self.local_rank)

        return decoder_inputs, labels

    def _run_batch(self, batch_idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked, attn_masks, bucket_alts, mol_type=''):
        self.model.zero_grad()

        errorD = torch.zeros(1).to(self.local_rank)
        errorD_tmp = .0

        for chunk in range(len(bucket_idx_masked)):
            # idx_masked = bucket_idx_masked[chunk].to(self.local_rank)
            # mask = attn_masks[chunk].to(self.local_rank)
            # labels = bucket_targets[chunk].to(self.local_rank)
            idx_not_masked = bucket_idx_not_masked[chunk]
            alternatives = bucket_alts[chunk]

            decoder_representations = [torch.nn.utils.rnn.pad_sequence([a]+b, batch_first=True, padding_value=2) for a, b in zip(idx_not_masked, alternatives)]
            max_seq = max(map(lambda x: x.shape[1], decoder_representations))
            decoder_representations = torch.cat([torch.nn.functional.pad(t, (0, max_seq - t.shape[1]), value=2) for t in decoder_representations])

            decoder_inputs, labels = self.prepare_decoder_inputs(decoder_representations)
            encoder_inputs = torch.nn.utils.rnn.pad_sequence(idx_not_masked, batch_first=True, padding_value=2)
            # encoder_inputs = torch.repeat_interleave(encoder_inputs, repeats=decoder_inputs.shape[0]//encoder_inputs.shape[0], dim=0).to(self.local_rank)
            encoder_inputs = torch.repeat_interleave(encoder_inputs, repeats=torch.tensor([len(l)+1 for l in alternatives]), dim=0).to(self.local_rank)

            # model forward
            output = self.model.module(encoder_inputs, decoder_inputs)
            prediction_scores = output.logits

            # loss function
            lm_loss = self.criterion(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

            errorD_tmp = lm_loss / len(bucket_idx_masked)
            errorD += errorD_tmp 

            torch.cuda.empty_cache()
        
        # Compute loss
        self.model.backward(errorD)
        self.model.step()

        if self.local_rank == 0:
            print(f'LossD {mol_type}: {errorD.item()}')
        return errorD
