# Data
import pandas as pd
import numpy as np
import os
import gc
import random
from tqdm import tqdm

# Torch
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        datasets: DataLoader,
        optimizers: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config
    ) -> None:
        self.config = config
        self.local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)

        # data
        self.train_data = datasets[0]
        self.valid_data = datasets[1]

        # optimizers
        self.opt_ae = optimizers[0]
        self.opt_disc = optimizers[1]
        
        # mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

        # checkpoint
        self.save_every = save_every
        self.epochs_run = 1
        self.last_batch_idx = -1
        self.save_checkpoint_path = save_checkpoint_path

        if os.path.exists(load_checkpoint_path):
            print("Loading checkpoint")
            self._load_checkpoint(load_checkpoint_path)

        self.model = DDP(self.model, find_unused_parameters=True)

    def _load_checkpoint(self, checkpoint_path):
        loc = f"cuda:{self.local_rank}"
        ckpt_dict = torch.load(checkpoint_path, map_location=loc)
        self.model.load_state_dict(ckpt_dict["MODEL_STATE"])
        self.opt_ae.load_state_dict(ckpt_dict["optimizer"][0])
        self.opt_disc.load_state_dict(ckpt_dict["optimizer"][1])
        self.scaler.load_state_dict(ckpt_dict["scaler"])
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

        print(f"Resuming training from checkpoint at Epoch {self.epochs_run}")

    def _save_checkpoint(self, epoch, config, last_idx):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state'] = torch.get_rng_state()
        out_dict['cuda_state'] = torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state'] = np.random.get_state()
        if random:
            out_dict['python_state'] = random.getstate()

        checkpoint = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "optimizer": [self.opt_ae.state_dict(), self.opt_disc.state_dict()],
            "scaler": self.scaler.state_dict(),
            "hparams": vars(config),
            "last_batch_idx": last_idx,
            "rng": out_dict
        }

        if last_idx == -1:
            filename = f'VQGAN_{epoch}.pt'
        else:
            filename = f'VQGAN_{last_idx}_{epoch}.pt'

        torch.save(checkpoint, os.path.join(self.save_checkpoint_path, filename))
        
        print(f"Epoch {epoch} | Training checkpoint saved at {os.path.join(self.save_checkpoint_path, filename)}.")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs+1):
            self._run_epoch(epoch)
            if self.global_rank == 0:
                self._save_checkpoint(epoch, self.config, last_idx=-1)

    def _run_epoch(self, epoch):
        # b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {self.config.model.batch_size} | Steps: {len(self.train_data)} | LastIdx: {self.last_batch_idx}")

        self.train_data.sampler.set_epoch(epoch)
        # self.valid_data.sampler.set_epoch(epoch)

        # training data
        train_losses = pd.DataFrame()
        for step, data in enumerate(tqdm(self.train_data)):
            # skip batches
            if step <= self.last_batch_idx:
                continue

            self.model.train()

            x_train = data['data'].to(self.local_rank)
            global_step = step * epoch
            recon_loss, aeloss, perceptual_loss, g_image_loss, image_gan_feat_loss, commitment_loss, perplexity = self._run_batch(global_step, x_train)

            # track loss
            if self.global_rank == 0:
                df = pd.DataFrame({
                    'recon_loss': [recon_loss.detach().cpu().item()], 
                    'aeloss': [aeloss.detach().cpu().item()], 
                    'perceptual_loss': [perceptual_loss.detach().cpu().item()], 
                    'g_image_loss': [g_image_loss.detach().cpu().item()], 
                    'image_gan_feat_loss': [image_gan_feat_loss.detach().cpu().item()],
                    'perplexity': [perplexity.detach().cpu().item()], 
                    'commitment_loss': [commitment_loss.detach().cpu().item()]
                })
                train_losses = pd.concat([train_losses, df], axis=0)
                print(f"[Training] recon_loss={recon_loss.item()}, aeloss={aeloss.item()}, perceptual_loss={perceptual_loss.item()}, g_image_loss={g_image_loss.item()}, image_gan_feat_loss={image_gan_feat_loss.item()}, perplexity={perplexity.item()}, commitment_loss={commitment_loss.item()}")

            # checkpoint
            if self.global_rank == 0 and step % self.save_every == 0 and step != 0:
                self._save_checkpoint(epoch, self.config, step)
                # WARN: due to job limit time - save loss for each iter
                train_losses.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_{step}_epoch{epoch}.csv'), index=False)
                train_losses = pd.Series()

        # TODO: use a properly validation split
        # validation data
        # val_losses = pd.DataFrame()
        # with torch.no_grad():
        #     for step, data in enumerate(self.valid_data):
        #         self.model.eval()
        #         x_valid = data['data'].to(self.local_rank)
        #         global_step = step * epoch
        #         recon_loss, _, vq_output, perceptual_loss = self.model.forward(global_step, x_valid, gpu_id=self.local_rank)

        #         # clear GPU memory
        #         torch.cuda.empty_cache()
        #         gc.collect()

        #         if self.local_rank == 0:
        #             df = pd.DataFrame({
        #                 'recon_loss': [recon_loss.detach().cpu().item()], 
        #                 'perceptual_loss': [perceptual_loss.detach().cpu().item()], 
        #                 'perplexity': [vq_output['perplexity'].detach().cpu().item()], 
        #                 'commitment_loss': [vq_output['commitment_loss'].detach().cpu().item()]
        #             })
        #             val_losses = pd.concat([val_losses, df], axis=0)
        #             print(f"[Validation] recon_loss={recon_loss.item()}, perceptual_loss={perceptual_loss.item()}, perplexity={vq_output['perplexity'].item()}, commitment_loss={vq_output['commitment_loss'].item()}")
            
        self.last_batch_idx = -1

        # save logs
        if self.global_rank == 0:
            train_losses.to_csv(os.path.join(self.save_checkpoint_path, f'training_loss_epoch{epoch}.csv'), index=False)
            # val_losses.to_csv(os.path.join(self.save_checkpoint_path, f'validation_losses_epoch{epoch}.csv'), index=False)

    def _run_batch(self, global_step, x):
        # autoencoder optimization
        self.opt_ae.zero_grad()
        recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss, others = self.model.forward(global_step, x, optimizer_idx=0, gpu_id=self.local_rank)
        commitment_loss = vq_output['commitment_loss']
        loss_ae = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        loss_ae.backward()
        self.opt_ae.step()

        # disc optimization
        self.opt_disc.zero_grad()
        loss_disc = self.model.forward(global_step, x, optimizer_idx=1, gpu_id=self.local_rank)
        loss_disc.backward()
        self.opt_disc.step()

        g_image_loss, image_gan_feat_loss, commitment_loss, perplexity = others[0], others[1], others[2], others[3]

        return recon_loss, aeloss, perceptual_loss, g_image_loss, image_gan_feat_loss, commitment_loss, perplexity
