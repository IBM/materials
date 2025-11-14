# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset.default import GridDataset
from utils import RMSELoss

# Data
import pandas as pd
import numpy as np

# Standard library
import random
import args
import os
import copy
import shutil
from tqdm import tqdm

# Machine Learning
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from scipy import stats
from utils import RMSE, sensitivity, specificity


class Trainer:

    def __init__(self, raw_data, grids_path, dataset_name, target, batch_size, hparams, internal_resolution,
                 target_metric='rmse', seed=0, num_workers=0, checkpoints_folder='./checkpoints', restart_filename=None, save_every_epoch=False, save_ckpt=True, device='cpu'):
        # data
        self.df_train = raw_data[0]
        self.df_valid = raw_data[1]
        self.df_test = raw_data[2]
        self.grids_path = grids_path
        self.dataset_name = dataset_name
        self.target = target
        self.batch_size = batch_size
        self.hparams = hparams
        self.internal_resolution = internal_resolution
        self.num_workers = num_workers
        self._prepare_data()

        # config
        self.target_metric = target_metric
        self.seed = seed
        self.checkpoints_folder = checkpoints_folder
        self.restart_filename = restart_filename
        self.start_epoch = 1
        self.save_every_epoch = save_every_epoch
        self.save_ckpt = save_ckpt
        self.best_vloss = float('inf')
        self.last_filename = None
        self._set_seed(seed)

        # multi-gpu
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

    def _prepare_data(self):
        train_dataset = GridDataset(
            dataset=self.df_train,
            target=self.target,
            root_dir=self.grids_path, 
            internal_resolution=self.internal_resolution,
        )
        valid_dataset = GridDataset(
            dataset=self.df_valid,
            target=self.target,
            root_dir=self.grids_path, 
            internal_resolution=self.internal_resolution,
        )
        test_dataset = GridDataset(
            dataset=self.df_test,
            target=self.target,
            root_dir=self.grids_path, 
            internal_resolution=self.internal_resolution,
        )

        # create dataloader
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(train_dataset), 
            shuffle=False, 
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(valid_dataset), 
            shuffle=False, 
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=DistributedSampler(test_dataset), 
            shuffle=False, 
            pin_memory=True
        )

    def compile(self, model, optimizer, loss_fn):
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self._print_configuration()
        
        if self.restart_filename:
            self._load_checkpoint(self.restart_filename)
            print('Checkpoint restored!')

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def fit(self, max_epochs=500):
        for epoch in range(self.start_epoch, max_epochs+1):
            print(f'\n=====Epoch [{epoch}/{max_epochs}]=====')

            # training
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)
            train_loss = self._train_one_epoch()

            # validation
            self.model.eval()
            val_preds, val_loss, val_metrics = self._validate_one_epoch(self.valid_loader)
            tst_preds, tst_loss, tst_metrics = self._validate_one_epoch(self.test_loader)

            if self.global_rank == 0:
                for m in val_metrics.keys():
                    print(f"[VALID] Evaluation {m.upper()}: {round(val_metrics[m], 4)}")
                print('-'*64)
                for m in tst_metrics.keys():
                    print(f"[TEST] Evaluation {m.upper()}: {round(tst_metrics[m], 4)}")

            ############################### Save Finetune checkpoint #######################################
            if ((val_loss < self.best_vloss) or self.save_every_epoch) and self.save_ckpt and self.global_rank == 0:
                # remove old checkpoint
                if (self.last_filename != None) and (not self.save_every_epoch):
                    os.remove(os.path.join(self.checkpoints_folder, self.last_filename))

                # filename
                model_name = f'{str(self.model.module)}-Finetune'
                self.last_filename = f"{model_name}_seed{self.seed}_{self.dataset_name}_epoch={epoch}_valloss={round(val_loss, 4)}.pt"

                # update best loss
                self.best_vloss = val_loss

                # save checkpoint
                print('Saving checkpoint...')
                self._save_checkpoint(epoch, self.last_filename)

    def evaluate(self, verbose=True):
        if verbose:
            print("\n=====Test Evaluation=====")

        # set model evaluation mode
        model_inf = copy.deepcopy(self.model)
        model_inf.eval()

        # evaluate on test set
        tst_preds, tst_loss, tst_metrics = self._validate_one_epoch(self.test_loader, model_inf)

        if verbose and self.global_rank == 0:
            # show metrics
            for m in tst_metrics.keys():
                print(f"[TEST] Evaluation {m.upper()}: {round(tst_metrics[m], 4)}")

            # save predictions
            pd.DataFrame(tst_preds).to_csv(
                os.path.join(
                    self.checkpoints_folder, 
                    f'{self.dataset_name}_{self.target if isinstance(self.target, str) else self.target[0]}_predict_test_seed{self.seed}.csv'
                ), 
                index=False
            )

    def _train_one_epoch(self):
        raise NotImplementedError

    def _validate_one_epoch(self, data_loader, model=None):
        raise NotImplementedError

    def _print_configuration(self):
        print('----Finetune information----')
        print('Dataset:\t', self.dataset_name)
        print('Target:\t\t', self.target)
        print('Batch size:\t', self.batch_size)
        print('LR:\t\t', self._get_lr())
        print('Device:\t\t', self.local_rank)
        print('Optimizer:\t', self.optimizer.__class__.__name__)
        print('Loss function:\t', self.loss_fn.__class__.__name__)
        print('Seed:\t\t', self.seed)
        print('Train size:\t', self.df_train.shape[0])
        print('Valid size:\t', self.df_valid.shape[0])
        print('Test size:\t', self.df_test.shape[0])

    def _load_checkpoint(self, filename):
        ckpt_path = os.path.join(self.checkpoints_folder, filename)
        ckpt_dict = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt_dict['MODEL_STATE'])
        self.start_epoch = ckpt_dict['EPOCHS_RUN'] + 1
        self.best_vloss = ckpt_dict['finetune_info']['best_vloss']

    def _save_checkpoint(self, current_epoch, filename):
        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)

        self.model.module.config['finetune'] = vars(self.hparams)
        hparams = self.model.module.config

        ckpt_dict = {
            'MODEL_STATE': self.model.module.state_dict(),
            'EPOCHS_RUN': current_epoch,
            'hparams': hparams,
            'finetune_info': {
                'dataset': self.dataset_name,
                'target`': self.target,
                'batch_size': self.batch_size,
                'lr': self._get_lr(),
                'device': self.local_rank,
                'optim': self.optimizer.__class__.__name__,
                'loss_fn': self.loss_fn.__class__.__name__,
                'train_size': self.df_train.shape[0],
                'valid_size': self.df_valid.shape[0],
                'test_size': self.df_test.shape[0],
                'best_vloss': self.best_vloss,
            },
            'seed': self.seed,
        }

        assert list(ckpt_dict.keys()) == ['MODEL_STATE', 'EPOCHS_RUN', 'hparams', 'finetune_info', 'seed']

        torch.save(ckpt_dict, os.path.join(self.checkpoints_folder, filename))

    def _set_seed(self, value):
        random.seed(value)
        torch.manual_seed(value)
        np.random.seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(value)
            torch.cuda.manual_seed_all(value)
            cudnn.deterministic = True
            cudnn.benchmark = False

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


class TrainerRegressor(Trainer):

    def __init__(self, raw_data, grids_path, dataset_name, target, batch_size, hparams, internal_resolution,
                 target_metric='rmse', seed=0, num_workers=0, checkpoints_folder='./checkpoints', restart_filename=None, save_every_epoch=False, save_ckpt=True, device='cpu'):
        super().__init__(raw_data, grids_path, dataset_name, target, batch_size, hparams, internal_resolution,
                         target_metric, seed, num_workers, checkpoints_folder, restart_filename, save_every_epoch, save_ckpt, device) 

    def _train_one_epoch(self):
        running_loss = 0.0

        if self.global_rank == 0:
            pbar = tqdm(total=len(self.train_loader))
        for idx, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            grids, targets = data
            targets = targets.to(self.local_rank)
            grids = grids.to(self.local_rank)

            # zero the parameter gradients (otherwise they are accumulated)
            self.optimizer.zero_grad()

            # Make predictions for this batch
            embeddings = self.model.module.feature_extraction(grids)
            outputs = self.model.module.net(embeddings).squeeze(1)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

            # progress bar
            if self.global_rank == 0:
                pbar.update(1)
                pbar.set_description('[TRAINING]')
                pbar.set_postfix(loss=running_loss/(idx+1))
                pbar.refresh()
        if self.global_rank == 0:
            pbar.close()

        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, data_loader, model=None):
        data_targets = []
        data_preds = []
        running_loss = 0.0

        model = self.model if model is None else model

        if self.global_rank == 0:
            pbar = tqdm(total=len(data_loader))
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                # Every data instance is an input + label pair
                grids, targets = data
                targets = targets.to(self.local_rank)
                grids = grids.to(self.local_rank)

                # Make predictions for this batch
                embeddings = model.module.feature_extraction(grids)
                predictions = model.module.net(embeddings).squeeze(1)

                # Compute the loss
                loss = self.loss_fn(predictions, targets)

                data_targets.append(targets.view(-1))
                data_preds.append(predictions.view(-1))

                # print statistics
                running_loss += loss.item()

                # progress bar
                if self.global_rank == 0:
                    pbar.update(1)
                    pbar.set_description('[EVALUATION]')
                    pbar.set_postfix(loss=running_loss/(idx+1))
                    pbar.refresh()
        if self.global_rank == 0:
            pbar.close()

        # Put together predictions and labels from batches
        preds = torch.cat(data_preds, dim=0).cpu().numpy()
        tgts = torch.cat(data_targets, dim=0).cpu().numpy()

        # Compute metrics
        mae = mean_absolute_error(tgts, preds)
        r2 = r2_score(tgts, preds)
        rmse = RMSE(preds, tgts)
        spearman = stats.spearmanr(tgts, preds).correlation # scipy 1.12.0

        # Rearange metrics
        metrics = {
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'spearman': spearman,
        }

        return preds, running_loss / len(data_loader), metrics