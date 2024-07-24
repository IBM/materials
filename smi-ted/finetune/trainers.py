# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import CustomDataset, CustomDatasetMultitask, RMSELoss, normalize_smiles

# Data
import pandas as pd
import numpy as np

# Standard library
import random
import args
import os
from tqdm import tqdm

# Machine Learning
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from scipy import stats
from utils import RMSE, sensitivity, specificity


class Trainer:

    def __init__(self, raw_data, dataset_name, target, batch_size, hparams,
                 target_metric='rmse', seed=0, checkpoints_folder='./checkpoints', save_ckpt=True, device='cpu'):
        # data
        self.df_train = raw_data[0]
        self.df_valid = raw_data[1]
        self.df_test = raw_data[2]
        self.dataset_name = dataset_name
        self.target = target
        self.batch_size = batch_size
        self.hparams = hparams
        self._prepare_data()

        # config
        self.target_metric = target_metric
        self.seed = seed
        self.checkpoints_folder = checkpoints_folder
        self.save_ckpt = save_ckpt
        self.device = device
        self._set_seed(seed)

    def _prepare_data(self):
        # normalize dataset
        self.df_train['canon_smiles'] = self.df_train['smiles'].apply(normalize_smiles)
        self.df_valid['canon_smiles'] = self.df_valid['smiles'].apply(normalize_smiles)
        self.df_test['canon_smiles'] = self.df_test['smiles'].apply(normalize_smiles)

        self.df_train = self.df_train.dropna(subset=['canon_smiles'])
        self.df_valid = self.df_valid.dropna(subset=['canon_smiles'])
        self.df_test = self.df_test.dropna(subset=['canon_smiles'])

        # create dataloader
        self.train_loader = DataLoader(
            CustomDataset(self.df_train, self.target), 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            CustomDataset(self.df_valid, self.target), 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True
        )
        self.test_loader = DataLoader(
            CustomDataset(self.df_test, self.target), 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True
        )

    def compile(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self._print_configuration()

    def fit(self, max_epochs=500):
        best_vloss = 1000
        best_vmetric = -1

        for epoch in range(1, max_epochs+1):
            print(f'\n=====Epoch [{epoch}/{max_epochs}]=====')

            # training
            self.model.to(self.device)
            self.model.train()
            train_loss = self._train_one_epoch()
            print(f'Training loss: {round(train_loss, 6)}')

            # Evaluate the model
            self.model.eval()
            val_preds, val_loss, val_metrics = self._validate_one_epoch(self.valid_loader)
            tst_preds, tst_loss, tst_metrics = self._validate_one_epoch(self.test_loader)

            print(f"Valid loss: {round(val_loss, 6)}")
            for m in val_metrics.keys():
                print(f"[VALID] Evaluation {m.upper()}: {round(val_metrics[m], 4)}")
            print("-"*32)
            print(f"Test loss: {round(tst_loss, 6)}")
            for m in tst_metrics.keys():
                print(f"[TEST] Evaluation {m.upper()}: {round(tst_metrics[m], 4)}")

            ############################### Save Finetune checkpoint #######################################
            if (val_loss < best_vloss) and self.save_ckpt:
                # remove old checkpoint
                if best_vmetric != -1:
                    os.remove(os.path.join(self.checkpoints_folder, filename))

                # filename
                model_name = f'{str(self.model)}-Finetune'
                metric = round(tst_metrics[self.target_metric], 4)
                filename = f"{model_name}_epoch={epoch}_{self.dataset_name}_seed{self.seed}_{self.target_metric}={metric}.pt"

                # save checkpoint
                print('Saving checkpoint...')
                self._save_checkpoint(epoch, filename)

                # save predictions
                pd.DataFrame(tst_preds).to_csv(
                    os.path.join(
                        self.checkpoints_folder, 
                        f'{self.dataset_name}_{self.target if isinstance(self.target, str) else self.target[0]}_predict_test_seed{self.seed}.csv'), 
                    index=False
                )

                # update best loss
                best_vloss = val_loss
                best_vmetric = metric

    def _train_one_epoch(self):
        raise NotImplementedError

    def _validate_one_epoch(self, data_loader):
        raise NotImplementedError

    def _print_configuration(self):
        print('----Finetune information----')
        print('Dataset:\t', self.dataset_name)
        print('Target:\t\t', self.target)
        print('Batch size:\t', self.batch_size)
        print('LR:\t\t', self._get_lr())
        print('Device:\t\t', self.device)
        print('Optimizer:\t', self.optimizer.__class__.__name__)
        print('Loss function:\t', self.loss_fn.__class__.__name__)
        print('Seed:\t\t', self.seed)
        print('Train size:\t', self.df_train.shape[0])
        print('Valid size:\t', self.df_valid.shape[0])
        print('Test size:\t', self.df_test.shape[0])

    def _save_checkpoint(self, current_epoch, filename):
        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)

        ckpt_dict = {
            'MODEL_STATE': self.model.state_dict(),
            'EPOCHS_RUN': current_epoch,
            'hparams': vars(self.hparams),
            'finetune_info': {
                'dataset': self.dataset_name,
                'target`': self.target,
                'batch_size': self.batch_size,
                'lr': self._get_lr(),
                'device': self.device,
                'optim': self.optimizer.__class__.__name__,
                'loss_fn': self.loss_fn.__class__.__name__,
                'train_size': self.df_train.shape[0],
                'valid_size': self.df_valid.shape[0],
                'test_size': self.df_test.shape[0],
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

    def __init__(self, raw_data, dataset_name, target, batch_size, hparams,
                 target_metric='rmse', seed=0, checkpoints_folder='./checkpoints', save_ckpt=True, device='cpu'):
        super().__init__(raw_data, dataset_name, target, batch_size, hparams,
                         target_metric, seed, checkpoints_folder, save_ckpt, device) 

    def _train_one_epoch(self):
        running_loss = 0.0

        for data in tqdm(self.train_loader):
            # Every data instance is an input + label pair
            smiles, targets = data
            targets = targets.clone().detach().to(self.device)

            # zero the parameter gradients (otherwise they are accumulated)
            self.optimizer.zero_grad()

            # Make predictions for this batch
            embeddings = self.model.extract_embeddings(smiles).to(self.device)
            outputs = self.model.net(embeddings).squeeze()

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, data_loader):
        data_targets = []
        data_preds = []
        running_loss = 0.0

        with torch.no_grad():
            for data in tqdm(data_loader):
                # Every data instance is an input + label pair
                smiles, targets = data
                targets = targets.clone().detach().to(self.device)

                # Make predictions for this batch
                embeddings = self.model.extract_embeddings(smiles).to(self.device)
                predictions = self.model.net(embeddings).squeeze()

                # Compute the loss
                loss = self.loss_fn(predictions, targets)

                data_targets.append(targets.view(-1))
                data_preds.append(predictions.view(-1))

                # print statistics
                running_loss += loss.item()

        # Put together predictions and labels from batches
        preds = torch.cat(data_preds, dim=0).cpu().numpy()
        tgts = torch.cat(data_targets, dim=0).cpu().numpy()

        # Compute metrics
        mae = mean_absolute_error(tgts, preds)
        r2 = r2_score(tgts, preds)
        rmse = RMSE(preds, tgts)
        spearman = stats.spearmanr(tgts, preds).statistic # scipy 1.12.0

        # Rearange metrics
        metrics = {
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'spearman': spearman,
        }

        return preds, running_loss / len(self.train_loader), metrics


class TrainerClassifier(Trainer):

    def __init__(self, raw_data, dataset_name, target, batch_size, hparams,
                 target_metric='roc-auc', seed=0, checkpoints_folder='./checkpoints', save_ckpt=True, device='cpu'):
        super().__init__(raw_data, dataset_name, target, batch_size, hparams,
                         target_metric, seed, checkpoints_folder, save_ckpt, device) 

    def _train_one_epoch(self):
        running_loss = 0.0

        for data in tqdm(self.train_loader):
            # Every data instance is an input + label pair
            smiles, targets = data
            targets = targets.clone().detach().to(self.device)

            # zero the parameter gradients (otherwise they are accumulated)
            self.optimizer.zero_grad()

            # Make predictions for this batch
            embeddings = self.model.extract_embeddings(smiles).to(self.device)
            outputs = self.model.net(embeddings).squeeze()

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, targets.long())
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, data_loader):
        data_targets = []
        data_preds = []
        running_loss = 0.0

        with torch.no_grad():
            for data in tqdm(data_loader):
                # Every data instance is an input + label pair
                smiles, targets = data
                targets = targets.clone().detach().to(self.device)

                # Make predictions for this batch
                embeddings = self.model.extract_embeddings(smiles).to(self.device)
                predictions = self.model.net(embeddings).squeeze()

                # Compute the loss
                loss = self.loss_fn(predictions, targets.long())

                data_targets.append(targets.view(-1))
                data_preds.append(predictions)

                # print statistics
                running_loss += loss.item()

        # Put together predictions and labels from batches
        preds = torch.cat(data_preds, dim=0).cpu().numpy()
        tgts = torch.cat(data_targets, dim=0).cpu().numpy()

        # Compute metrics
        preds_cpu = F.softmax(torch.tensor(preds), dim=1).cpu().numpy()[:, 1]

        # accuracy
        y_pred = np.where(preds_cpu >= 0.5, 1, 0)
        accuracy = accuracy_score(tgts, y_pred)

        # sensitivity
        sn = sensitivity(tgts, y_pred)

        # specificity
        sp = specificity(tgts, y_pred)

        # roc-auc
        fpr, tpr, _ = roc_curve(tgts, preds_cpu)
        roc_auc = auc(fpr, tpr)

        # prc-auc
        precision, recall, _ = precision_recall_curve(tgts, preds_cpu)
        prc_auc = auc(recall, precision)

        # Rearange metrics
        metrics = {
            'acc': accuracy,
            'roc-auc': roc_auc,
            'prc-auc': prc_auc,
            'sensitivity': sn,
            'specificity': sp,
        }

        return preds, running_loss / len(self.train_loader), metrics


class TrainerClassifierMultitask(Trainer):

    def __init__(self, raw_data, dataset_name, target, batch_size, hparams,
                 target_metric='roc-auc', seed=0, checkpoints_folder='./checkpoints', save_ckpt=True, device='cpu'):
        super().__init__(raw_data, dataset_name, target, batch_size, hparams,
                         target_metric, seed, checkpoints_folder, save_ckpt, device)

    def _prepare_data(self):
        # normalize dataset
        self.df_train['canon_smiles'] = self.df_train['smiles'].apply(normalize_smiles)
        self.df_valid['canon_smiles'] = self.df_valid['smiles'].apply(normalize_smiles)
        self.df_test['canon_smiles'] = self.df_test['smiles'].apply(normalize_smiles)

        self.df_train = self.df_train.dropna(subset=['canon_smiles'])
        self.df_valid = self.df_valid.dropna(subset=['canon_smiles'])
        self.df_test = self.df_test.dropna(subset=['canon_smiles'])

        # create dataloader
        self.train_loader = DataLoader(
            CustomDatasetMultitask(self.df_train, self.target), 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            CustomDatasetMultitask(self.df_valid, self.target), 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True
        )
        self.test_loader = DataLoader(
            CustomDatasetMultitask(self.df_test, self.target), 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True
        )

    def _train_one_epoch(self):
        running_loss = 0.0

        for data in tqdm(self.train_loader):
            # Every data instance is an input + label pair + mask
            smiles, targets, target_masks = data
            targets = targets.clone().detach().to(self.device)

            # zero the parameter gradients (otherwise they are accumulated)
            self.optimizer.zero_grad()

            # Make predictions for this batch
            embeddings = self.model.extract_embeddings(smiles).to(self.device)
            outputs = self.model.net(embeddings, multitask=True).squeeze()
            outputs = outputs * target_masks.to(self.device)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, data_loader):
        data_targets = []
        data_preds = []
        data_masks = []
        running_loss = 0.0

        with torch.no_grad():
            for data in tqdm(data_loader):
                # Every data instance is an input + label pair + mask
                smiles, targets, target_masks = data
                targets = targets.clone().detach().to(self.device)

                # Make predictions for this batch
                embeddings = self.model.extract_embeddings(smiles).to(self.device)
                predictions = self.model.net(embeddings, multitask=True).squeeze()
                predictions = predictions * target_masks.to(self.device)

                # Compute the loss
                loss = self.loss_fn(predictions, targets)

                data_targets.append(targets)
                data_preds.append(predictions)
                data_masks.append(target_masks)

                # print statistics
                running_loss += loss.item()

        # Put together predictions and labels from batches
        preds = torch.cat(data_preds, dim=0)
        tgts = torch.cat(data_targets, dim=0)
        mask = torch.cat(data_masks, dim=0)
        mask = mask > 0

        # Compute metrics
        roc_aucs = []
        prc_aucs = []
        sns = []
        sps = []
        num_tasks = len(self.target)
        for idx in range(num_tasks):
            actuals_task = torch.masked_select(tgts[:, idx], mask[:, idx].to(self.device))
            preds_task = torch.masked_select(preds[:, idx], mask[:, idx].to(self.device))

            # accuracy
            y_pred = np.where(preds_task.cpu().detach() >= 0.5, 1, 0)
            accuracy = accuracy_score(actuals_task.cpu().numpy(), y_pred)

            # sensitivity
            sn = sensitivity(actuals_task.cpu().numpy(), y_pred)

            # specificity
            sp = specificity(actuals_task.cpu().numpy(), y_pred)

            # roc-auc
            roc_auc = roc_auc_score(actuals_task.cpu().numpy(), preds_task.cpu().numpy())

            # prc-auc
            precision, recall, thresholds = precision_recall_curve(actuals_task.cpu().numpy(), preds_task.cpu().numpy())
            prc_auc = auc(recall, precision)

            # append
            sns.append(sn)
            sps.append(sp)
            roc_aucs.append(roc_auc)
            prc_aucs.append(prc_auc)
        average_sn = torch.mean(torch.tensor(sns))
        average_sp = torch.mean(torch.tensor(sps))
        average_roc_auc = torch.mean(torch.tensor(roc_aucs))
        average_prc_auc = torch.mean(torch.tensor(prc_aucs))

        # Rearange metrics
        metrics = {
            'acc': accuracy,
            'roc-auc': average_roc_auc.item(),
            'prc-auc': average_prc_auc.item(),
            'sensitivity': average_sn.item(),
            'specificity': average_sp.item(),
        }

        return preds, running_loss / len(self.train_loader), metrics