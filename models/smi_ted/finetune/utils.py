# Deep learning
import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix

# Data
import pandas as pd
import numpy as np

# Standard library
import os

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
PandasTools.RenderImagesInAllDataFrames(True)


def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


class RMSELoss:
    def __init__(self):
        pass

    def __call__(self, yhat, y):
        return torch.sqrt(torch.mean((yhat-y)**2))


def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp/(tp+fn))


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tn/(tn+fp)) 


def get_optim_groups(module, keep_decoder=False):
    # setup optimizer
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if not keep_decoder and 'decoder' in fpn: # exclude decoder components
                continue

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    return optim_groups


class CustomDataset(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, idx):
        smiles = self.dataset['canon_smiles'].iloc[idx]
        labels = self.dataset[self.target].iloc[idx]
        return smiles, labels


class CustomDatasetMultitask(Dataset):
    def __init__(self, dataset, targets):
        self.dataset = dataset
        self.targets = targets

    def __len__(self):
        return len(self.dataset)    
    
    def __getitem__(self, idx):
        smiles = self.dataset['canon_smiles'].iloc[idx]
        labels = self.dataset[self.targets].iloc[idx].to_numpy()
        mask = [0.0 if np.isnan(x) else 1.0 for x in labels]
        labels = [0.0 if np.isnan(x) else x for x in labels]
        return smiles, torch.tensor(labels, dtype=torch.float32), torch.tensor(mask)