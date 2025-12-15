from __future__ import absolute_import, division, print_function, annotations

import re
import math
import torch
from pathlib import Path
from torch import nn
from enum import Enum


def get_single_device(cpu=True):
    torch_version = torch.__version__[:5]

    if cpu:
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif torch_version >= '2.6.0':  # other GPU support
        if torch.xpu.is_available():
            return torch.device('xpu')
        elif torch.mps.is_available():
            return torch.device('mps')
    return None


class LoadCheckpointMode(Enum):
    SKIP = 1
    LAST = 2
    FILENAME = 3


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_order(file):
    file_pattern = re.compile(r'.*?(\d+).*?')
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


def get_tqdm_eta(pbar):
    return pbar.format_dict["elapsed"]


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
