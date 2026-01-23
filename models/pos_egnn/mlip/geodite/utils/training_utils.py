import os
import random
import socket

import torch
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.profilers import PyTorchProfiler


def get_exp_name(args):
    if args.experiment:
        exp_name = str(args.experiment)
    else:
        exp_name = socket.gethostname()
        exp_name = f"{random.randint(10**8, 10**9 - 1)}_{exp_name}"
    return exp_name


def get_checkpoint_folder(config, exp_name):
    ckpt_path = f"{config['general']['ckpt_path']}/{exp_name}"
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt = os.path.join(ckpt_path, "last.ckpt")
    ckpt_exists = os.path.exists(ckpt)
    return ckpt_path, ckpt, ckpt_exists


def set_profiler(config, exp_name):
    if config["general"]["profiler"]:
        profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"profiler/{exp_name}"),
            schedule=torch.profiler.schedule(skip_first=5, repeat=0, wait=5, warmup=5, active=5),
        )
    else:
        profiler = None
    return profiler


def set_logger(config, exp_name):
    if config["general"].get("aim_path") is not None:
        return AimLogger(experiment=exp_name, run_name=exp_name, repo=str(config["general"]["aim_path"]))


def get_dataloader(config):
    from geodite.dataloaders import GeoditeDataLoader

    dataloader = GeoditeDataLoader(config)
    return dataloader
