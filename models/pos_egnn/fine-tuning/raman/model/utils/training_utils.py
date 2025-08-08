import os
import random
import socket
import traceback

import numpy as np
import torch
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.profilers import PyTorchProfiler

from .communications import SlackSender


def set_seed(seed):
    random.seed(seed)

    # Decorrelate the seeds, but still make the output reproducible
    max_seed = int(2**32 - 1)

    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def message_finished(model):
    constants_file_path = model.get_constants_file_path()
    message_finish = ""
    if constants_file_path is not None:
        message_finish = f"\n\n```To bypass constants computation, you can provide the argument --constants {constants_file_path} to the training script next time.```"
    message_finish += "\nRemember to clean your completed jobs to release resources."
    return message_finish


def message_error():
    traceback_str = traceback.format_exc()
    if len(traceback_str) > 2000:
        traceback_str = traceback_str[-2000:]
    message_error = f"\n\n-----ACTION REQUIRED:-----\n\nTraceback:\n```{traceback_str}```\n\n"
    return message_error


def send_external_communication(args, config, exp_name, status, model=None):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0 and args.slack and ("master" in exp_name):
        messageSender = SlackSender(slack_webhook=args.slack, config=config, exp_name=exp_name, args=args)
        if status == "error":
            message = message_error()
        elif status == "finished":
            message = message_finished(model)
        else:
            message = None
        message_blocks = messageSender.create_slack_message(
            status=status, message=message, training_name=exp_name, training_env=socket.gethostname()
        )
        messageSender.send_slack_message(message_blocks=message_blocks)


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
        logger = AimLogger(experiment=exp_name, run_name=exp_name, repo=str(config["general"]["aim_path"]))
        return logger
    else:
        # you must implement your favorite logger here
        logger = None

    if logger is None:
        raise Exception("If you are not going to use AIM for tracking, you must implement your own Logger on 'training_utils.py'.")
    else:
        return logger


def get_dataloader(config, omat_path):
    # Import inside function to avoid circular import
    from model.dataloaders import ModelDataLoader, OMatDataloader

    if "OMAT24" in config["dataset"]["datasets"].keys():
        assert len(config["dataset"]["datasets"]) == 1
        dataloader = OMatDataloader(omat_path, config)
    else:
        dataloader = ModelDataLoader(config)
    return dataloader
