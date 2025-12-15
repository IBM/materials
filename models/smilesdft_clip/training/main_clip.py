import os
import socket
import datetime

from typing import Optional, Union, Callable

import itertools
import pandas as pd

import torch
from torch.distributed import init_process_group, destroy_process_group

from contrastive_model.args import parse_args
from contrastive_model.dataset import CLIPDataset, build_loaders
from contrastive_model.models import CLIPModel
from contrastive_model.trainer import Trainer


# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    WITH_CUDA = torch.cuda.is_available()
    DEVICE = 'gpu' if WITH_CUDA else 'CPU'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'


def ddp_setup(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
       backend = 'nccl' if backend is None else str(backend)
    else:
       backend = 'gloo' if backend is None else str(backend)

    init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )
    torch.cuda.set_device(int(LOCAL_RANK))


def load_train_objs(config):
    # datasets
    catalog = pd.read_csv(config.catalog_path)
    npy_files = catalog['File Name']  # 3D electron density
    smi_files = catalog['Canonical']  # SMILES (canonicalized)

    dataset = CLIPDataset(config.data_path, npy_files, smi_files)
    train_dataloader, val_dataloader = build_loaders(config, SIZE, RANK, dataset, valid_split=config.valid_split, mode='train')

    train_size = len(train_dataloader)*config.batch_size*int(SIZE)
    valid_size = len(val_dataloader)*config.valid_batch_size*int(SIZE)
    print('Train data size:', train_size)
    print('Valid data size:', valid_size)
    print('Total data size: ~', train_size+valid_size)

    # model
    model = CLIPModel(config)

    # optimizer
    params = [
        {"params": model.image_encoder.parameters(), "lr": config.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": config.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": config.head_lr, "weight_decay": config.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.patience, factor=config.factor
    )
                            
    return [train_dataloader, val_dataloader], model, optimizer, lr_scheduler


def main(config):
    ddp_setup(RANK, SIZE, backend='nccl')

    total_gpus = int(SIZE)
    init_job_datetime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    print('Job init datetime:', init_job_datetime)
    print('Total GPUs:', total_gpus)

    datasets, model, optimizer, scheduler = load_train_objs(config)
    trainer = Trainer(
        init_job_datetime,
        model, 
        datasets,
        optimizer,
        scheduler,
        config.save_every_steps, 
        config.save_checkpoint_path, 
        config.load_checkpoint_path,
        config.load_checkpoint_mode,
        config.load_checkpoint_filename,
        config,
        total_gpus
    )
    trainer.train(config.max_epochs)

    destroy_process_group()


if __name__ == "__main__":
    config = parse_args()
    main(config)