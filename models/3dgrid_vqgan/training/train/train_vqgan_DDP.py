"Adapted from https://github.com/SongweiGe/TATS"

from __future__ import absolute_import, division, print_function, annotations
import os
import socket

from typing import Optional, Union, Callable

import sys
sys.path.insert(1, "../diffusion-chem-fm")

import hydra
from omegaconf import DictConfig, open_dict
from train.get_dataset import get_dataset
from vq_gan_3d.model import VQGAN
from trainer import Trainer

# Torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group


# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
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


def load_train_objs(cfg):
    # datasets
    train_dataset, val_dataset, _ = get_dataset(cfg)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.model.batch_size,
        num_workers=cfg.model.num_workers, 
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=SIZE,
            rank=RANK,
        ),
        shuffle=False,
        pin_memory=True
    )
    val_dataloader = None

    # model
    model = VQGAN(cfg)

    # optimizers
    opt_ae = torch.optim.Adam(list(model.encoder.parameters()) +
                                list(model.decoder.parameters()) +
                                list(model.pre_vq_conv.parameters()) +
                                list(model.post_vq_conv.parameters()) +
                                list(model.codebook.parameters()) +
                                list(model.perceptual_model.parameters()),
                                lr=cfg.model.lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(list(model.image_discriminator.parameters()),
                                lr=cfg.model.lr, betas=(0.5, 0.9))
                            
    return [train_dataloader, val_dataloader], model, [opt_ae, opt_disc]


def run(cfg, save_every: int, total_epochs: int, save_checkpoint_path: str, load_checkpoint_path: str):
    ddp_setup(RANK, SIZE, backend='nccl')

    print('### DDP Info ###')
    print('WORLD SIZE:', SIZE)
    print('LOCAL_RANK:', LOCAL_RANK)
    print('RANK:', RANK)
    print('')

    datasets, model, optimizers = load_train_objs(cfg)

    trainer = Trainer(
        model, 
        datasets, 
        optimizers, 
        save_every, 
        save_checkpoint_path, 
        load_checkpoint_path,
        cfg
    )
    trainer.train(total_epochs)

    destroy_process_group()


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def main(cfg: DictConfig):
    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(cfg.model.lr, accumulate, ngpu/8, bs/4, base_lr))

    if not os.path.isdir(cfg.model.save_checkpoint_path):
        os.mkdir(cfg.model.save_checkpoint_path)

    run(
        cfg, 
        save_every=cfg.model.checkpoint_every, 
        total_epochs=cfg.model.max_epochs,
        save_checkpoint_path=cfg.model.save_checkpoint_path,
        load_checkpoint_path=cfg.model.resume_from_checkpoint
    )


if __name__ == '__main__':
    main()
