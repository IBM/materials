# Deep learning
import torch
import torch.nn as nn
from torch import optim
from trainers import TrainerRegressor
from vq_gan_3d.model.vqgan_DDP import load_VQGAN
from utils import init_weights, RMSELoss

# Parallel
from torch.distributed import init_process_group, destroy_process_group

# Data
import pandas as pd
import numpy as np

# Standard library
import math
import args
import os


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddp_setup()

    # load dataset
    df_train = pd.read_csv(f"{config.data_root}/train.csv")
    df_valid = pd.read_csv(f"{config.data_root}/valid.csv")
    df_test  = pd.read_csv(f"{config.data_root}/test.csv")

    # load model
    model = load_VQGAN(folder=config.model_path, filename=config.ckpt_filename)
    model.net.apply(init_weights)
    print(model.net)

    # disable gradients to frozen parts
    for param in model.decoder.parameters():  # decoder
        param.requires_grad = False
    for param in model.post_vq_conv.parameters():  # after codebook
        param.requires_grad = False
    for param in model.codebook.parameters():  # codebook
        param.requires_grad = False
    for param in model.image_discriminator.parameters():  # GAN discriminator
        param.requires_grad = False

    if config.loss_fn == 'rmse':
        loss_function = RMSELoss()
    elif config.loss_fn == 'mae':
        loss_function = nn.L1Loss()

    # init trainer
    trainer = TrainerRegressor(
        raw_data=(df_train, df_valid, df_test),
        grids_path=config.grid_path,
        dataset_name=config.dataset_name,
        target=config.measure_name,
        batch_size=config.n_batch,
        hparams=config,
        internal_resolution=model.config['model']['internal_resolution'],
        target_metric=config.target_metric,
        seed=config.start_seed,
        num_workers=config.num_workers,
        checkpoints_folder=config.checkpoints_folder,
        restart_filename=config.restart_filename,
        device=device,
        save_every_epoch=bool(config.save_every_epoch),
        save_ckpt=bool(config.save_ckpt)
    )
    trainer.compile(
        model=model,
        optimizer=optim.AdamW(
            list(model.encoder.parameters())
            +list(model.pre_vq_conv.parameters())
            +list(model.net.parameters()), 
            lr=config.lr_start, betas=(0.9, 0.999)
        ),
        loss_fn=loss_function
    )
    trainer.fit(max_epochs=config.max_epochs)
    trainer.evaluate()
    destroy_process_group()


if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    main(config)