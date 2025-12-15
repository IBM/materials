# Deep learning
import torch
import torch.nn as nn
from torch import optim
from trainers import TrainerRegressor
from contrastive_model.load import load_clip, load_siglip
from utils import init_weights, RMSELoss

# Data
import pandas as pd
import numpy as np

# Standard library
import args


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    df_train = pd.read_csv(f"{config.data_root}/train.csv")
    df_valid = pd.read_csv(f"{config.data_root}/valid.csv")
    df_test  = pd.read_csv(f"{config.data_root}/test.csv")

    # load model
    if config.arch == 'clip':
        model = load_clip(folder='../data/checkpoints/clip/pretrained', ckpt_filename=config.ckpt_filename, device=device)
    elif config.arch == 'siglip':
        model = load_siglip(folder='../data/checkpoints/siglip/pretrained', ckpt_filename=config.ckpt_filename, device=device)
    else:
        raise Exception('No architecture found. Options: `clip` or `siglip`.')
    model.net.apply(init_weights)
    print(model.net)

    # model.text_encoder.eval()
    # model.image_encoder.eval()

    if config.loss_fn == 'rmse':
        loss_function = RMSELoss()
    elif config.loss_fn == 'mae':
        loss_function = nn.L1Loss()

    # init trainer
    trainer = TrainerRegressor(
        raw_data=(df_train, df_valid, df_test),
        npy_data_dir=config.grid_path,
        dataset_name=config.dataset_name,
        target=config.measure_name,
        batch_size=config.n_batch,
        hparams=config,
        target_metric=config.target_metric,
        seed=config.start_seed,
        num_workers=config.num_workers,
        checkpoints_folder=f'../data/checkpoints/{config.arch}/finetuned/qm9/{config.measure_name}',
        restart_filename=config.restart_filename,
        image_filename=config.image_filename,
        text_filename=config.text_filename,
        device=device,
        save_every_epoch=bool(config.save_every_epoch),
        save_ckpt=bool(config.save_ckpt)
    )
    trainer.compile(
        model=model,
        optimizer=optim.AdamW(
            list(model.image_projection.parameters())
            +list(model.text_projection.parameters())
            +list(model.net.parameters())
            +list(model.image_encoder.model.parameters())
            +list(model.text_encoder.model.parameters()), 
            lr=config.lr_start, betas=(0.9, 0.999)
        ),
        loss_fn=loss_function
    )
    trainer.fit(max_epochs=config.max_epochs)
    trainer.evaluate()


if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    main(config)