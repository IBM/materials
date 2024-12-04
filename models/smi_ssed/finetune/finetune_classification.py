# Deep learning
import torch
import torch.nn as nn
from torch import optim
from smi_ssed.load import load_smi_ssed
from trainers import TrainerClassifier
from utils import get_optim_groups

# Data
import pandas as pd
import numpy as np

# Standard library
import args
import os


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    df_train = pd.read_csv(f"{config.data_root}/train.csv")
    df_valid = pd.read_csv(f"{config.data_root}/valid.csv")
    df_test  = pd.read_csv(f"{config.data_root}/test.csv")

    # load model
    model = load_smi_ssed(folder=config.model_path, ckpt_filename=config.ckpt_filename, n_output=config.n_output)
    model.net.apply(model._init_weights)
    print(model.net)

    lr = config.lr_start*config.lr_multiplier
    optim_groups = get_optim_groups(model, keep_decoder=bool(config.train_decoder))
    if config.loss_fn == 'crossentropy':
        loss_function = nn.CrossEntropyLoss()

    # init trainer
    trainer = TrainerClassifier(
        raw_data=(df_train, df_valid, df_test),
        dataset_name=config.dataset_name,
        target=config.measure_name,
        batch_size=config.n_batch,
        hparams=config,
        target_metric=config.target_metric,
        seed=config.start_seed,
        checkpoints_folder=config.checkpoints_folder,
        restart_filename=config.restart_filename,
        device=device,
        save_every_epoch=bool(config.save_every_epoch),
        save_ckpt=bool(config.save_ckpt)
    )
    trainer.compile(
        model=model,
        optimizer=optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99)),
        loss_fn=loss_function
    )
    trainer.fit(max_epochs=config.max_epochs)
    trainer.evaluate()


if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    main(config)