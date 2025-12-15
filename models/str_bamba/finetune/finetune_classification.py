# Deep learning
import torch
import torch.nn as nn
from torch import optim
from trainers import TrainerClassifier
from utils import get_optim_groups, _init_weights
from str_bamba.load import load_strbamba

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
    model = load_strbamba(
        config.ckpt_filename, 
        config.model_path, 
        config.model_config_filename, 
        config.tokenizer_filename,
        config.n_output, 
        config.dropout
    )
    model.net.apply(_init_weights)
    print(model.net)

    if config.loss_fn == 'crossentropy':
        loss_function = nn.CrossEntropyLoss()

    # init trainer
    trainer = TrainerClassifier(
        raw_data=(df_train, df_valid, df_test),
        dataset_name=config.dataset_name,
        inputs=config.inputs,
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
    params = [
        {'params': model.encoder.parameters(), 'lr': config.lr_encoder, 'weight_decay': 0.0},
        {'params': model.net.parameters(), 'lr': config.lr_predictor, 'weight_decay': 0.001},
    ]
    trainer.compile(
        model=model,
        optimizer=optim.AdamW(params, betas=(0.9, 0.99)),
        loss_fn=loss_function
    )
    trainer.fit(max_epochs=config.max_epochs)
    trainer.evaluate()


if __name__ == '__main__':
    parser = args.get_parser()
    config = parser.parse_args()
    main(config)