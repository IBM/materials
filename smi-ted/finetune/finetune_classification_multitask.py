# Deep learning
import torch
import torch.nn as nn
from torch import optim
from trainers import TrainerClassifierMultitask
from utils import get_optim_groups

# Data
import pandas as pd
import numpy as np

# Standard library
import args
import os


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define Target and Causal Features
    if config.dataset_name == 'tox21':
        targets = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                  'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    elif config.dataset_name == 'clintox':
        targets = ['FDA_APPROVED', 'CT_TOX']
    elif config.dataset_name == 'sider':
        targets = [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
            'Product issues', 'Eye disorders', 'Investigations',
            'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances',
            'Immune system disorders', 'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions',
            'Endocrine disorders', 'Surgical and medical procedures',
            'Vascular disorders', 'Blood and lymphatic system disorders',
            'Skin and subcutaneous tissue disorders',
            'Congenital, familial and genetic disorders', 'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
            'Renal and urinary disorders',
            'Pregnancy, puerperium and perinatal conditions',
            'Ear and labyrinth disorders', 'Cardiac disorders',
            'Nervous system disorders', 'Injury, poisoning and procedural complications'
        ]
    elif config.dataset_name == 'muv':
        targets = [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ]
    config.n_output = len(targets)

    # load dataset
    df_train = pd.read_csv(f"{config.data_root}/train.csv")
    df_valid = pd.read_csv(f"{config.data_root}/valid.csv")
    df_test  = pd.read_csv(f"{config.data_root}/test.csv")

    # load model
    if config.smi_ted_version == 'v1':
        from smi_ted_light.load import load_smi_ted
    elif config.smi_ted_version == 'v2':
        from smi_ted_large.load import load_smi_ted

    model = load_smi_ted(folder=config.model_path, ckpt_filename=config.ckpt_filename, n_output=len(targets), eval=False)
    model.net.apply(model._init_weights)
    print(model.net)

    lr = config.lr_start*config.lr_multiplier
    optim_groups = get_optim_groups(model, keep_decoder=bool(config.train_decoder))
    if config.loss_fn == 'bceloss':
        loss_function = nn.BCELoss()

    # init trainer
    trainer = TrainerClassifierMultitask(
        raw_data=(df_train, df_valid, df_test),
        dataset_name=config.dataset_name,
        target=targets,
        batch_size=config.n_batch,
        hparams=config,
        target_metric=config.target_metric,
        seed=config.start_seed,
        smi_ted_version=config.smi_ted_version,
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