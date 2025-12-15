"""
Some parts of the code are inspired from: https://github.com/deepspeedai/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py
"""


# Standard library
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import os
import socket
import args
import random
import json
import glob
import math

# Deep learning
import torch
import deepspeed
from torch_optimizer.lamb import Lamb
from torch.utils.data import DataLoader
from deepspeed.accelerator import get_accelerator
from trainer import TrainerEncoder
from str_bamba.bamba_config import BambaConfig, BambaEncoderDecoderConfig
from str_bamba.bamba import BambaEncoderDecoder

# Parallel
from torch.utils.data.distributed import DistributedSampler

# Data
from datasets import load_dataset, concatenate_datasets
from data_encoder import TextEncoder4ModelEncoder
from str_datasets import (
    MolecularFormulaDataset, 
    SMILESDataset,
    IUPACDataset,
    InChIDataset,
    SELFIESDataset,
    PolymerSPGDataset,
    FormulationDataset
)


def get_all_files_from_directory(path):
    return [os.path.basename(file) for file in glob.glob(os.path.join(path, '*'))]


def load_train_objs(config):
    ### load data ###
    # pubchem_files = {'train': [f'normprops_{i}.csv' for i in range(20)]}
    pubchem_files = {'train': get_all_files_from_directory(config.pubchem_files_path)}
    polymer_spg_files = {'train': get_all_files_from_directory(config.polymer_files_path)}
    formulation_files = {'train': get_all_files_from_directory(config.formulation_files_path)}

    # might be loading data, let rank 0 load first
    # if torch.distributed.get_rank() != 0:
    #     torch.distributed.barrier()

    # load files
    pubchem_dataset = load_dataset(config.pubchem_files_path, data_files=pubchem_files, cache_dir=config.data_cache_dir, split='train', trust_remote_code=True)
    polymer_spg_dataset = load_dataset(config.polymer_files_path, data_files=polymer_spg_files, cache_dir=config.data_cache_dir, split='train', trust_remote_code=True)
    formulation_dataset = load_dataset(config.formulation_files_path, data_files=formulation_files, cache_dir=config.data_cache_dir, split='train', trust_remote_code=True)
    str_bag = concatenate_datasets([pubchem_dataset, polymer_spg_dataset, formulation_dataset]).shuffle()
    print(str_bag)

    # datasets
    mf_dataset = MolecularFormulaDataset(pubchem_dataset)
    smiles_dataset = SMILESDataset(pubchem_dataset)
    iupac_dataset = IUPACDataset(pubchem_dataset)
    inchi_dataset = InChIDataset(pubchem_dataset)
    selfies_dataset = SELFIESDataset(pubchem_dataset)
    polymer_spg_dataset = PolymerSPGDataset(polymer_spg_dataset)
    formulation_dataset = FormulationDataset(formulation_dataset)

    # encoding
    text_encoder = TextEncoder4ModelEncoder(config.tokenizer_path, config.max_len, bag=str_bag)
    vocab_size = text_encoder.vocab_size

    # data is loaded, indicate other ranks can proceed
    # if torch.distributed.get_rank() == 0:
    #     torch.distributed.barrier()

    world_size = deepspeed.comm.get_world_size()
    rank = deepspeed.comm.get_rank()

    # dataloaders
    mf_loader = DataLoader(
        mf_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_encoder.process, 
        sampler=DistributedSampler(
            mf_dataset,
            num_replicas=world_size,
            rank=rank,
        ),
        num_workers=config.n_workers
    )
    smiles_loader = DataLoader(
        smiles_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_encoder.process, 
        sampler=DistributedSampler(
            smiles_dataset,
            num_replicas=world_size,
            rank=rank,
        ),
        num_workers=config.n_workers
    )
    iupac_loader = DataLoader(
        iupac_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_encoder.process, 
        sampler=DistributedSampler(
            iupac_dataset,
            num_replicas=world_size,
            rank=rank,
        ),
        num_workers=config.n_workers
    )
    inchi_loader = DataLoader(
        inchi_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_encoder.process, 
        sampler=DistributedSampler(
            inchi_dataset,
            num_replicas=world_size,
            rank=rank,
        ),
        num_workers=config.n_workers
    )
    selfies_loader = DataLoader(
        selfies_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_encoder.process, 
        sampler=DistributedSampler(
            selfies_dataset,
            num_replicas=world_size,
            rank=rank,
        ),
        num_workers=config.n_workers
    )
    polymer_spg_loader = DataLoader(
        polymer_spg_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_encoder.process, 
        sampler=DistributedSampler(
            polymer_spg_dataset,
            num_replicas=world_size,
            rank=rank,
        ),
        num_workers=config.n_workers
    )
    formulation_loader = DataLoader(
        formulation_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_encoder.process, 
        sampler=DistributedSampler(
            formulation_dataset,
            num_replicas=world_size,
            rank=rank,
        ),
        num_workers=config.n_workers
    )
    data_loaders = [mf_loader, smiles_loader, iupac_loader, inchi_loader, selfies_loader, polymer_spg_loader, formulation_loader]

    ### load model ###
    with open(config.config_path) as json_data:
        config_json = json.load(json_data)
    bamba_config = BambaEncoderDecoderConfig(
        encoder_config=BambaConfig(**config_json['encoder_config']),
        decoder_config=BambaConfig(**config_json['decoder_config']),
        tie_word_embeddings=config_json['tie_word_embeddings'],
        seed=config_json['seed']
    )
    model = BambaEncoderDecoder(bamba_config)
    print(sum(p.numel() for p in model.parameters()))

    # disable decoder gradients
    for n, p in model.named_parameters():
        if 'decoder' in n:
            p.requires_grad = False

    ### load optimizer ###
    optimizer = Lamb(model.encoder.parameters(), lr=config.lr_start, betas=(0.9, 0.99), weight_decay=config.weight_decay)

    ### deepspeed ###
    model, optimizer, _, _ = deepspeed.initialize(
        args=config, 
        model=model, 
        model_parameters=model.encoder.parameters(),
        optimizer=optimizer,
    )
    
    return data_loaders, model, optimizer, vocab_size


def main(
        config, 
        save_every: int, 
        total_epochs: int, 
        save_checkpoint_path: str,
        load_checkpoint_path: str
    ):
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(local_rank)

    ngpus_per_node = torch.cuda.device_count()
    device = torch.device("cuda", local_rank)
    print('Number of GPUs:', ngpus_per_node)
    print('device:', device)

    # training objects
    train_data, model, optimizer, vocab_size = load_train_objs(config)

    # init trainer
    trainer = TrainerEncoder(
        model, 
        vocab_size,
        train_data, 
        optimizer, 
        save_every, 
        save_checkpoint_path,
        load_checkpoint_path, 
        config
    )
    trainer.train(total_epochs)


if __name__ == '__main__':
    parser = args.get_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    main(
        args, 
        args.checkpoint_every, 
        args.max_epochs, 
        save_checkpoint_path=args.save_checkpoint_path,
        load_checkpoint_path=args.load_checkpoint_path, 
    )