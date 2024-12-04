# Deep learning
import torch
from smi_ssed.load import Smi_ssed
from trainer import TrainerDirectDecoder

# Parallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

# Data
from utils import MoleculeModule
from torch.utils.data import DataLoader

# Standard library
import os
import args


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_train_objs(config):
    train_loader = MoleculeModule(
        config.max_len, 
        config.train_load, 
        config.data_root
    )
    train_loader.setup()
    train_dataset = train_loader.pubchem
    loader = DataLoader(
        train_dataset,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=train_loader.text_encoder.process, 
        sampler=DistributedSampler(train_dataset),
        num_workers=config.n_workers
    )

    # load model
    model = Smi_ssed(config, train_loader.get_vocab())
    model.apply(model._init_weights)

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_decoder, betas=(0.9, 0.99))

    return loader, model, optimizer


def main(
        config, 
        save_every: int, 
        total_epochs: int, 
        save_checkpoint_path: str,
        load_checkpoint_path: str
    ):
    ddp_setup()
    train_data, model, optimizers = load_train_objs(config)
    trainer = TrainerDirectDecoder(
        model, 
        train_data, 
        optimizers, 
        save_every, 
        save_checkpoint_path,
        load_checkpoint_path, 
        config
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == '__main__':
    parser = args.get_parser()
    args = parser.parse_args()
    main(
        args, 
        args.checkpoint_every, 
        args.max_epochs, 
        save_checkpoint_path=args.save_checkpoint_path,
        load_checkpoint_path=args.load_checkpoint_path, 
    )