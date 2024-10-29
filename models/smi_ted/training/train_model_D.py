# This code uses the decoder loss directly.
#
#

# Deep learning
import torch
from torch_optimizer.lamb import Lamb
from trainer import TrainerDirectDecoder

# Parallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

# Data
from utils import MoleculeModule, get_optim_groups
from torch.utils.data import DataLoader

# Standard library
import os
import args


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_train_objs(config):
    # load data
    train_loader = MoleculeModule(
        config.max_len, 
        config.train_load, 
        config.data_root
    )
    train_loader.setup()
    
    loader = DataLoader(
        train_loader.pubchem,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=train_loader.text_encoder.process, 
        sampler=DistributedSampler(train_loader.pubchem),
        num_workers=config.n_workers
    )

    # load model
    if config.smi_ted_version == 'v1':
        from smi_ted_light.load import Smi_ted
    elif config.smi_ted_version == 'v2':
        from smi_ted_large.load import Smi_ted
    
    model = Smi_ted(config, train_loader.get_vocab()).to('cuda')
    model.apply(model._init_weights)

    # load optimizer
    optim_groups = get_optim_groups(model)
    optimizer = torch.optim.AdamW(optim_groups, lr=config.lr_decoder, betas=(0.9, 0.99), fused=True)
    
    return loader, model, optimizer


def main(
        config, 
        save_every: int, 
        total_epochs: int, 
        save_checkpoint_path: str,
        load_checkpoint_path: str
    ):
    ddp_setup()

    # training objects
    train_data, model, optimizer = load_train_objs(config)

    # init trainer
    trainer = TrainerDirectDecoder(
        model, 
        train_data, 
        optimizer, 
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
