import sys
import argparse
sys.path.append('./')

import torch
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm

from contrastive_model.dataset import CLIPDataset
from contrastive_model.load import load_clip, load_siglip
from contrastive_model.utils import get_single_device

DEVICE = get_single_device(cpu=False)


def get_embeddings(dataloader, model):
    with torch.no_grad():
        embeddings_list = [model.feature_extraction(batch['image'], batch['caption']) for batch in tqdm(dataloader)]
        embeddings_list = torch.cat(embeddings_list, dim=0)
    return embeddings_list


def main(args):
    # load model
    if args.arch == 'clip':
        model = load_clip(folder='../data/checkpoints/clip/pretrained', ckpt_filename=args.ckpt_filename, device=DEVICE).eval()
    elif args.arch == 'siglip':
        model = load_siglip(folder='../data/checkpoints/siglip/pretrained', ckpt_filename=args.ckpt_filename, device=DEVICE).eval()
    else:
        raise Exception('No architecture found. Options: `clip` or `siglip`.')
    
    # load dataset
    df = pd.read_csv(args.dataset_path)
    npy_files = df['File Name'].to_list()
    smi_files = df['Canonical'].to_list()

    # load data
    dataloader = DataLoader(
        CLIPDataset(args.data_dir, npy_files, smi_files),
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=False,
        pin_memory=False,
    )

    # extract CLIP embeddings
    embeddings = get_embeddings(dataloader, model).cpu().numpy()

    # concat embeddings with dataset
    df_embeddings = pd.DataFrame(embeddings)
    df_full = pd.concat([df, df_embeddings], axis=1)

    # save to disk
    df_full.to_csv(args.save_dataset_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_dataset_path', type=str, required=True)
    parser.add_argument('--ckpt_filename', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False)
    parser.add_argument('--num_workers', type=int, default=0, required=False)
    args = parser.parse_args()
    main(args)