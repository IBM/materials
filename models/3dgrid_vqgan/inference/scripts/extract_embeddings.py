import sys
sys.path.append('./')

import pandas as pd
import numpy as np

import argparse
import torch
from torch.utils.data import DataLoader
from vq_gan_3d import load_VQGAN, get_single_device, VQGANDataset
from tqdm import tqdm

DEVICE = get_single_device(cpu=False)


def extract_embeddings(vqgan, dataloader):
    with torch.no_grad():
        batch_embds = [vqgan.feature_extraction(x.to(DEVICE)) for x in tqdm(dataloader)]
        batch_embds = torch.cat(batch_embds, dim=0)
    return batch_embds


def main(args):
    # load model
    vqgan = load_VQGAN(
        folder='../data/checkpoints/pretrained', 
        ckpt_filename=args.ckpt_filename
    ).eval().to(DEVICE)

    # load data
    df = pd.read_csv(args.dataset_path)
    dataloader = DataLoader(
        VQGANDataset(
            args.data_dir, 
            df['3d_grid'].to_list(), 
            vqgan.config['model']['internal_resolution']
        ), 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=False,
        pin_memory=False,
    )

    # debug
    print('VQGAN model and data loaded!')
    print('\tCheckpoint:', args.ckpt_filename)
    print('\tDataset size:', df.shape)

    # extract vqgan embeddings
    embeddings = extract_embeddings(vqgan, dataloader).cpu().numpy()

    # concat embeddings with dataset
    df_embeddings = pd.DataFrame(embeddings)
    df_full = pd.concat([df, df_embeddings], axis=1)

    # save to disk
    df_full.to_csv(args.save_dataset_path, index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_dataset_path', type=str, required=True)
    parser.add_argument('--ckpt_filename', type=str, default='VQGAN_43.pt', required=False)
    parser.add_argument('--data_dir', type=str, default='/scratch/vyukio/npy_datasets/qm9_npy/', required=False)
    parser.add_argument('--batch_size', type=int, default=1, required=False)
    parser.add_argument('--num_workers', type=int, default=0, required=False)
    args = parser.parse_args()
    main(args)