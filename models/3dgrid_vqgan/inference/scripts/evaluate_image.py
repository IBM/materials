import sys
sys.path.append('./')

import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from vq_gan_3d.model.vqgan_DDP import load_VQGAN, VQGAN
from vq_gan_3d.metrics import ImageMetrics
from dataset.default import DEFAULTDataset
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_epoch_from_filename(filename):
    try:
        epoch = int(filename.split('_')[-1].split('.')[0])
    except:
        epoch = None
    return epoch


def main():
    # settings
    model_folder = './checkpoints'
    model_filename = 'VQGAN_43.pt'
    data_folder = '../data/sample_data_schema'
    save_results_folder = './results'
    epoch = infer_epoch_from_filename(model_filename)
    print(model_filename)
    if not os.path.exists(save_results_folder):
        os.mkdir(save_results_folder)
    if epoch is None:
        raise Exception("Cannot find the epoch number from filename. Please, provide an epoch manually.") 

    # load model
    vqgan = load_VQGAN(folder=model_folder, filename=model_filename).to(DEVICE)
    vqgan.eval()

    # load data
    eval_dataset = DEFAULTDataset(
        root_dir=data_folder, 
        internal_resolution=vqgan.config['model']['internal_resolution']
    )
    dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=True
    )

    # decoder quantative evaluation
    print('Starting evaluation...')
    dfs = []
    for idx, data in enumerate(tqdm(dataloader)):
        x = data['data'].to(DEVICE)
        with torch.no_grad():
            print(f'{idx}/{len(dataloader)}')
            # forward
            encds = vqgan.encode(x)
            x_recon = vqgan.decode(encds)

            torch.cuda.empty_cache()
            
            # image metrics
            eval_metrics = ImageMetrics(x.squeeze(), x_recon.squeeze(), device=DEVICE)
            metrics_dict = eval_metrics.get_metrics()
            metrics_dict = {k: [v] for k, v in metrics_dict.items()}
            dfs.append(pd.DataFrame(metrics_dict))

            torch.cuda.empty_cache()
            
    # save results
    df = pd.concat(dfs, axis=0)
    df.to_csv(os.path.join(save_results_folder, f'./vqgan_image_evaluation_epoch={epoch}.csv'), index=False)
    print(df.mean())


if __name__ == '__main__':
    main()