import sys
import argparse
sys.path.append('./')

import torch
import torch.nn.functional as F
from tqdm import tqdm

import pandas as pd

from contrastive_model.dataset import CLIPDataset, build_loaders
from contrastive_model.load import load_clip, load_siglip
from contrastive_model.utils import get_single_device

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
PandasTools.RenderImagesInAllDataFrames(True)

DEVICE = get_single_device(cpu=False)


# function to canonicalize SMILES
def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


def get_image_embeddings(config, valid_df, model):
    data_loader = build_loaders(config, valid_df, mode="valid")
    
    with torch.no_grad():
        print('Extracting Grid embeddings...')
        embeddings_list = [model.encode_grid(batch["image"]) for batch in tqdm(data_loader)]
        embeddings_list = torch.cat(embeddings_list, dim=0)
    return embeddings_list


def find_matches(model, image_embeddings, query, smi_files, npy_files, topk=6):
    with torch.no_grad():
        text_embeddings = model.encode_text(query)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), topk)  # Top k matches

    selected_SMILES = [smi_files[i] for i in indices]
    selected_GRIDS = [npy_files[i] for i in indices]
    similarity_values = [values[i].item() for i in range(len(values))]  # Extract similarity values
    
    return selected_SMILES, selected_GRIDS, similarity_values


def main(args):
    # load model
    if args.arch == 'clip':
        model = load_clip(folder='../data/checkpoints/clip/pretrained', ckpt_filename=args.ckpt_filename, device=DEVICE).eval()
    elif args.arch == 'siglip':
        model = load_siglip(folder='../data/checkpoints/siglip/pretrained', ckpt_filename=args.ckpt_filename, device=DEVICE).eval()
    else:
        raise Exception('No architecture found. Options: `clip` or `siglip`.')

    # load dataset
    catalog = pd.read_csv(args.dataset_path)
    npy_files = catalog['3d_grid'].to_list()
    smi_files = catalog['canon_smiles'].apply(normalize_smiles).to_list()
    dataset = CLIPDataset(args.data_dir, npy_files, smi_files)

    # extract grid embeddings
    image_embeddings = get_image_embeddings(model.config, dataset, model)

    # retrieval evaluation
    df_results = []
    print('Starting retrieval...')
    for query in tqdm(smi_files):
        top_match_smiles, top_match_grids, top_match_scores = find_matches(model, 
                                                                           image_embeddings, 
                                                                           query, 
                                                                           smi_files, 
                                                                           npy_files, 
                                                                           args.topk)

        first_match_smiles = top_match_smiles[0]
        first_match_grid = top_match_grids[0]
        first_match_score = top_match_scores[0]

        data_results = {
            'query_smiles': [query],
            'first_match_smiles': [first_match_smiles],
            'first_match_grid': [first_match_grid],
            'first_match_score': [first_match_score],
            'top_match_smiles': [top_match_smiles],
            'top_match_grids': [top_match_grids],
            'top_match_scores': [top_match_scores],
            'is_first_query_match': [query==first_match_smiles],
            'is_query_in_topk=2': [query in top_match_smiles[:2]],
            'is_query_in_topk=3': [query in top_match_smiles[:3]],
        }

        if args.topk > 3:
            for k in range(4, args.topk+1):
                data_results[f'is_query_in_topk={k}'] = [query in top_match_smiles[:k]]

        df_results.append(pd.DataFrame(data_results))

    # save results to disk
    df_results = pd.concat(df_results, axis=0)
    df_results.to_csv(args.save_dataset_path, index=False)

    # statistics
    total_queries = len(df_results)
    first_match_acc = df_results['is_first_query_match'].sum() / total_queries
    topk2_match_acc = df_results['is_query_in_topk=2'].sum() / total_queries
    topk3_match_acc = df_results['is_query_in_topk=3'].sum() / total_queries
        
    print('Topk=1 match accuracy:', first_match_acc)
    print('Topk=2 match accuracy:', topk2_match_acc)
    print('Topk=3 match accuracy:', topk3_match_acc)
    
    if args.topk > 3:
        for k in range(4, args.topk+1):
            topk_match_acc = df_results[f'is_query_in_topk={k}'].sum() / total_queries
            print(f'Topk={k} match accuracy:', topk_match_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_dataset_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='', required=True)
    parser.add_argument('--ckpt_filename', type=str, default='', required=True)
    parser.add_argument('--topk', type=int, default=50, required=False)
    args = parser.parse_args()
    main(args)