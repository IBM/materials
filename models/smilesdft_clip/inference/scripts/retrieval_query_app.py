import sys
import argparse
sys.path.append('./')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from contrastive_model.dataset import CLIPDataset, load_smi_files, load_npy_files, build_loaders
from contrastive_model.load import load_clip, load_siglip


def get_image_embeddings(config, valid_df, model, device='cpu'):
    data_loader = build_loaders(config, valid_df, mode="valid")
    
    embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            image_features = model.image_encoder(batch["image"].unsqueeze(0).to(device))
            image_embeddings = model.image_projection(image_features.float())
            embeddings_list.append(image_embeddings)
    return torch.cat(embeddings_list)


def find_matches(model, image_embeddings, query, smi_files):
    with torch.no_grad():
        text_features = model.text_encoder(query)
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), 6)  # Top 6 matches
    selected_SMILES = [smi_files[i] for i in indices]
    similarity_values = [values[i].item() for i in range(len(values))]  # Extract similarity values
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Retrieved DFT Energy Grids based on QUERY: {query}", fontsize=16)

    for i, (match, score, ax) in enumerate(zip(selected_SMILES, similarity_values, axes.flatten())):
        # image = cv2.imread(f"images/{match}.png")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ax.imshow(image)
        ax.set_title(f"SMILES: {match}\nSimilarity: {score:.4f}", fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print('Similarity Scores:', values)
    print('Selected SMILES:', selected_SMILES)


def main(args):
    # load dataset
    smi_files = load_smi_files('../data/sample_data_schema')
    npy_files = load_npy_files('../data/sample_data_schema')
    dataset = CLIPDataset('../data/sample_data_schema', npy_files, smi_files)

    # load model
    if args.arch == 'clip':
        model = load_clip(folder='../data/checkpoints/clip/pretrained', ckpt_filename='SMILESDFT-CLIP_96.pt')
    elif args.args == 'siglip':
        model = load_siglip(folder='../data/checkpoints/clip/pretrained', ckpt_filename='SMILESDFT-SigLIP_96.pt')
    else:
        raise Exception('No architecture found. Options: `clip` or `siglip`.')
    model.eval()

    image_embeddings = get_image_embeddings(model.config, dataset, model)
    while True:
        query = input("Enter your query SMILES (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        find_matches(model, image_embeddings, query, smi_files)
        print('Query SMILES: ', query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    args = parser.parse_args()
    main(args)