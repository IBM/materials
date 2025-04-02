import torch
import selfies as sf
import numpy as np
import pandas as pd
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
import gc
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SELFIESDataset(Dataset):
    def __init__(self, selfies_list):
        self.selfies = selfies_list

    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, idx):
        return self.selfies[idx]

class SELFIES(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.invalid = []

    def smiles_to_selfies(self, smiles):
        try:
            return sf.encoder(smiles.strip()).replace('][', '] [')
        except:
            try:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles.strip()))
                return sf.encoder(smiles).replace('][', '] [')
            except:
                return None

    def get_selfies(self, smiles_list):
        with Pool(cpu_count()) as pool:
            selfies = list(pool.map(self.smiles_to_selfies, smiles_list))

        self.invalid = [i for i, s in enumerate(selfies) if s is None]
        selfies = [s if s is not None else '[nop]' for s in selfies]
        return selfies

    @torch.no_grad()
    def get_embedding_batch(self, selfies_batch):
        encodings = self.tokenizer(
            selfies_batch,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        encodings = {k: v.to(self.model.device) for k, v in encodings.items()}

        outputs = self.model.encoder(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask']
        )

        model_output = outputs.last_hidden_state
        input_mask_expanded = encodings['attention_mask'].unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        return pooled_output.cpu().numpy()

    def load(self, checkpoint=None):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
        self.model = AutoModel.from_pretrained("ibm/materials.selfies-ted")
        self.model.eval()

    def encode(self, smiles_list=[], use_gpu=False, return_tensor=False, batch_size=128, num_workers=4):
        selfies = self.get_selfies(smiles_list)
        dataset = SELFIESDataset(selfies)

        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        embeddings = []
        for batch in tqdm(loader, desc="Encoding"):
            emb = self.get_embedding_batch(batch)
            embeddings.append(emb)
            del emb
            gc.collect()

        emb = np.vstack(embeddings)

        for idx in self.invalid:
            emb[idx] = np.nan
            print(f"Cannot encode {smiles_list[idx]} to selfies. Embedding replaced by NaN.")

        return torch.tensor(emb) if return_tensor else pd.DataFrame(emb)
