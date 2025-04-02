import gc
from tqdm import tqdm

import os
import sys
import torch
import selfies as sf  # selfies>=2.1.1
import pickle
import pandas as pd
import numpy as np
from datasets import Dataset
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel


class SELFIES(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.invalid = []

    def get_selfies(self, smiles_list):
        self.invalid = []
        spaced_selfies_batch = []
        for i, smiles in enumerate(smiles_list):
            try:
                selfies = sf.encoder(smiles.strip())
            except:
                try:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles.strip()))
                    selfies = sf.encoder(smiles)
                except:
                    selfies = "[]"
                    self.invalid.append(i)

            spaced_selfies_batch.append(selfies.replace('][', '] ['))
        return spaced_selfies_batch

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
        outputs = self.model.encoder(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
        model_output = outputs.last_hidden_state

        input_mask_expanded = encodings['attention_mask'].unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        return pooled_output.cpu().numpy()

    def load(self, checkpoint="bart-2908.pickle"):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
        self.model = AutoModel.from_pretrained("ibm/materials.selfies-ted")
        self.model.eval()

    def encode(self, smiles_list=[], use_gpu=False, return_tensor=False):
        selfies = self.get_selfies(smiles_list)

        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        batch_size = 128
        embeddings = []

        for i in tqdm(range(0, len(selfies), batch_size), desc="Encoding batches"):
            batch = selfies[i:i + batch_size]
            emb = self.get_embedding_batch(batch)
            embeddings.append(emb)
            del emb
            gc.collect()

        emb = np.vstack(embeddings)

        for idx in self.invalid:
            emb[idx] = np.nan
            print(f"Cannot encode {smiles_list[idx]} to selfies. Embedding replaced by NaN.")

        return torch.tensor(emb) if return_tensor else pd.DataFrame(emb)
