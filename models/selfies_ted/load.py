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
                selfies = sf.encoder(smiles.rstrip())
            except:
                try:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles.rstrip()))
                    selfies = sf.encoder(smiles)
                except:
                    selfies = "[]"
                    self.invalid.append(i)

            spaced_selfies_batch.append(selfies.replace('][', '] ['))

        return spaced_selfies_batch


    def get_embedding(self, selfies):
        encoding = self.tokenizer(selfies["selfies"], return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        model_output = outputs.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        model_output = sum_embeddings / sum_mask

        del encoding['input_ids']
        del encoding['attention_mask']

        encoding["embedding"] = model_output

        return encoding


    def load(self, checkpoint="bart-2908.pickle"):
        """
            inputs :
                   checkpoint (pickle object)
        """

        self.tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
        self.model = AutoModel.from_pretrained("ibm/materials.selfies-ted")





    # TODO: remove `use_gpu` argument in validation pipeline
    def encode(self, smiles_list=[], use_gpu=False, return_tensor=False):
        """
            inputs :
                   checkpoint (pickle object)
            :return: embedding
        """
        selfies = self.get_selfies(smiles_list)
        selfies_df = pd.DataFrame(selfies,columns=["selfies"])
        data = Dataset.from_pandas(selfies_df)
        embedding = data.map(self.get_embedding, batched=True, num_proc=1, batch_size=32)
        emb = np.asarray(embedding["embedding"].copy())

        for idx in self.invalid:
            emb[idx] = np.nan
            print("Cannot encode {0} to selfies and embedding replaced by NaN".format(smiles_list[idx]))

        if return_tensor:
            return torch.tensor(emb)
        return pd.DataFrame(emb)
