import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        print('Original embeddings:\n', out)
        return out


class Expert(nn.Module):
    def __init__(self, model, output_size, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.model = model
        self.output_size = output_size

    def forward(self, x):
        # Check if input is empty and return an empty tensor of the appropriate shape
        if len(x) == 0:
            return torch.empty(size=(0, self.output_size))

        # Generate embeddings using the model's encode method
        out = self.model.encode(x)

        # Check if out is a Pandas DataFrame or list and convert to torch tensor if needed
        if isinstance(out, pd.DataFrame):
            out = torch.tensor(out.values, dtype=torch.float32)
        elif isinstance(out, list):
            out = torch.stack(out, dim=0)
        
        # Pad the embeddings to match the desired output size
        out = F.pad(out, pad=(0, self.output_size - out.shape[1], 0, 0), value=0)

        # Optionally print the embeddings if verbose mode is enabled
        if self.verbose:
            print(f'Original embeddings:\n', out)

        return out


class Net(nn.Module):
    def __init__(self, smiles_embed_dim, output_dim=2, dropout=0.2):
        super().__init__()
        self.desc_skip_connection = True 
        self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(smiles_embed_dim, output_dim)

    def forward(self, smiles_emb):
        x_out = self.fc1(smiles_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + smiles_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)

        return z