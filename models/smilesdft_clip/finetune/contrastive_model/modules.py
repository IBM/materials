import torch
from torch import nn
import torch.nn.functional as F

import json

from external_models import load_smi_ted, load_VQGAN
from external_models import Smi_ted, MolTranBertTokenizer, VQGAN


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, config, pretrained=True, trainable=False):
        super().__init__()
        
        if pretrained:
            self.model = load_VQGAN(folder=config['model']['external_models_ckpt_path'], ckpt_filename=config['model']['image_ckpt_filename'])
        else:
            self.model = VQGAN()

        # Check and set requires_grad for each parameter
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        output = self.model.feature_extraction(x)
        return output
    
    def get_hparams(self):
        return self.model.config


class TextEncoder(nn.Module):
    def __init__(self, config, pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = load_smi_ted(folder=config['model']['external_models_ckpt_path'], ckpt_filename=config['model']['text_ckpt_filename'])
        else:
            tokenizer = MolTranBertTokenizer('./external_models/smi_ted_light/bert_vocab_curated.txt')
            self.model = Smi_ted(tokenizer, config['text_encoder'])
            
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids):
        output = self.model.extract_embeddings(input_ids)[2]
        return output
    
    def get_hparams(self):
        return self.model.config


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        if len(x.size()) == 4:  # Check for 4D image features
            num_features = x.size(1) * x.size(2) * x.size(3)  # Calculate number of features from image size
            x = x.view(x.size(0), num_features)
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class Net(nn.Module):
    
    def __init__(self, smiles_embed_dim, n_output=1, dropout=0.2):
        super().__init__()
        self.desc_skip_connection = True
        self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(smiles_embed_dim, n_output)

    def forward(self, smiles_emb, multitask=False):
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

        if multitask:
            return F.sigmoid(z)
        return z