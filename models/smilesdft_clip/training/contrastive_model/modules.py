import torch
from torch import nn

from external_models import load_smi_ted, load_VQGAN
from external_models import Smi_ted, MolTranBertTokenizer, VQGAN


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, config, pretrained=True, trainable=False):
        super().__init__()
        
        if pretrained:
            self.model = load_VQGAN(folder=config.external_models_ckpt_path, ckpt_filename=config.image_ckpt_filename)
        else:
            self.model = VQGAN()

        # Check and set requires_grad for each parameter
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model.feature_extraction(x)
    
    def get_hparams(self):
        return self.model.config


class TextEncoder(nn.Module):
    def __init__(self, config, pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = load_smi_ted(folder=config.external_models_ckpt_path, ckpt_filename=config.text_ckpt_filename)
        else:
            import json
            with open('./smi_ted_config.json') as f:
                config = json.load(f)
            tokenizer = MolTranBertTokenizer('./external_models/smi_ted_light/bert_vocab_curated.txt')
            self.model = Smi_ted(tokenizer, config)
            
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids):
        return self.model.extract_embeddings(input_ids)[2]
    
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

