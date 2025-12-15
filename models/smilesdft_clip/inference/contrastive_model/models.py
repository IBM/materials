import torch
import torch.nn.functional as F
from torch import nn

import random
import numpy as np

from .modules import ImageEncoder, TextEncoder, ProjectionHead
from .utils import cross_entropy, dotdict


class CLIPModel(nn.Module):
    def __init__(self, device='cpu', config=None, temperature=1.0):
        super().__init__()
        if config:
            self.image_encoder = ImageEncoder(config, config.pretrained, config.trainable)
            self.text_encoder = TextEncoder(config, config.pretrained, config.trainable)
            self.image_projection = ProjectionHead(config.image_embedding, config.projection_dim, config.dropout)
            self.text_projection = ProjectionHead(config.text_embedding, config.projection_dim, config.dropout)
        self.temperature = temperature
        self.device = device
        self.to(device)

    def forward(self, batch):
        text_features = self.text_encoder(batch["caption"])
        image_features = self.image_encoder(batch["image"].to(self.device))

        text_embeddings = self.text_projection(text_features.float())
        image_embeddings = self.image_projection(image_features.float())

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / (2 * self.temperature), dim=-1)

        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')

        loss = (images_loss + texts_loss) / 2.0

        return loss.mean()

    def encode_grid(self, grid):
        if grid.dim() == 4:
            grid = grid.unsqueeze(0)
        image_features = self.image_encoder(grid.to(self.device))
        image_embeddings = self.image_projection(image_features.float())
        return image_embeddings

    def encode_text(self, smiles):
        text_features = self.text_encoder(smiles)
        text_embeddings = self.text_projection(text_features.float())
        return text_embeddings

    def feature_extraction(self, image, caption):
        text_features = self.text_encoder(caption)
        image_features = self.image_encoder(image.to(self.device))

        text_embeddings = self.text_projection(text_features.float())
        image_embeddings = self.image_projection(image_features.float())

        embeddings = torch.cat([text_embeddings, image_embeddings], dim=-1)
        return embeddings

    def load_checkpoint(self, ckpt_path):
        ckpt_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        self.config = dotdict(ckpt_dict['hparams'])
        self.config_model = dotdict(self.config.model)
        self.config_model.pretrained = False

        # instatiate modules
        self.image_encoder = ImageEncoder(self.config, self.config_model.pretrained, self.config_model.trainable)
        self.text_encoder = TextEncoder(self.config, self.config_model.pretrained, self.config_model.trainable)
        self.image_projection = ProjectionHead(self.config_model.image_embedding, self.config_model.projection_dim, self.config_model.dropout)
        self.text_projection = ProjectionHead(self.config_model.text_embedding, self.config_model.projection_dim, self.config_model.dropout)

        # restore weights
        self.load_state_dict(ckpt_dict['MODEL_STATE'], strict=True)

        # put all modules to device
        self.to(self.device)

        # load RNG states each time the model and states are loaded from checkpoint
        if 'rng' in ckpt_dict:
            rng = ckpt_dict['rng']
            for key, value in rng.items():
                if key =='torch_state':
                    torch.set_rng_state(value.cpu())
                elif key =='cuda_state':
                    # torch.cuda.set_rng_state(value.cpu())
                    pass
                elif key =='numpy_state':
                    np.random.set_state(value)
                elif key =='python_state':
                    random.setstate(value)
                else:
                    print('unrecognized state')

    def __str__(self):
        return 'CLIP'


class SigLIPModel(nn.Module):
    def __init__(self, device='cpu', config=None, temperature=1.0):
        super().__init__()
        if config:
            self.image_encoder = ImageEncoder(config, config.pretrained, config.trainable)
            self.text_encoder = TextEncoder(config, config.pretrained, config.trainable)
            self.image_projection = ProjectionHead(config.image_embedding, config.projection_dim, config.dropout)
            self.text_projection = ProjectionHead(config.text_embedding, config.projection_dim, config.dropout)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.device = device
        self.to(device)

    def forward(self, batch):
        text_features = self.text_encoder(batch["caption"])
        image_features = self.image_encoder(batch["image"].to(self.device))

        text_embeddings = self.text_projection(text_features.float())
        image_embeddings = self.image_projection(image_features.float())

        zimg = F.normalize(image_embeddings, p=2, dim=1)
        ztxt = F.normalize(text_embeddings, p=2, dim=1)

        logits = torch.matmul(zimg, ztxt.T) * torch.exp(self.temperature) + self.bias

        batch_size = logits.size(0)
        labels = 2 * torch.eye(batch_size, device=logits.device) - torch.ones(batch_size, batch_size, device=logits.device)

        loss = -torch.sum(F.logsigmoid(labels * logits)) / batch_size

        return loss

    def encode_grid(self, grid):
        if grid.dim() == 4:
            grid = grid.unsqueeze(0)
        image_features = self.image_encoder(grid.to(self.device))
        image_embeddings = self.image_projection(image_features.float())
        return image_embeddings

    def encode_text(self, smiles):
        text_features = self.text_encoder(smiles)
        text_embeddings = self.text_projection(text_features.float())
        return text_embeddings

    def feature_extraction(self, image, caption):
        text_features = self.text_encoder(caption)
        image_features = self.image_encoder(image.to(self.device))

        text_embeddings = self.text_projection(text_features.float())
        image_embeddings = self.image_projection(image_features.float())

        embeddings = torch.cat([text_embeddings, image_embeddings], dim=-1)
        return embeddings

    def load_checkpoint(self, ckpt_path):
        ckpt_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        self.config = dotdict(ckpt_dict['hparams'])
        self.config_model = dotdict(self.config.model)
        self.config_model.pretrained = False

        # instatiate modules
        self.image_encoder = ImageEncoder(self.config, self.config_model.pretrained, self.config_model.trainable)
        self.text_encoder = TextEncoder(self.config, self.config_model.pretrained, self.config_model.trainable)
        self.image_projection = ProjectionHead(self.config_model.image_embedding, self.config_model.projection_dim, self.config_model.dropout)
        self.text_projection = ProjectionHead(self.config_model.text_embedding, self.config_model.projection_dim, self.config_model.dropout)

        # restore weights
        self.load_state_dict(ckpt_dict['MODEL_STATE'], strict=True)

        # put all modules to device
        self.to(self.device)

        # load RNG states each time the model and states are loaded from checkpoint
        if 'rng' in ckpt_dict:
            rng = ckpt_dict['rng']
            for key, value in rng.items():
                if key =='torch_state':
                    torch.set_rng_state(value.cpu())
                elif key =='cuda_state':
                    # torch.cuda.set_rng_state(value.cpu())
                    pass
                elif key =='numpy_state':
                    np.random.set_state(value)
                elif key =='python_state':
                    random.setstate(value)
                else:
                    print('unrecognized state')

    def __str__(self):
        return 'SigLIP'