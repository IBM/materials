import torch
from torch import nn
import torch.nn.functional as F

from .modules import ImageEncoder, TextEncoder, ProjectionHead
from .utils import cross_entropy


class CLIPModel(nn.Module):
    def __init__(self, config, temperature=1.0):
        super().__init__()
        self.image_encoder = ImageEncoder(config, config.pretrained, config.trainable)
        self.text_encoder = TextEncoder(config, config.pretrained, config.trainable)
        self.image_projection = ProjectionHead(config.image_embedding, config.projection_dim, config.dropout)
        self.text_projection = ProjectionHead(config.text_embedding, config.projection_dim, config.dropout)
        self.temperature = temperature


    def forward(self, batch, local_rank):
        text_features = self.text_encoder(batch["caption"])
        image_features = self.image_encoder(batch["image"].to(local_rank))

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

    def __str__(self):
        return 'CLIP'


class SigLIPModel(nn.Module):
    def __init__(self, config, temperature=1.0):
        super().__init__()
        self.image_encoder = ImageEncoder(config, config.pretrained, config.trainable)
        self.text_encoder = TextEncoder(config, config.pretrained, config.trainable)
        self.image_projection = ProjectionHead(config.image_embedding, config.projection_dim, config.dropout)
        self.text_projection = ProjectionHead(config.text_embedding, config.projection_dim, config.dropout)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch, local_rank):
        text_features = self.text_encoder(batch["caption"])
        image_features = self.image_encoder(batch["image"].to(local_rank))

        text_embeddings = self.text_projection(text_features.float())
        image_embeddings = self.image_projection(image_features.float())

        zimg = F.normalize(image_embeddings, p=2, dim=1)
        ztxt = F.normalize(text_embeddings, p=2, dim=1)

        logits = torch.matmul(zimg, ztxt.T) * torch.exp(self.temperature) + self.bias

        batch_size = logits.size(0)
        labels = 2 * torch.eye(batch_size, device=logits.device) - torch.ones(batch_size, batch_size, device=logits.device)

        loss = -torch.sum(F.logsigmoid(labels * logits)) / batch_size

        return loss

    def __str__(self):
        return 'SigLIP'