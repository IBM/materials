import os
from .models import CLIPModel, SigLIPModel


def load_clip(folder="../data/checkpoints/pretrained", ckpt_filename="CLIP_8_20250127-024339.pt", device='cpu'):
    model = CLIPModel(device)
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    return model


def load_siglip(folder="../data/checkpoints/pretrained", ckpt_filename="SigLIP_8_20250127-024339.pt", device='cpu'):
    model = SigLIPModel(device)
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    return model