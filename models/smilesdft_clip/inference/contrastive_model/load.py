import os
from .models import CLIPModel, SigLIPModel


def load_clip(folder="../data/checkpoints/pretrained", ckpt_filename="", device='cpu'):
    model = CLIPModel(device)
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    model.eval()
    return model


def load_siglip(folder="../data/checkpoints/pretrained", ckpt_filename="", device='cpu'):
    model = SigLIPModel(device)
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    model.eval()
    return model