import os
from .models import CLIPModel, SigLIPModel


def load_clip(folder="../data/checkpoints/clip/pretrained", ckpt_filename=""):
    model = CLIPModel()
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    model.eval()
    return model


def load_siglip(folder="../data/checkpoints/siglip/pretrained", ckpt_filename=""):
    model = SigLIPModel()
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    model.eval()
    return model