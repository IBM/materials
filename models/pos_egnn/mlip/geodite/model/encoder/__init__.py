from .geodite.encoder import Geodite

ENCODER_CLASS_MAP = {
    "Geodite": Geodite,
}

__all__ = ["ENCODER_CLASS_MAP"]
