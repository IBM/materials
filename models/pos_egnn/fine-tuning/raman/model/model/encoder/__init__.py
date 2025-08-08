from .gotennet.encoder import GotenNet

ENCODER_CLASS_MAP = {
    "GotenNet": GotenNet,
}

__all__ = ["ENCODER_CLASS_MAP"]
