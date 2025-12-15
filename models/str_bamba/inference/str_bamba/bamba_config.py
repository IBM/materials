from dataclasses import dataclass, field


@dataclass
class BambaConfig:

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    max_position_embeddings: int = 262144
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


@dataclass
class BambaEncoderDecoderConfig:

    encoder_config: BambaConfig = None
    decoder_config: BambaConfig = None
    tie_word_embeddings: bool = True
    seed: int = 0