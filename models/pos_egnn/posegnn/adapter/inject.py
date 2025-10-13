import re
import torch
import torch.nn as nn
from .layers import LoRALinear
from .config import LoRAConfig

def apply_lora(model: nn.Module, cfg: LoRAConfig) -> tuple[int, int]:
    """
    Replace leaf linear-like layers under include patterns with LoRA.
    Safely wraps linears with internal norm/activation since LoRA is pre-activation.
    Returns (num_scalar_wrapped, 0).
    """
    include_patterns = list(cfg.include_names or [])
    exclude_patterns = list(cfg.exclude_names or [])
    if getattr(cfg, "preset", None) == "posegnn" and not include_patterns:
        include_patterns = [r"^encoder\.", r"^readout\."]

    inc_re = [re.compile(p) for p in include_patterns]
    exc_re = [re.compile(p) for p in exclude_patterns]

    def wants(name: str) -> bool:
        if any(p.search(name) for p in exc_re):
            return False
        if inc_re and not any(p.search(name) for p in inc_re):
            return False
        return True

    def is_linear_like(m: nn.Module) -> bool:
        w = getattr(m, "weight", None)
        if isinstance(m, nn.Embedding):
            return False
        return isinstance(w, torch.Tensor) and w.ndim == 2

    n_scalar = 0

    for full_name, module in list(model.named_modules()):
        if not is_linear_like(module):
            continue
        if not wants(full_name):
            continue

        parent_name, _, child = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        # already wrapped guard
        if hasattr(module, "base") and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            continue

        wrapped = LoRALinear(
            module, cfg.rank, cfg.alpha, cfg.dropout, cfg.merge_on_save, cfg.freeze_base
        )
        setattr(parent, child, wrapped)
        n_scalar += 1

    return n_scalar, 0