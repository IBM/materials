# inject.py
import re
import torch
import torch.nn as nn
from .layers import LoRALinear
from .config import LoRAConfig

def apply_lora(model: nn.Module, cfg: LoRAConfig) -> tuple[int, int]:
    """
    Replace leaf linear-like layers under include patterns with LoRA.
    Skips any module that has a non-identity .activation to guarantee mergeability.
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

    def has_post_act(m: nn.Module) -> bool:
        act = getattr(m, "activation", None)
        return (act is not None) and (not isinstance(act, nn.Identity))

    n_scalar = 0
    skipped = []  # <— track skipped post-activation linears

    for full_name, module in list(model.named_modules()):
        if not is_linear_like(module):
            continue
        if not wants(full_name):
            continue
        if has_post_act(module):
            skipped.append(full_name)  # <— record and skip
            continue

        parent_name, _, child = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        if hasattr(module, "base") and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            continue

        wrapped = LoRALinear(
            module, cfg.rank, cfg.alpha, cfg.dropout, cfg.merge_on_save, cfg.freeze_base
        )
        setattr(parent, child, wrapped)
        n_scalar += 1

    if getattr(cfg, "log_skipped", False) and skipped:
        print("[lora] skipped post-activation linears:")
        for n in skipped:
            print("  -", n)

    return n_scalar, 0