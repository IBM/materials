import torch
import torch.nn as nn
from collections import OrderedDict
from .config import LoRAConfig
from .inject import apply_lora

def _is_lora_module(m: nn.Module) -> bool:
    return hasattr(m, "base") and hasattr(m, "lora_A") and hasattr(m, "lora_B")

class PosEGNNLoRAModel(nn.Module):
    """
    Wrap a PosEGNN backbone, inject LoRA only into mergeable linear layers
    (post-activation linears are skipped by the injector), and expose a merged export.
    """
    def __init__(self, backbone: nn.Module, lora_config: LoRAConfig = LoRAConfig()):
        super().__init__()
        self.backbone = backbone
        self.lora_config = lora_config

        # Injector must skip any module with a non-identity .activation
        self.n_scalar, _ = apply_lora(self.backbone, self.lora_config)

        if getattr(lora_config, "freeze_base", False):
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.lora_parameters():
                p.requires_grad_(True)

        self._adapters_enabled = True

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def lora_parameters(self):
        for _, m in self.backbone.named_modules():
            if _is_lora_module(m):
                yield from m.lora_A.parameters()
                yield from m.lora_B.parameters()

    @torch.no_grad()
    def enable_adapter(self):
        self._adapters_enabled = True
        for _, m in self.backbone.named_modules():
            if _is_lora_module(m):
                setattr(m, "enable_lora", True)

    @torch.no_grad()
    def disable_adapter(self):
        self._adapters_enabled = False
        for _, m in self.backbone.named_modules():
            if _is_lora_module(m):
                setattr(m, "enable_lora", False)

    @torch.no_grad()
    def _export_merged_state_dict(self) -> OrderedDict:
        """
        Build a plain PosEGNN state dict (original layout):
          - merge adapter weights into '...weight'
          - copy '...bias'
          - copy other base.* params with 'base.' stripped
          - drop all '...lora_*'
        """
        sd_out = OrderedDict()
        sd_in = self.backbone.state_dict()
        wrapper_paths = {name for name, m in self.backbone.named_modules() if _is_lora_module(m)}

        def wrapper_path_for(key: str):
            for p in wrapper_paths:
                if key.startswith(p + "."):
                    return p
            return None

        for key, val in sd_in.items():
            # drop adapter tensors
            if ".lora_A." in key or ".lora_B." in key:
                continue

            wp = wrapper_path_for(key)
            if wp is not None:
                if key.endswith(".base.weight"):
                    path = key[:-len(".base.weight")]
                    mod = self.backbone.get_submodule(path)  # LoRA wrapper
                    sd_out[path + ".weight"] = mod.merged_weight()
                    continue
                if key.endswith(".base.bias"):
                    sd_out[key.replace(".base.bias", ".bias")] = val
                    continue
                prefix = f"{wp}.base."
                if key.startswith(prefix):
                    sd_out[key.replace(prefix, f"{wp}.", 1)] = val
                    continue
                continue  # ignore other wrapper internals

            # passthrough for non-wrapper keys
            sd_out[key] = val

        return sd_out

    @torch.no_grad()
    def state_dict_backbone(self, merged: bool = False) -> OrderedDict:
        """
        Return backbone-only weights. If merged=True, adapters are fused and
        keys match the original PosEGNN layout.
        """
        if merged:
            return self._export_merged_state_dict()
        return self.backbone.state_dict()

    @torch.no_grad()
    def save_pretrained_merged(self, path: str):
        torch.save(self.state_dict_backbone(merged=True), path)

    @torch.no_grad()
    def save_pretrained_adapted(self, path: str):
        torch.save(self.state_dict(), path)

    def lora_report(self) -> str:
        return f"LoRA injected - scalar layers: {self.n_scalar}"