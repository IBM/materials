import torch
import torch.nn as nn
from typing import Optional

def _init_lora(linear: nn.Linear, freeze_base: bool):
    if freeze_base:
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

class LoRALinear(nn.Module):
    """
    LoRA for linear layers:
      y = base(x) + scaling * B(A(dropout(x)))
    """
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: Optional[float],
                 dropout: float, merge_on_save: bool, freeze_base: bool):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        _init_lora(self.base, freeze_base)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(rank)
        self.lora_alpha = float(alpha) if alpha is not None else float(self.r)
        self.scaling = self.lora_alpha / max(self.r, 1)
        self.enable_lora = True
        self.merged = False

        self._post_act = getattr(base_linear, "activation", None)
        self._has_post_act = self._post_act is not None and not isinstance(self._post_act, nn.Identity)
        self.merge_on_save = bool(merge_on_save and not self._has_post_act)

        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        if self.merge_on_save:
            self._register_state_dict_hook(self._merge_on_state_dict)
        self._register_load_state_dict_pre_hook(self._strict_fill_on_load, with_module=True)

    def forward(self, x):
        y = self.base(x)
        if self._has_post_act:
            y = self._post_act(y)
        if self.enable_lora and self.r > 0:
            z = self.lora_dropout(x)
            z = self.lora_A(z)
            z = self.lora_B(z)
            y = y + self.scaling * z
        return y

    @torch.no_grad()
    def merged_weight(self):
        if self._has_post_act:
            return self.base.weight
        return self.base.weight + self.scaling * (self.lora_B.weight @ self.lora_A.weight)

    def _merge_on_state_dict(self, module, state_dict, prefix, local_metadata):
        # replace the tensor stored at base.weight with merged values
        key_w = prefix + "base.weight"
        if key_w in state_dict:
            state_dict[key_w] = self.merged_weight()
        # drop adapter tensors from the saved dict
        state_dict.pop(prefix + "lora_A.weight", None)
        state_dict.pop(prefix + "lora_B.weight", None)
        return state_dict

    @torch.no_grad()
    def _strict_fill_on_load(self, module, state_dict, prefix, local_metadata, strict, missing, unexpected, errors):
        for k, ref in [(prefix + "lora_A.weight", self.lora_A.weight),
                       (prefix + "lora_B.weight", self.lora_B.weight)]:
            if k not in state_dict:
                state_dict[k] = torch.zeros_like(ref)