import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def _init_lora(linear: nn.Linear, freeze_base: bool):
    if freeze_base:
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

class LoRALinear(nn.Module):
    """
    LoRA for linear layers applied pre-activation:
      y = act( norm( (W x + b) + scaling * B(A(dropout(x))) ) )
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

        # Optional submodules carried by custom Dense
        self._norm = getattr(base_linear, "norm", None)
        if not isinstance(self._norm, nn.Module):
            self._norm = None

        self._post_act = getattr(base_linear, "activation", None)
        self._has_post_act = self._post_act is not None and not isinstance(self._post_act, nn.Identity)

        # Always allow merge on save now that we inject pre-activation
        self.merge_on_save = bool(merge_on_save)

        # LoRA adapters
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)  # down
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False) # up

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        if self.merge_on_save:
            self._register_state_dict_hook(self._merge_on_state_dict)
        self._register_load_state_dict_pre_hook(self._strict_fill_on_load, with_module=True)

    def _apply_activation(self, y):
        if not self._has_post_act:
            return y
        act = self._post_act
        # support nn.Module or callable (e.g. torch.nn.functional.silu)
        if isinstance(act, nn.Module):
            return act(y)
        if callable(act):
            return act(y)
        return y

    def forward(self, x):
        # linear pre-activation
        y = F.linear(x, self.base.weight, self.base.bias)

        # add LoRA delta pre-activation
        if self.enable_lora and self.r > 0:
            z = self.lora_dropout(x)
            z = self.lora_A(z)
            z = self.lora_B(z)
            y = y + self.scaling * z

        # optional norm then activation
        if self._norm is not None:
            y = self._norm(y)
        y = self._apply_activation(y)
        return y

    @torch.no_grad()
    def merged_weight(self):
        # Always valid since injected pre-activation
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