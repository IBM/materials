import io
from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from ...utils.constants import ACT_CLASS_MAPPING

class AbstractDecoder(nn.Module):
    def __init__(self, in_channels, num_residues, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_residues = num_residues
        self.hidden_channels = kwargs.get("hidden_channels", None)
        activation = kwargs.get("activation", None)
        if activation:
            self.activation = ACT_CLASS_MAPPING[activation]
        self.normalize_after_forward = kwargs.get("normalize_targets", False)
        self._dataset_name = kwargs.get("dataset_name", None)

    @property
    def id(self) -> str:
        base = "SnapshotDecoder"
        if self._dataset_name is not None:
            return f"{base}_{self._dataset_name}"
        return base

    @property
    def name(self) -> str:
        return "SnapshotDecoder"

    @property
    def task(self):
        DECODER_TO_TASK = {
            "SnapshotDecoder": "Snapshot",
        }

        base = self.id.split("_")[0]
        return DECODER_TO_TASK[base]

    def forward(self):
        return

    def loss(self):
        return

    def metric(self):
        return

    @property
    def target_keys(self):
        return

    @property
    def loss_keys(self):
        return

    def normalize(self, data):
        return data

    def unnormalize(self, data):
        return data

    def _mock_data(self, data, head_out):
        return Data(z=data.z, ptr=data.ptr, batch=data.batch, **head_out)

    def set_context_state(self, fidelity: str):
        # Set the correct normalization constants to be used with a given fidelity
        if self.constants is not None:
            for constant_name, value in self.constants[fidelity].items():
                setattr(self, constant_name, value)

    def state_dict(self, **kwargs):
        state = super().state_dict(**kwargs)
        attributes = ["constants", "context"]

        for attr in attributes:
            buffer = io.BytesIO()
            torch.save(getattr(self, attr), buffer)
            buffer.seek(0)
            state[kwargs["prefix"] + attr] = torch.ByteTensor(list(buffer.getvalue()))

        return state

    def load_state_dict(self, state_dict, strict=True, device="cpu"):
        attributes = ["constants", "context"]

        for attr in attributes:
            if attr in state_dict:
                buffer = io.BytesIO(state_dict[attr].cpu().numpy())
                buffer.seek(0)
                setattr(self, attr, torch.load(buffer, weights_only=False, map_location=device))

        super().load_state_dict(state_dict, strict)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        def move_dict_to_device(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    data[key] = move_dict_to_device(value)
            elif torch.is_tensor(data):
                data = data.to(*args, **kwargs)
            return data

        self.constants = move_dict_to_device(self.constants)
        return self

    @staticmethod
    def _merge_dicts(dic: Dict[str, dict]):
        merged_d = {}
        for val in dic.values():
            merged_d |= val
        return merged_d
