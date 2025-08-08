import io
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from ...utils.constants import ACT_CLASS_MAPPING
from ...utils.yaml_utils import update_yaml


class AbstractDecoder(nn.Module, ABC):
    def __init__(self, in_channels, num_residues, hidden_channels, activation, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_residues = num_residues
        self.hidden_channels = hidden_channels
        self.activation = ACT_CLASS_MAPPING[activation]
        self.normalize_after_forward = kwargs.get("normalize_targets", False)

        self.constants = {}
        self.context = {}

    @property
    def id(self):
        return self.__class__.__name__

    @property
    def task(self):
        from . import DECODER_TO_TASK

        return DECODER_TO_TASK[self.id]

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def metric(self):
        pass

    @property
    @abstractmethod
    def target_keys(self):
        pass

    @property
    @abstractmethod
    def loss_keys(self):
        pass

    def normalize(self, data):
        return data

    def unnormalize(self, data):
        return data

    def _mock_data(self, data, head_out):
        return Data(z=data.z, ptr=data.ptr, batch=data.batch, **head_out)

    # TODO: GET FIDELITY FROM BATCH ITSELF, MAKING THE ARCHITECTURE MORE PARALELLIZABLE
    def forward_loss(self, target_data: Batch, fidelity: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self.set_context_state(fidelity)

        head_out = self.forward(target_data)
        pred_data = self._mock_data(target_data, head_out)

        metrics = {}

        with torch.no_grad():
            metrics.update(self.metric(pred_data, target_data))

        if self.normalize_after_forward:
            pred_data = self.normalize(pred_data)
            target_data = self.normalize(target_data)

        loss = self.loss(pred_data, target_data)

        # Add detached loss values to logging metrics
        metrics.update({name: value.detach().clone() for name, value in loss.items()})

        for results_dict in [metrics, loss]:
            items = list(results_dict.items())
            for key, value in items:
                new_key = f"{fidelity}/{self.id}/{key}/"
                results_dict[new_key] = value
                del results_dict[key]

        return loss, metrics

    def set_context_state(self, fidelity: str):
        # Set the correct normalization constants to be used with a given fidelity
        if self.constants:
            for constant_name, value in self.constants[fidelity].items():
                setattr(self, constant_name, value)

    def register_datasets(self, dataset_batch=None, precomputed_constants=None, file_to_store_constants=None):
        context = set()
        constants = {}

        # Select active data source and determine processing method
        constants_source = precomputed_constants or dataset_batch
        is_precomputed = precomputed_constants is not None

        for fidelity, item in constants_source.items():
            # Process constants based on data source type
            if is_precomputed:
                # Just convert lists to tensors
                processed_constants = {key: torch.tensor(value) if isinstance(value, list) else value for key, value in item.items()}
            else:
                processed_constants = self.store_constants(item)

                update_yaml(file_to_store_constants, {self.task: {fidelity: processed_constants}})

            # Update constants and context
            constants.setdefault(fidelity, {})
            constants[fidelity].update(processed_constants)
            context.add(fidelity)

            log_msg = f"{self.id}/{fidelity}{f' (N={item.num_graphs})' if not is_precomputed else ''}: {processed_constants}"
            print(log_msg)

        self.constants = constants
        self.context = context

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
