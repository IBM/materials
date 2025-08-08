from typing import Dict
from torch import Tensor, nn
from torch_geometric.data import Data
from torchmetrics import R2Score, PearsonCorrCoef
import torch
from .._base_decoder import AbstractDecoder
from ..readouts import GlobalReadout


def non_zero_indices(array_with_zeros):
    return [i for i, num in enumerate(array_with_zeros) if num != 0]


class RamanDecoder(AbstractDecoder):
    def __init__(self, in_channels, num_residues, hidden_channels, activation,  **kwargs):
        super().__init__(in_channels, num_residues, hidden_channels, activation,  **kwargs)
        num_labels = 4000
        self.readout = GlobalReadout(in_channels, num_residues, hidden_channels, num_labels, activation, cueq_config=kwargs.get("cueq_config"))

        # Loss
        self.mse = torch.nn.MSELoss()

        # Metrics
        self.mae = nn.L1Loss()
        self.r2 = R2Score()
        self.relu = nn.ReLU()
        self.pearson = PearsonCorrCoef()

    def forward(self, data: Data) -> Dict[str, Tensor]:
        spectrum = torch.sigmoid(self.readout(data.embedding_0, data.batch))

        return {"spectrum": spectrum}

    def loss(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        target = target_data.spectrum
        pred = torch.reshape(pred_data.spectrum,(-1,))

        indices = torch.nonzero(target) # Versao gold
        filtered_target = target[indices] # Versao gold
        filtered_pred = pred[indices] # Versao gold
        loss1 = self.mae(filtered_pred,filtered_target) # Versao gold
        mask = torch.ones(len(target)).bool() # Versao gold
        mask[indices] = 0 # Versao gold
        zeros_target = target[mask] # Versao gold
        zeros_pred = pred[mask] # Versao gold
        loss2 = self.mae(zeros_pred, zeros_target) # Versao gold
        if loss1 > loss2: # Versao gold
            loss_value = loss1 # Versao gold
        else: # Versao gold
            loss_value = loss2 # Versao gold

        loss = {"MSE": loss_value}

        return loss

    def metric(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        target = target_data.spectrum
        pred = pred_data.spectrum
        pred2 = torch.reshape(pred,(-1,))
        metrics = {
            "MAE": self.mae(pred2, target),
            "R2": self.r2(pred2, target),
            "RMSE": torch.sqrt(self.mse(pred2, target)),
            "PEARSON": self.pearson(pred2, target),
        }
        return metrics

    # TODO: Is there a way to get the keys automatically so we don't need to rewrite them here?
    @property
    def target_keys(self):
        return ["spectrum"]

    @property
    def loss_keys(self):
        return ["MSE"]

    @property
    def metric_keys(self):
        return ["MAE", "R2", "RMSE", "PEARSON"]

    def store_constants(self, data):
        return {"dummy": 0} # Just to don't break when there is no normalization

    @property
    def constants_keys(self):
        return ["spectrum"]