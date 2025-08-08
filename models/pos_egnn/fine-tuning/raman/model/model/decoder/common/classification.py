from typing import Dict

from torch import Tensor, nn, sigmoid
from torch_geometric.data import Data
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall

from .._base_decoder import AbstractDecoder
from ..readouts import MultipleAggregationGlobalReadout


class BinaryClassificationDecoder(AbstractDecoder):
    def __init__(self, in_channels, num_residues, hidden_channels, activation, **kwargs):
        super().__init__(in_channels, num_residues, hidden_channels, activation, **kwargs)
        # For binary classification, we output a single logit per instance.
        self.readout = MultipleAggregationGlobalReadout(in_channels, num_residues, hidden_channels, 1, activation)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.auroc = AUROC(task="binary")

    def forward(self, data: Data) -> Dict[str, Tensor]:
        # Compute the raw logits for binary classification.
        pred = self.readout(data.embedding_0, data.batch)
        return {
            "target": pred,
        }

    def loss(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        # Assume target_data.target contains binary labels (0 or 1).
        # BCEWithLogitsLoss expects both input and target to have the same shape.
        target = target_data.target.float()
        pred = pred_data.target.view(-1)
        loss = {
            "Loss": self.loss_fn(pred, target),
        }
        return loss

    def metric(self, pred_data: Data, target_data: Data) -> Dict[str, Tensor]:
        target = target_data.target.float()
        pred = pred_data.target.view(-1)
        pred_prob = sigmoid(pred)
        # Threshold probabilities at 0.5 to get binary predictions.
        pred_labels = (pred_prob > 0.5).float()

        if len(pred) > 1:
            acc = self.accuracy(pred_labels, target)
            prec = self.precision(pred_labels, target)
            rec = self.recall(pred_labels, target)
            f1 = self.f1(pred_labels, target)
            auroc = self.auroc(pred_prob, target)
        else:
            acc = prec = rec = f1 = auroc = 0

        return {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUROC": auroc,
        }

    @property
    def target_keys(self):
        return ["target"]

    @property
    def loss_keys(self):
        return ["Loss"]

    @property
    def metric_keys(self):
        return ["Accuracy", "Precision", "Recall", "F1", "AUROC"]

    def store_constants(self, data):
        return {
            "dummy": 0,
        }

    @property
    def constants_keys(self):
        return ["target"]
