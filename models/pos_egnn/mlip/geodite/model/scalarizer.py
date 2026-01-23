import torch.nn as nn
from torch import Tensor


class SumScalarizer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dynamic = False

    def forward(self, losses: Tensor) -> Tensor:
        loss = losses.sum()
        return loss


class MeanScalarizer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dynamic = False

    def forward(self, losses: Tensor) -> Tensor:
        loss = losses.mean()
        return loss


class ScaleInvariantScalarizer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dynamic = False

    def forward(self, losses: Tensor) -> Tensor:
        return (losses[losses > 0]).log1p().sum()


SCALARIZER_CLASS_MAP = {
    "Sum": SumScalarizer,
    "Mean": MeanScalarizer,
    "ScaleInvariant": ScaleInvariantScalarizer,
}
