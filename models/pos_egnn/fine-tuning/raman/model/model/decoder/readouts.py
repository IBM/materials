import torch
from torch import Tensor, nn
from torch_geometric.nn import MaxAggregation, MeanAggregation, MinAggregation, MultiAggregation, QuantileAggregation
from torch_scatter import scatter

from ...utils.constants import ACT_CLASS_MAPPING


class NodeInvariantReadout(nn.Module):
    def __init__(self, in_channels, num_residues, hidden_channels, out_channels, activation, **kwargs):
        super().__init__()
        act = ACT_CLASS_MAPPING[activation]

        # Define a Linear layer for each layer output except the last one
        self.linears = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(num_residues - 1)])

        # Define the nonlinear layer for the last layer's output
        self.non_linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            act(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, embedding_0: Tensor) -> Tensor:
        layer_outputs = embedding_0.squeeze(2)  # [n_nodes, in_channels, num_residues]

        processed_outputs = []
        for i, linear in enumerate(self.linears):
            processed_outputs.append(linear(layer_outputs[:, :, i]))

        processed_outputs.append(self.non_linear(layer_outputs[:, :, -1]))
        output = torch.stack(processed_outputs, dim=0).sum(dim=0).squeeze(-1)
        return output


class GlobalReadout(nn.Module):
    def __init__(self, in_channels, num_residues, hidden_channels, out_channels, activation, **kwargs):
        super().__init__()
        self.node_readout = NodeInvariantReadout(in_channels, num_residues, hidden_channels, out_channels, activation)

    def forward(self, embedding_0: Tensor, batch: Tensor, reduce="sum") -> Tensor:
        node_output = self.node_readout(embedding_0)
        output = scatter(src=node_output, index=batch, dim=0, reduce=reduce)
        return output


class MultipleAggregationGlobalReadout(nn.Module):
    def __init__(self, in_channels, num_residues, hidden_channels, out_channels, activation, **kwargs):
        super().__init__()
        self.node_readout = NodeInvariantReadout(in_channels, num_residues, hidden_channels, out_channels, activation)

        aggregations = [
            MeanAggregation(),
            MaxAggregation(),
            MinAggregation(),
            QuantileAggregation(q=0.1),
            QuantileAggregation(q=0.2),
            QuantileAggregation(q=0.3),
            QuantileAggregation(q=0.4),
            QuantileAggregation(q=0.5),
            QuantileAggregation(q=0.6),
            QuantileAggregation(q=0.7),
            QuantileAggregation(q=0.8),
            QuantileAggregation(q=0.9),
        ]

        self.global_aggregation = MultiAggregation(aggrs=aggregations, mode="cat")

        self.mlp = nn.Sequential(
            nn.Linear(out_channels * len(aggregations), hidden_channels),
            ACT_CLASS_MAPPING[activation](),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, embedding_0: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        node_output = self.node_readout(embedding_0).unsqueeze(-1)

        aggregated_output = self.global_aggregation(node_output, batch)
        output = self.mlp(aggregated_output).squeeze()

        return output
