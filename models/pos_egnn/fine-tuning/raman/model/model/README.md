# Architecture

This folder is the house for all of our models, of which `/Model` is the first. As a example, it has 4 files:

- `model.py`: Contains the PyTorch Lightning model, which is a wrapper around the TorchMD-Net model (we plan to change this with our own model in the future). Also, comes with training logic, logging, optimizer configuration, train/validation splitting, etc.
- `heads.py`: Contains all the task-specific heads that are feeded by the model's latent space. Here, we can enforce symmetries and properties of the target, therefore adding `inductive biases` (which arevery beneficial!)
- `losses.py`: Contains all the loss functions for each head. Here, we can define either classification or regression tasks. Also, this allow us to calculate as many metrics as we want and log them into `Tensorboard`. The main point is to go beyond `MSELoss` for everything, and be flexible.
- `utils.py`: Contains training utils. For example, it has a helper class that implements the Periodic Boundary Condition for batches.

Here is the main idea: We want a model take the atomic postions (with or without PBCs) and give us embeddings. These embeddings are used for many tasks, from CO2 capture performance to predicting atomic forces and energies. For global tasks, we do not care about equivariance, but we do care about them when predicting forces, therefore, we need to use an equivariant model.

<hr/><p align="center"><img src="../../images/Equivariant_encoder_highlight.svg"></p>
<hr/><p align="center"><img src="../../images/Equivariant_encoder_highlight_detailed.svg"></p>

The initial idea was to perform only pre-training tasks. However, we argue that using self-supervised AND supervised learning could improve fine-tuning capabilities. For that reason, we created a flexible way to handle multiple tasks. Each task must have a (so called task-specific) head, and a respective loss. The head will enforce physical contraints in the predictions, and behave as the user wants. Here is the example of the charge prediction head:

```python
class ChargeHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, context_channels: int) -> None:
        super(ChargeHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(in_channels + context_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: Tensor, context_vector: Tensor, batch: Tensor) -> Tensor:
        x_with_fidelity = torch.hstack([x, context_vector[batch]])
        charges = self.head(x_with_fidelity)

        # Enforce 0 sum following the PACMOF procedure
        for b in torch.unique(batch):
            batch_mask = batch == b
            batch_sum = charges[batch_mask].sum()
            charges_abs = torch.abs(charges[batch_mask])
            batch_abs_sum = charges_abs.sum()
            charges[batch_mask] -= batch_sum * (charges_abs / batch_abs_sum)

        return charges
```

Here, we can enforce the sum of the charges to be 0, which is expected due to the neutral nature of the materials.

<hr/><p align="center"><img src="../../images/Multitask_module_highlight.svg"></p>
<hr/><p align="center"><img src="../../images/Multitask_module_highlight_detailed.svg"></p>

