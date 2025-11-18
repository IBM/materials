# PosEGNN + LoRA

This adapter injects LoRA into mergeable linear layers of **PosEGNN** and exports merged weights that load into a plain `PosEGNN` with `strict=True`.

## Usage

```python
# 1) build and load the backbone
backbone = PosEGNN(checkpoint_dict["config"])
backbone.load_state_dict(checkpoint_dict["state_dict"], strict=True)

# 2) wrap with LoRA (post-activation linears are skipped automatically)
cfg = LoRAConfig(rank=8, alpha=8, dropout=0.0, freeze_base=True, merge_on_save=True)
model = PosEGNNLoRAModel(backbone, cfg)

# 3) train or evaluate
out = model(batch)

# 4) export merged weights that load into plain PosEGNN
merged = model.state_dict_backbone(merged=True)
plain = PosEGNN(checkpoint_dict["config"])
plain.load_state_dict(merged, strict=True)