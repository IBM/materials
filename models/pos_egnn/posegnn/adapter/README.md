# PosEGNN + LoRA

This adapter injects LoRA into mergeable linear layers of **PosEGNN** and exports merged weights that load into a plain `PosEGNN` with `strict=True`.

## Skipped layers

These layers have a built-in activation inside their Dense block, which makes algebraic merging incorrect. They are always skipped so that merged exports match adapter-enabled outputs exactly.

- `encoder.neighbor_embedding.combine.dense_layers.0`
- `encoder.edge_embedding.edge_up.dense_layers.0`
- `encoder.gata.0.gamma_s.0`
- `encoder.gata.0.gamma_v.0`
- `encoder.gata.0.phik_w_ra`
- `encoder.gata.0.edge_attr_up.dense_layers.0`
- `encoder.gata.1.gamma_s.0`
- `encoder.gata.1.gamma_v.0`
- `encoder.gata.1.phik_w_ra`
- `encoder.gata.1.edge_attr_up.dense_layers.0`
- `encoder.gata.2.gamma_s.0`
- `encoder.gata.2.gamma_v.0`
- `encoder.gata.2.phik_w_ra`
- `encoder.gata.2.edge_attr_up.dense_layers.0`
- `encoder.gata.3.gamma_s.0`
- `encoder.gata.3.gamma_v.0`
- `encoder.gata.3.phik_w_ra`
- `encoder.eqff.0.gamma_m.0`
- `encoder.eqff.1.gamma_m.0`
- `encoder.eqff.2.gamma_m.0`
- `encoder.eqff.3.gamma_m.0`

Skipping only affects where LoRA is attached. The base model behavior is unchanged.

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