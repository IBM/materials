import yaml
from torch import Tensor


def tensor_representer(dumper, value):
    if value.dim() == 0:
        return dumper.represent_float(value.item())
    else:
        return dumper.represent_list(value.tolist())


yaml.add_representer(Tensor, tensor_representer)


def deep_update(existing_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in existing_dict and isinstance(existing_dict[key], dict):
            deep_update(existing_dict[key], value)
        else:
            existing_dict[key] = value


def update_yaml(yaml_path, dict_to_add):
    with open(yaml_path, "r") as f:
        existing_dict = yaml.safe_load(f) or {}

    deep_update(existing_dict, dict_to_add)

    with open(yaml_path, "w") as f:
        yaml.dump(existing_dict, f, default_flow_style=False)
