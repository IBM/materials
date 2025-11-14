from dataset import DEFAULTDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir, internal_resolution=cfg.model.internal_resolution)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir, internal_resolution=cfg.model.internal_resolution)
        sampler = None
        return train_dataset, val_dataset, sampler
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
