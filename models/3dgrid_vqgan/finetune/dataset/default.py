import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.multiprocessing as mp

class GridDataset(Dataset):
    def __init__(self, dataset, target: str, root_dir: str, internal_resolution: int):
        super().__init__()
        self.dataset = dataset
        self.target = target
        self.root_dir = root_dir
        self.internal_resolution = internal_resolution

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        target = self.dataset.iloc[idx][self.target]
        filename = self.dataset.iloc[idx]['3d_grid']
        try:
            numpy_file = np.load(os.path.join(self.root_dir, filename))
            torch_np = torch.from_numpy(numpy_file)
            torch_np = torch_np.unsqueeze(0).unsqueeze(0).float()  # Convert to float and move to appropriate device
            interpolated_data = F.interpolate(input=torch_np, size=(self.internal_resolution, self.internal_resolution, self.internal_resolution), mode='trilinear')

            # Apply tanh and log operations
            # interpolated_data_tanh = torch.tanh(interpolated_data)
            interpolated_data_log = torch.log(interpolated_data + 1).squeeze(0)  # Adding 1 to avoid log(0)

            return interpolated_data_log, target
        except Exception as e:
            print(f"Error loading file '{filename}': {e}")
            return None