import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.multiprocessing as mp


class VQGANDataset(Dataset):
    def __init__(self, root_dir: str, file_paths: str, internal_resolution: int):
        super().__init__()
        self.root_dir = root_dir
        self.file_paths = file_paths
        self.internal_resolution = internal_resolution

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        filename = os.path.join(self.root_dir, self.file_paths[idx])
        try:
            numpy_file = np.load(filename)
            torch_np = torch.from_numpy(numpy_file)
            torch_np = torch_np.unsqueeze(0).unsqueeze(0).float()  # Convert to float and move to appropriate device
            interpolated_data = F.interpolate(input=torch_np, size=(self.internal_resolution, self.internal_resolution, self.internal_resolution), mode='trilinear')

            # Apply tanh and log operations
            interpolated_data_tanh = torch.tanh(interpolated_data)
            interpolated_data_log = torch.log(interpolated_data + 1).squeeze(0)  # Adding 1 to avoid log(0)

            return interpolated_data_log
        except Exception as e:
            print(f"Error loading file '{filename}': {e}")
            return None