import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.multiprocessing as mp

class DEFAULTDataset(Dataset):
    def __init__(self, root_dir: str, internal_resolution: int):
        super().__init__()
        self.root_dir = root_dir
        self.internal_resolution = internal_resolution
        self.file_paths = self.get_data_files()

    def get_data_files(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory '{self.root_dir}' does not exist.")
        
        npy_file_names = os.listdir(self.root_dir)
        folder_names = [os.path.join(self.root_dir, npy_file_name) 
                        for npy_file_name in npy_file_names if npy_file_name.endswith('.npy')]
        return folder_names

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        filename = self.file_paths[idx]
        try:
            numpy_file = np.load(filename)
            torch_np = torch.from_numpy(numpy_file)
            torch_np = torch_np.unsqueeze(0).unsqueeze(0).float()  # Convert to float and move to appropriate device
            interpolated_data = F.interpolate(input=torch_np, size=(self.internal_resolution, self.internal_resolution, self.internal_resolution), mode='trilinear')

            # Apply tanh and log operations
            interpolated_data_tanh = torch.tanh(interpolated_data)
            interpolated_data_log = torch.log(interpolated_data + 1).squeeze(0)  # Adding 1 to avoid log(0)

            return {'data': interpolated_data_log}
        except Exception as e:
            print(f"Error loading file '{filename}': {e}")
            return None
