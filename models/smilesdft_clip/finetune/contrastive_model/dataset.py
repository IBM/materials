import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset


# ###################################### Load .smi files (SMILES) ######################################
def load_smi_files(directory):
    """
    Load the contents of all .smi files in the specified directory.

    Parameters:
    directory (str): The path to the directory containing the .smi files.

    Returns:
    list: A list containing the contents of all .smi files.
    """
    # Initialize an empty list to hold the contents of all files
    all_smi_contents = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".smi"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                all_smi_contents.append(file.read())

    return all_smi_contents

# ###################################### Load .npy files (3D Grids) ####################################
def load_npy_files(directory):
    # train_images = VQGANDataset(root_dir=directory)
    # samples_list = []

    # for sample in train_images:
    #     samples_list.append(torch.tensor(sample['data']))

    # consolidated_npy_files = torch.stack(samples_list).to(CFG.device)
    # return consolidated_npy_files
    """
    Load the contents of all .npy files in the specified directory.

    Parameters:
    directory (str): The path to the directory containing the .npy files.

    Returns:
    list: A list containing the contents of all .npy files.
    """
    # Initialize an empty list to hold the contents of all files
    all_npy_contents = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            all_npy_contents.append(filename)

    return all_npy_contents


#################################### Data Loader #####################################################
# class CLIPDataset(Dataset):
#     def __init__(self, tensor_data, list_data):
#         self.tensor_data = tensor_data
#         self.list_data = list_data

#     def __len__(self):
#         return len(self.list_data)

#     def __getitem__(self, idx):
#         return {
#             'image': self.tensor_data[idx],
#             'caption': self.list_data[idx]
#         }


class CLIPDataset(Dataset):
    def __init__(self, dataset, target, directory, grid_filenames, smi_contents):
        self.dataset = dataset
        self.target = target
        self.directory = directory
        self.grid_filenames = grid_filenames
        self.smi_contents = smi_contents

    def __len__(self):
        return len(self.grid_filenames)

    def __getitem__(self, idx):
        npy_file = self.grid_filenames.iloc[idx]
        smi = self.smi_contents.iloc[idx]
        target = self.dataset.iloc[idx][self.target]

        # preprocess 3D Grids
        try:
            numpy_file = np.load(os.path.join(self.directory, npy_file))
            torch_np = torch.from_numpy(numpy_file)
            torch_np = torch_np.unsqueeze(0).unsqueeze(0).float()  # Convert to float and move to appropriate device
            interpolated_data = F.interpolate(input=torch_np, size=(128, 128, 128), mode='trilinear')

            # Apply tanh and log operations
            interpolated_data_tanh = torch.tanh(interpolated_data)
            interpolated_data_log = torch.log(interpolated_data + 1).squeeze(0)  # Adding 1 to avoid log(0)
        except Exception as e:
            print(f"Error loading file '{npy_file}': {e}")
            interpolated_data_log = None

        return {
            'image': interpolated_data_log,
            'caption': smi,
            'target': target
        }
    

def build_loaders(config, dataset, valid_split=0.1, mode="train"):
    if mode == "train":
        # Calculate sizes for training and validation splits
        train_size = int((1 - valid_split) * len(dataset))
        valid_size = len(dataset) - train_size
        
        # Split the dataset
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        
        # Create DataLoader for training data
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=False,
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=config.valid_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=False,
        )
        
        return train_loader, valid_loader
    else:
        # For validation mode, we don't split the dataset, just return the DataLoader
        return DataLoader(
            dataset, 
            batch_size=config.valid_batch_size,
            num_workers=config.num_workers if config.num_workers else 0,
            shuffle=False,
            pin_memory=False,
        )