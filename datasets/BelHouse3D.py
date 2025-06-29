import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class BelHouse3DSemSegDataset(Dataset):
    def __init__(self, root, split='train', num_points=2048, use_blocks=True, transform=None):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.use_blocks = use_blocks

        subdir = 'blocks' if use_blocks else 'rooms'
        search_path = os.path.join(root, split, subdir, '*.npy')
        self.files = glob.glob(search_path)
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {search_path}")

        self.label_values = list(range(19))   
        self.ignored_labels = []             

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        if data.shape[0] >= self.num_points:
            indices = np.random.choice(data.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(data.shape[0], self.num_points, replace=True)
        data = data[indices]

        points = data[:, :3].astype(np.float32)
        labels = data[:, 3].astype(np.int64)

        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points), torch.from_numpy(labels)
