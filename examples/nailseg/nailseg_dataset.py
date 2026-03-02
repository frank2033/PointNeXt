"""
NailSeg Dataset for fingernail point cloud segmentation.

Expected data directory structure:
    data/NailSeg/
        train/
            sample_0.npy
            sample_1.npy
            ...
        val/
            sample_0.npy
            ...
        test/
            sample_0.npy
            ...

Each .npy file should contain a numpy array of shape (N, C), where:
    - Columns 0-2: x, y, z coordinates
    - Columns 3 to C-2 (optional): additional features (e.g., normals, colors)
    - Last column (C-1): integer label (0=non-nail, 1=nail)

If C == 4, no additional features are used (xyz + label only).
If C > 4, columns 3 to C-2 are treated as point features.
"""
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from openpoints.dataset.build import DATASETS


@DATASETS.register_module()
class NailSeg(Dataset):
    """Fingernail segmentation dataset.

    Loads .npy point cloud files for binary segmentation of fingernails.

    Args:
        data_root (str): Root directory of the dataset.
        num_points (int): Number of points to sample per cloud.
        split (str): One of 'train', 'val', 'test'.
        transform: Data augmentation transforms.
    """
    num_classes = 2
    classes = ['non_nail', 'nail']

    def __init__(self,
                 data_root='data/NailSeg',
                 num_points=1024,
                 split='train',
                 transform=None,
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.num_points = num_points
        self.split = split
        self.transform = transform

        # Load all .npy files from the split directory
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Data directory not found: {split_dir}. "
                f"Please organize your data as: {data_root}/{{train,val,test}}/*.npy"
            )

        self.file_list = []
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith('.npy') and not fname.startswith('._'):
                full_path = os.path.join(split_dir, fname)
                try:
                    # Quick file accessibility check (not full format validation)
                    with open(full_path, 'rb') as f:
                        f.read(10)
                    self.file_list.append(full_path)
                except Exception as e:
                    logging.warning(f"Skipping unreadable file: {full_path}, error: {e}")
        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"No valid .npy files found in {split_dir}."
            )

        logging.info(f"NailSeg [{split}]: loaded {len(self.file_list)} samples from {split_dir}")

    @property
    def num_per_class(self):
        """Count number of points per class across the dataset (for weighted loss)."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for f in self.file_list:
            data = np.load(f)
            labels = data[:, -1].astype(np.int64)
            for c in range(self.num_classes):
                counts[c] += np.sum(labels == c)
        return counts

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Load point cloud from .npy file
        data_np = np.load(self.file_list[index]).astype(np.float32)

        # Extract coordinates, features, and labels
        pos = data_np[:, :3]
        label = data_np[:, -1].astype(np.int64)

        # Additional features (columns between xyz and label)
        if data_np.shape[1] > 4:
            feat = data_np[:, 3:-1]
        else:
            feat = None

        # Sample or pad to fixed number of points
        n_points = pos.shape[0]
        if n_points >= self.num_points:
            if 'train' in self.split:
                choice = np.random.choice(n_points, self.num_points, replace=False)
            else:
                choice = np.arange(self.num_points)
        else:
            choice = np.random.choice(n_points, self.num_points, replace=True)

        pos = pos[choice]
        label = label[choice]

        data = {'pos': pos, 'y': label}
        if feat is not None:
            data['x'] = feat[choice]

        if self.transform is not None:
            data = self.transform(data)

        return data
