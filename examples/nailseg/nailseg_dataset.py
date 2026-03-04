"""
NailSeg dataset for nail point cloud segmentation.

Expects pre-split .npy files under data_root/{train,val,test}/.
Each .npy file has shape (N, 7): columns 0-2 are xyz, columns 3-5 are
additional features (e.g. normals or colors), and column 6 is the
segmentation label (0 = non-nail, 1 = nail).
"""
import os
import glob
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
from openpoints.dataset.build import DATASETS


@DATASETS.register_module()
class NailSeg(Dataset):
    """Nail segmentation dataset (ShapeNetPart-style).

    Class attributes mirror the ShapeNetPartNormal interface so that
    BasePartSeg / PointNextPartDecoder work out-of-the-box.
    """

    classes = ['nail']                       # single shape category
    seg_num = [2]                            # 2 part labels per shape
    cls_parts = {'nail': [0, 1]}             # part indices for each class
    cls2parts = [[0, 1]]                     # list form used by validation
    cls2partembed = torch.zeros(1, 2)        # one-hot embed (1 class × 2 parts)
    cls2partembed[0].scatter_(0, torch.LongTensor([0, 1]), 1)
    part2cls = {0: 'nail', 1: 'nail'}
    num_classes = 2
    shape_classes = 1

    def __init__(self,
                 data_root='data/nailseg',
                 num_points=2048,
                 split='train',
                 transform=None,
                 presample=False,
                 **kwargs):
        super().__init__()
        self.npoints = num_points
        self.split = split
        self.transform = transform

        # Resolve split directory
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            # fall back: treat val/test as test
            for alt in ('val', 'test'):
                alt_dir = os.path.join(data_root, alt)
                if os.path.isdir(alt_dir):
                    split_dir = alt_dir
                    break

        npy_files = sorted(glob.glob(os.path.join(split_dir, '*.npy')))
        # Filter out macOS hidden files
        npy_files = [f for f in npy_files if not os.path.basename(f).startswith('._')]

        if len(npy_files) == 0:
            raise RuntimeError(
                f'No .npy files found in {split_dir}. '
                f'Please place pre-split npy files under '
                f'{data_root}/train, {data_root}/val, {data_root}/test.')

        self.file_list = npy_files
        logging.info(f'NailSeg [{split}]: loaded {len(self.file_list)} samples '
                     f'from {split_dir}')

    def __getitem__(self, index):
        raw = np.load(self.file_list[index]).astype(np.float32)  # (N, 7)

        point_set = raw[:, :6]   # xyz + features
        seg = raw[:, 6].astype(np.int64)  # label

        # Sample / pad to fixed number of points
        if 'train' in self.split:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice]
            seg = seg[choice]
        else:
            if len(seg) >= self.npoints:
                point_set = point_set[:self.npoints]
                seg = seg[:self.npoints]
            else:
                # pad by repeating last point
                pad_n = self.npoints - len(seg)
                point_set = np.concatenate(
                    [point_set, np.tile(point_set[-1:], (pad_n, 1))], axis=0)
                seg = np.concatenate(
                    [seg, np.tile(seg[-1:], (pad_n,))], axis=0)

        # Single shape class → cls is always 0
        cls = np.array([0]).astype(np.int64)

        data = {
            'pos': point_set[:, 0:3],
            'x': point_set[:, 3:6],
            'cls': cls,
            'y': seg,
        }

        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.file_list)
