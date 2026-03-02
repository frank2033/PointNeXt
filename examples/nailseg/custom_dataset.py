"""
Standalone CustomDataset for fingernail point cloud segmentation.

This is a self-contained dataset class that can be used independently
of the openpoints framework. It returns (points, seg) tuples.

Data format: each .npy file is (N, C) where:
    - Columns 0-2: x, y, z coordinates
    - Columns 3 to C-2: additional features (e.g., r, g, b)
    - Last column (C-1): label (0=non_nail, 1=nail)

Directory structure: root/{train,val,test}/*.npy
"""
import os
import math
import numpy as np
import torch
import torch.utils.data as data


def pc_normalize(pc):
    """
    对点云数据进行归一化
    参数:
        pc: 点云数据，形状为 (N, D)，其中 N 是点的数量，D 是每个点的维度
    返回:
        归一化后的点云数据
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class CustomDataset(data.Dataset):
    """
    自定义数据集类，用于加载点云分割数据
    """
    def __init__(self, root, classification=False, split='train',
                 data_augmentation=True, num_points=2048):
        """
        初始化自定义数据集
        参数:
            root: 数据集根目录路径
            classification: 是否用于分类任务（默认为 False，即分割任务）
            split: 数据集划分 ('train', 'val', 'test')
            data_augmentation: 是否进行数据增强
            num_points: 每个点云的固定点数（默认2048）
        """
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.num_seg_classes = 2
        self.classification = classification
        self.num_points = num_points

        self.datapath = []
        split_dir = os.path.join(self.root, self.split)

        if not os.path.exists(split_dir):
            raise ValueError(f"数据集划分目录不存在：{split_dir}")

        for filename in os.listdir(split_dir):
            if filename.endswith('.npy') and not filename.startswith('._'):
                full_path = os.path.join(split_dir, filename)
                try:
                    with open(full_path, 'rb') as f:
                        f.read(10)
                    self.datapath.append(full_path)
                except Exception as e:
                    print(f"跳过损坏/不可读文件：{full_path}，错误：{e}")

        if len(self.datapath) == 0:
            raise ValueError(f"在 {split_dir} 中未找到有效的.npy数据文件！")

    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        返回:
            points: 点云数据 (num_points, C-1)
            seg: 分割标签 (num_points,)
        """
        fn = self.datapath[index]
        try:
            data = np.load(fn, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"加载文件 {fn} 失败：{e}")

        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError(
                f"文件 {fn} 数据格式错误，期望至少4列（x,y,z,label），"
                f"实际：{data.shape}"
            )

        # 分离点云特征和标签（最后一列为标签）
        points = data[:, :-1]
        seg = data[:, -1]

        # 固定点数采样
        current_points = points.shape[0]
        if current_points >= self.num_points:
            choice = np.random.choice(current_points, self.num_points, replace=False)
        else:
            choice = np.random.choice(current_points, self.num_points, replace=True)

        points = points[choice, :]
        seg = seg[choice]

        # 应用数据增强（仅训练集）
        if self.data_augmentation and self.split == 'train':
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            points[:, :3] = np.dot(points[:, :3], rotation_matrix)
            coord_noise = np.random.normal(0, 0.01, points[:, :3].shape)
            points[:, :3] += coord_noise

        # 对点云坐标归一化
        points[:, :3] = pc_normalize(points[:, :3])

        points = torch.from_numpy(points.astype(np.float32))
        seg = torch.from_numpy(seg.astype(np.int64))

        if self.classification:
            cls = seg[0]
            return points, cls
        else:
            return points, seg

    def __len__(self):
        return len(self.datapath)

    def seg_classes(self):
        return self.num_seg_classes
