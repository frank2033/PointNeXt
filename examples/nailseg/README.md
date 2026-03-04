# NailSeg — 指甲二类别点云分割

Binary point cloud segmentation for nail vs non-nail using PointNeXt.

## 数据准备 / Data Preparation

将 `.npy` 文件按 **train / val / test** 三个子文件夹放入数据目录中：

```
data/nailseg/
├── train/
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
├── val/
│   ├── sample_100.npy
│   └── ...
└── test/
    ├── sample_200.npy
    └── ...
```

每个 `.npy` 文件的形状为 `(N, 7)`：
- 列 0–2：xyz 坐标
- 列 3–5：额外特征（法线、颜色等）
- 列 6：分割标签（`0` = 非指甲，`1` = 指甲）

Each `.npy` file has shape `(N, 7)`:
- Columns 0–2: xyz coordinates
- Columns 3–5: additional features (normals, colors, etc.)
- Column 6: segmentation label (`0` = non-nail, `1` = nail)

## 快速开始 / Quick Start

### 训练基线模型 / Train baseline (original PointNeXt-S)

```bash
python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s.yaml
```

如需自定义数据路径：
```bash
python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s.yaml \
    dataset.common.data_root /path/to/your/nailseg
```

### 消融实验 / Ablation Experiments

| 实验 | 配置文件 | 编码器 |
|------|---------|--------|
| (1) 原始 PointNeXt | `cfgs/nailseg/pointnext-s.yaml` | `PointNextEncoder` |
| (2) 仅多尺度融合+通道注意力 | `cfgs/nailseg/pointnext-s-msca.yaml` | `PointNextEncoderMSCA` |
| (3) 仅曲率敏感池化 | `cfgs/nailseg/pointnext-s-cswap.yaml` | `PointNextEncoderCSWAP` |
| (4) 完整改进模型 | `cfgs/nailseg/pointnext-s-improved.yaml` | `PointNextEncoderImproved` |

```bash
# (1) Baseline
python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s.yaml

# (2) Multi-Scale Feature Fusion + Channel Attention only
python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s-msca.yaml

# (3) Curvature-Sensitive Weighted Average Pooling only
python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s-cswap.yaml

# (4) Full improved model (MSCA + CSWAP)
python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s-improved.yaml
```

### 验证/测试 / Validation / Test

```bash
python examples/nailseg/main.py --cfg cfgs/nailseg/pointnext-s.yaml \
    mode=test pretrained_path=/path/to/ckpt_best.pth
```

## 文件结构 / File Structure

```
examples/nailseg/
├── main.py                  # 训练入口 (training entry point)
├── nailseg_dataset.py       # NailSeg 数据集类 (dataset class)
├── modules.py               # 改进模块: ChannelAttention, MultiScaleFeatureFusion, CurvatureSensitivePool
├── pointnext_improved.py    # 改进编码器: PointNextEncoderMSCA, PointNextEncoderCSWAP, PointNextEncoderImproved
└── README.md                # 本文件

cfgs/nailseg/
├── default.yaml             # 默认数据集和训练配置
├── pointnext-s.yaml         # (1) 原始 PointNeXt-S 基线
├── pointnext-s-msca.yaml    # (2) 仅 MSCA 改进
├── pointnext-s-cswap.yaml   # (3) 仅 CSWAP 改进
├── pointnext-s-improved.yaml# (4) 完整改进
├── pointnet.yaml            # PointNet 基线
└── pointnet++.yaml          # PointNet++ 基线
```

## 关键配置参数 / Key Configuration

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset.common.data_root` | `data/nailseg` | npy 数据根目录 |
| `dataset.common.num_points` | `2048` | 每个样本采样点数 |
| `num_classes` | `2` | 类别数 (non-nail + nail) |
| `batch_size` | `8` | 批大小 |
| `epochs` | `300` | 训练轮数 |
| `lr` | `0.001` | 初始学习率 |

## 改进模块说明 / Improvement Modules

### Multi-Scale Feature Fusion + Channel Attention (MSCA)

将编码器各尺度的特征投影到统一通道宽度，用最近邻插值上采样到最高分辨率，
求和融合后通过 Squeeze-and-Excitation 通道注意力重新加权。

### Curvature-Sensitive Weighted Average Pooling (CSWAP)

将 SetAbstraction 中的 max-pool 替换为基于局部曲率的注意力加权平均池化。
通过邻域相对位置向量的方差估计局部曲率，使模型对高曲率区域（如指甲边缘）
更加敏感。
