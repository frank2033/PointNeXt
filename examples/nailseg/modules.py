"""
Improvement modules for PointNeXt ablation experiments.

Contains:
- ChannelAttention: Squeeze-and-Excitation style channel attention
- MultiScaleFeatureFusion: Fuses multi-scale encoder features with channel attention
- CurvatureSensitivePool: Curvature-sensitive weighted average pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Given input of shape (B, C, ...) it computes per-channel attention
    weights via global-average-pool → FC → ReLU → FC → Sigmoid and
    scales the channels accordingly.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, N) point features
        Returns:
            (B, C, N) re-weighted features
        """
        # Global average pooling over points
        w = x.mean(dim=-1)           # (B, C)
        w = self.fc(w).unsqueeze(-1)  # (B, C, 1)
        return x * w


class MultiScaleFeatureFusion(nn.Module):
    """Multi-Scale Feature Fusion with Channel Attention.

    After the encoder produces feature maps at different scales
    [f1, f2, ..., fK] with different channel widths and point counts,
    this module:
      1. Projects each scale to a common channel dimension via 1×1 conv
      2. Interpolates all scales back to the finest resolution
      3. Sums the projected features
      4. Applies channel attention on the fused result
    """

    def __init__(self, channel_list, target_channels=None, reduction=4):
        """
        Args:
            channel_list: list of channel dimensions from encoder stages
                          (e.g. [32, 64, 128, 256] for 4 downsampling stages)
            target_channels: the unified channel width after projection.
                             Defaults to channel_list[0] (finest level).
            reduction: reduction ratio for the channel attention.
        """
        super().__init__()
        if target_channels is None:
            target_channels = channel_list[0]
        self.target_channels = target_channels

        self.projections = nn.ModuleList()
        for c in channel_list:
            self.projections.append(
                nn.Conv1d(c, target_channels, 1, bias=False)
            )
        self.ca = ChannelAttention(target_channels, reduction=reduction)
        self.norm = nn.BatchNorm1d(target_channels)

    def forward(self, p_list, f_list):
        """
        Args:
            p_list: list of point positions [(B,N1,3), (B,N2,3), ...]
                    from fine to coarse (encoder output order, skipping input)
            f_list: list of point features [(B,C1,N1), (B,C2,N2), ...]
        Returns:
            fused: (B, target_channels, N1) fused features at finest resolution
        """
        target_n = f_list[0].shape[-1]  # finest resolution point count
        fused = torch.zeros(
            f_list[0].shape[0], self.target_channels, target_n,
            device=f_list[0].device, dtype=f_list[0].dtype)

        for i, (proj, fi) in enumerate(zip(self.projections, f_list)):
            projected = proj(fi)  # (B, target_channels, Ni)
            if projected.shape[-1] != target_n:
                projected = F.interpolate(
                    projected, size=target_n, mode='nearest')
            fused = fused + projected

        fused = self.norm(fused)
        fused = self.ca(fused)
        return fused


def estimate_curvature(pos, dp):
    """Estimate curvature from local position differences.

    Uses the variance of the relative position vectors in each
    neighbourhood as a proxy for local curvature. High variance
    indicates high surface curvature.

    Args:
        pos: (B, N_query, 3) query point positions (unused, kept for API)
        dp:  (B, 3, N_query, K) relative positions of K neighbours
    Returns:
        curvature: (B, 1, N_query, K) per-neighbour curvature weight
    """
    # dp: (B, 3, N, K) — relative position vectors
    # Variance of relative positions across the neighbourhood
    dp_var = dp.var(dim=-1, keepdim=True)  # (B, 3, N, 1)
    curvature = dp_var.sum(dim=1, keepdim=True)  # (B, 1, N, 1)
    # Per-neighbour: weight by distance magnitude deviation
    dp_norm = dp.norm(dim=1, keepdim=True)  # (B, 1, N, K)
    dp_mean = dp_norm.mean(dim=-1, keepdim=True)  # (B, 1, N, 1)
    deviation = (dp_norm - dp_mean).abs()  # (B, 1, N, K)
    # Broadcasting: curvature (B,1,N,1) + deviation (B,1,N,K) -> (B,1,N,K)
    weight = curvature + deviation  # (B, 1, N, K)
    return weight


class CurvatureSensitivePool(nn.Module):
    """Curvature-Sensitive Weighted Average Pooling.

    Instead of plain max-pooling over neighbourhood features, this
    module computes curvature-derived attention weights from the
    position differences (dp) and uses them to perform a weighted
    average. A learnable temperature parameter controls sharpness.
    """

    def __init__(self):
        super().__init__()
        # Learnable temperature (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, fj, dp):
        """
        Args:
            fj: (B, C, N, K) neighbourhood features after convolutions
            dp: (B, 3, N, K) relative position vectors (from grouper)
        Returns:
            f:  (B, C, N) aggregated features
        """
        weight = estimate_curvature(None, dp)  # (B, 1, N, K)
        # Softmax attention over neighbours (dim K)
        attn = F.softmax(weight * self.temperature, dim=-1)  # (B, 1, N, K)
        f = (fj * attn).sum(dim=-1)  # (B, C, N)
        return f
