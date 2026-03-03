"""
Improved PointNeXt encoder variants for ablation experiments.

Registers three new encoder classes with the openpoints model registry:
  - PointNextEncoderMSCA:     Multi-Scale Feature Fusion + Channel Attention only
  - PointNextEncoderCSWAP:    Curvature-Sensitive Weighted Average Pooling only
  - PointNextEncoderImproved: Both improvements combined

All variants inherit from PointNextEncoder so they share the same
constructor interface and config structure.
"""
import copy
from typing import List, Type
import logging
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.models.build import MODELS
from openpoints.models.backbone.pointnext import (
    PointNextEncoder, SetAbstraction, InvResMLP,
    LocalAggregation, FeaturePropogation, get_reduction_fn,
)
from openpoints.models.layers import (
    create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP,
    create_grouper, furthest_point_sample, random_sample,
    three_interpolation, get_aggregation_feautres,
)

from modules import MultiScaleFeatureFusion, CurvatureSensitivePool


# ---------------------------------------------------------------------------
# Helper: a SetAbstraction variant that exposes dp for curvature pooling
# ---------------------------------------------------------------------------
class SetAbstractionCSWAP(nn.Module):
    """SetAbstraction with Curvature-Sensitive Weighted Average Pooling.

    Identical to the original SetAbstraction except that, when not in
    head mode, the neighbourhood aggregation step uses CurvatureSensitivePool
    instead of max-pool.
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1, stride=1,
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None
            ) if in_channels != channels[-1] else nn.Identity()
            self.act = create_act(act_args)

        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(
                channels[i], channels[i + 1],
                norm_args=norm_args if not is_head else None,
                act_args=None if i == len(channels) - 2 and (self.use_res or is_head) else act_args,
                **conv_args))
        self.convs = nn.Sequential(*convs)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            # Replace max-pool with curvature-sensitive pool
            self.cswap = CurvatureSensitivePool()
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            f = self.convs(f)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            dp, fj = self.grouper(new_p, p, f)
            fj = get_aggregation_feautres(new_p, dp, fi, fj, feature_type=self.feature_type)
            f = self.cswap(self.convs(fj), dp)  # curvature-sensitive pooling
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f


# ---------------------------------------------------------------------------
# 1. PointNextEncoderMSCA — multi-scale fusion + channel attention only
# ---------------------------------------------------------------------------
@MODELS.register_module()
class PointNextEncoderMSCA(PointNextEncoder):
    """PointNeXt encoder with Multi-Scale Feature Fusion + Channel Attention.

    After the standard encoder stages, a MultiScaleFeatureFusion module
    combines all stage outputs back to the finest resolution with
    channel attention, and the fused feature is prepended to the feature
    list so that the decoder can leverage it.
    """

    def __init__(self, msca_reduction: int = 4, **kwargs):
        super().__init__(**kwargs)
        # Build the MSCA fusion after parent init so channel_list is ready
        self.msca = MultiScaleFeatureFusion(
            self.channel_list, target_channels=self.channel_list[0],
            reduction=msca_reduction)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        # Multi-scale fusion: fuse encoder stage features (f[1:])
        fused = self.msca(p[1:], f[1:])  # (B, C0, N1)
        # Replace the finest encoder feature with fused version
        f[1] = fused
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)


# ---------------------------------------------------------------------------
# 2. PointNextEncoderCSWAP — curvature-sensitive pooling only
# ---------------------------------------------------------------------------
@MODELS.register_module()
class PointNextEncoderCSWAP(PointNextEncoder):
    """PointNeXt encoder with Curvature-Sensitive Weighted Avg Pooling.

    Replaces the standard SetAbstraction (max-pool) blocks with
    SetAbstractionCSWAP blocks that use curvature-derived attention.
    """

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        # Use CSWAP variant instead of plain SetAbstraction
        layers.append(SetAbstractionCSWAP(
            self.in_channels, channels,
            self.sa_layers if not is_head else 1, stride,
            group_args=group_args,
            sampler=self.sampler,
            norm_args=self.norm_args, act_args=self.act_args,
            conv_args=self.conv_args,
            is_head=is_head, use_res=self.sa_use_res,
            **self.aggr_args))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(
                self.in_channels,
                aggr_args=self.aggr_args,
                norm_args=self.norm_args, act_args=self.act_args,
                group_args=group_args, conv_args=self.conv_args,
                expansion=self.expansion, use_res=self.use_res))
        return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 3. PointNextEncoderImproved — both improvements
# ---------------------------------------------------------------------------
@MODELS.register_module()
class PointNextEncoderImproved(PointNextEncoder):
    """PointNeXt encoder with *both* MSCA and CSWAP improvements."""

    def __init__(self, msca_reduction: int = 4, **kwargs):
        # We override _make_enc *before* calling super().__init__ by
        # monkeypatching; but since Python MRO calls __init__ in order,
        # we simply override _make_enc and add MSCA in __init__.
        super().__init__(**kwargs)
        self.msca = MultiScaleFeatureFusion(
            self.channel_list, target_channels=self.channel_list[0],
            reduction=msca_reduction)

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        """Use CSWAP SetAbstraction."""
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstractionCSWAP(
            self.in_channels, channels,
            self.sa_layers if not is_head else 1, stride,
            group_args=group_args,
            sampler=self.sampler,
            norm_args=self.norm_args, act_args=self.act_args,
            conv_args=self.conv_args,
            is_head=is_head, use_res=self.sa_use_res,
            **self.aggr_args))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(
                self.in_channels,
                aggr_args=self.aggr_args,
                norm_args=self.norm_args, act_args=self.act_args,
                group_args=group_args, conv_args=self.conv_args,
                expansion=self.expansion, use_res=self.use_res))
        return nn.Sequential(*layers)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        # Multi-scale fusion on encoder stage features
        fused = self.msca(p[1:], f[1:])
        f[1] = fused
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)
