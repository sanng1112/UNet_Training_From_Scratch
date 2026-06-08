"""Khối xây dựng cho UNet-Lite — tách riêng để dễ test và tái sử dụng.

Các block được giữ tương thích với cv_nets layers.
"""
from __future__ import annotations

from typing import Any, List

import torch
from torch import nn, Tensor

from cv_nets.layers import Conv2d, ConvTranspose2d
from cv_nets.layers.activation import build_activation_layer
from cv_nets.layers.normalization import build_normalization_layer


class DoubleConvBlock(nn.Module):
    """Hai lần (Conv 3x3 -> Norm -> Act) — dùng cho bottleneck."""

    def __init__(self, opts: Any, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, opts=opts)
        self.norm1 = build_normalization_layer(opts, num_features=out_channels)
        self.act1 = build_activation_layer(opts)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, opts=opts)
        self.norm2 = build_normalization_layer(opts, num_features=out_channels)
        self.act2 = build_activation_layer(opts)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class InvertedResidual(nn.Module):
    """Khối cơ sở của MobileNetV2 (expand -> depthwise -> project)."""

    def __init__(self, opts: Any, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        use_expand = expand_ratio != 1

        layers: List[nn.Module] = []
        if use_expand:
            layers += [
                Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, opts=opts),
                build_normalization_layer(opts, num_features=hidden_dim),
                build_activation_layer(opts),
            ]
        layers += [
            Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, opts=opts),
            build_normalization_layer(opts, num_features=hidden_dim),
            build_activation_layer(opts),
        ]
        layers += [
            Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, opts=opts),
            build_normalization_layer(opts, num_features=out_channels),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MV2DoubleBlock(nn.Module):
    """Hai Inverted Residual liên tiếp (stride=1)."""

    def __init__(self, opts: Any, in_channels: int, out_channels: int, expand_ratio: int = 2) -> None:
        super().__init__()
        self.block1 = InvertedResidual(opts, in_channels, out_channels, stride=1, expand_ratio=expand_ratio)
        self.block2 = InvertedResidual(opts, out_channels, out_channels, stride=1, expand_ratio=expand_ratio)

    def forward(self, x: Tensor) -> Tensor:
        return self.block2(self.block1(x))



class AttentionBottleneck(nn.Module):
    """Bottleneck với self-attention: MV2Block → Attention → MV2Block.

    Dùng patch-based Linear Self Attention để capture global context
    mà không tốn quá nhiều VRAM.
    """

    def __init__(self, opts: Any, in_channels: int, out_channels: int,
                 expand_ratio: int = 2, patch_size: int = 2):
        super().__init__()
        self.in_proj = MV2DoubleBlock(opts, in_channels, out_channels, expand_ratio=expand_ratio)
        self.attn = LinearSelfAttention(opts=opts, embed_dim=out_channels, patch_size=patch_size)
        self.out_proj = MV2DoubleBlock(opts, out_channels, out_channels, expand_ratio=expand_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.attn(x)
        x = self.out_proj(x)
        return x


class LinearSelfAttention(nn.Module):
    """Linear Self Attention với patch unfolding/folding.

    Input: [B, C, H, W] → unfold thành patches [B, C, P, N]
    → Scaled dot-product attention → fold về [B, C, H, W]
    """

    def __init__(self, opts: Any, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.P = patch_size * patch_size
        self.qkv_proj = Conv2d(embed_dim, embed_dim * 3, kernel_size=1, stride=1, padding=0, opts=opts)
        self.out_proj = Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0, opts=opts)

    def _unfold(self, x: Tensor):
        """[B, C, H, W] → [B, C, P, N] với P=patch_size^2, N=số patch."""
        b, c, h, w = x.shape
        ph = pw = self.patch_size
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
            h, w = h + pad_h, w + pad_w
        num_h, num_w = h // ph, w // pw
        n = num_h * num_w
        x = x.view(b, c, num_h, ph, num_w, pw)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, c, n, self.P)
        x = x.permute(0, 1, 3, 2).contiguous()
        return x, (num_h, num_w, h, w)

    def _fold(self, x: Tensor, grid_shape: tuple):
        """[B, C, P, N] → [B, C, H, W]."""
        b, c, p, n = x.shape
        num_h, num_w, h, w = grid_shape
        ph = pw = self.patch_size
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, c, num_h, num_w, ph, pw)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, c, h, w)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x_patch, shape = self._unfold(x)
        qkv = self.qkv_proj(x)
        qkv_patch, _ = self._unfold(qkv)
        q, k, v = torch.chunk(qkv_patch, 3, dim=1)

        scale = (q.size(1) * q.size(2)) ** -0.5
        q_flat = q.permute(0, 3, 1, 2).flatten(2)
        k_flat = k.permute(0, 3, 1, 2).flatten(2)

        attn = (q_flat @ k_flat.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        v_flat = v.permute(0, 3, 1, 2).flatten(2)
        out_flat = attn @ v_flat
        out = out_flat.transpose(1, 2).view_as(v)

        out = self._fold(out, shape)
        out = self.out_proj(out)
        return out
