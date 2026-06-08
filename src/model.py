"""Kiến trúc UNet-Lite cho Human Segmentation.

Kế thừa triết lý U-Net (encoder/decoder + skip-connection) nhưng thay khối học
nặng bằng Inverted Residual (MobileNetV2) để giảm số tham số mà vẫn giữ trường
nhìn. Mô-đun được giữ NGUYÊN tên với phiên bản notebook gốc để các checkpoint
`.pth` đã train trước đó vẫn nạp được bằng `load_state_dict`.

Các block được import từ src.blocks — giảm trùng lặp code.

Phụ thuộc cv_nets (chạy từ thư mục gốc dự án):
    Conv2d / ConvTranspose2d  -> bias=False mặc định
    build_normalization_layer -> BatchNorm2d
    build_activation_layer    -> LeakyReLU
"""
from __future__ import annotations

from typing import Any, List

import torch
from torch import nn, Tensor
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional

from cv_nets.layers import Conv2d, ConvTranspose2d

from .blocks import (
    MV2DoubleBlock,
    DoubleConvBlock,
    AttentionBottleneck,
)


class UNetLite(nn.Module):
    """UNet-Lite: encoder/decoder MobileNetV2 + skip connections.

    Args:
        opts: cấu hình cho cv_nets (None -> Conv bias=False, BatchNorm, LeakyReLU).
        num_classes: số kênh logit đầu ra (1 cho phân vùng nhị phân person/nền).
        features: số kênh từng tầng encoder, mặc định (8, 16, 32, 64, 128).
        expand_ratio: hệ số mở rộng của Inverted Residual.
        use_attention: nếu True, dùng AttentionBottleneck thay DoubleConvBlock.
    """

    def __init__(
        self,
        opts: Any = None,
        num_classes: int = 1,
        features=(8, 16, 32, 64, 128),
        expand_ratio: int = 2,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        features = list(features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.f_cat = FloatFunctional()

        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        in_ch = 3
        for feat in features:
            self.encoder_blocks.append(MV2DoubleBlock(opts, in_ch, feat, expand_ratio=expand_ratio))
            self.downsample_layers.append(
                Conv2d(feat, feat, kernel_size=2, stride=2, padding=0, opts=opts)
            )
            in_ch = feat

        if use_attention:
            self.bottleneck = AttentionBottleneck(
                opts, features[-1], features[-1] * 2,
                expand_ratio=expand_ratio, patch_size=2,
            )
        else:
            self.bottleneck = DoubleConvBlock(opts, features[-1], features[-1] * 2)

        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for feat in reversed(features):
            self.upsample_layers.append(
                ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2, padding=0, opts=opts)
            )
            self.decoder_blocks.append(MV2DoubleBlock(opts, feat * 2, feat, expand_ratio=expand_ratio))

        self.final_conv = Conv2d(features[0], num_classes, kernel_size=1, stride=1, padding=0, opts=opts)

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        skips: List[Tensor] = []
        for enc, down in zip(self.encoder_blocks, self.downsample_layers):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for i, (up, dec) in enumerate(zip(self.upsample_layers, self.decoder_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]               # lấy skip tương ứng theo thứ tự ngược
            x = self.f_cat.cat((skip, x), dim=1)
            x = dec(x)

        x = self.final_conv(x)
        x = self.dequant(x)
        return x


def build_model(cfg=None, opts: Any = None) -> UNetLite:
    """Khởi tạo UNetLite từ Config (nếu có), trả về model trên CPU."""
    if cfg is None:
        return UNetLite(opts=opts)
    return UNetLite(
        opts=opts,
        num_classes=cfg.num_classes,
        features=cfg.features,
        expand_ratio=cfg.expand_ratio,
        use_attention=getattr(cfg, 'use_attention', False),
    )


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    """Đếm số tham số học được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
