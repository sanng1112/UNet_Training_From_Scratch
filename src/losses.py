"""Hàm mất mát cho Human Segmentation — xử lý mất cân bằng lớp person/nền.

Tất cả nhận `pred` là logits (chưa sigmoid), `target` là mask nhị phân float.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _dice_loss(pred_sig: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice, tính trên từng sample (dim H, W) rồi lấy trung bình batch."""
    intersection = (pred_sig * target).sum(dim=(2, 3))
    denom = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - (2 * intersection + eps) / (denom + eps)
    return dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE (phạt từng pixel) + Dice (tối ưu hình dạng tổng thể)."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = _dice_loss(torch.sigmoid(pred), target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


class FocalDiceLoss(nn.Module):
    """Focal Loss (giảm trọng số pixel dễ) + Dice. Mặc định alpha=0.25, gamma=2."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, bce_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_sig = torch.sigmoid(pred)
        p_t = pred_sig * target + (1 - pred_sig) * (1 - target)
        focal = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()
        dice = _dice_loss(pred_sig, target)
        return self.bce_weight * focal + (1 - self.bce_weight) * dice


class OHEMDiceLoss(nn.Module):
    """OHEM BCE (chỉ phạt top-k pixel khó nhất) + Dice. Tốt cho person rất nhỏ."""

    def __init__(self, top_k_ratio: float = 0.25, bce_weight: float = 0.5):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none").view(-1)
        k = max(1, int(self.top_k_ratio * bce.numel()))
        ohem_bce = torch.topk(bce, k)[0].mean()
        dice = _dice_loss(torch.sigmoid(pred), target)
        return self.bce_weight * ohem_bce + (1 - self.bce_weight) * dice


def build_loss(cfg) -> nn.Module:
    """Tạo hàm mất mát theo cfg.loss_name."""
    name = cfg.loss_name.lower()
    if name == "focal_dice":
        return FocalDiceLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, bce_weight=cfg.bce_weight)
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight=cfg.bce_weight)
    if name == "ohem_dice":
        return OHEMDiceLoss(top_k_ratio=cfg.ohem_top_k, bce_weight=cfg.bce_weight)
    raise ValueError(f"Không hỗ trợ loss_name='{cfg.loss_name}'. Chọn: focal_dice | bce_dice | ohem_dice")
