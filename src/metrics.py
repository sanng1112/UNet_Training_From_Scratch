"""Độ đo cho Semantic Segmentation nhị phân (logits đầu vào)."""
from __future__ import annotations

import torch


@torch.no_grad()
def calculate_iou(preds: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection over Union trung bình trên batch.

    Args:
        preds: logits, shape [B, 1, H, W].
        masks: ground-truth nhị phân, shape [B, 1, H, W].
    """
    preds = torch.sigmoid(preds) > threshold
    masks = masks > threshold
    intersection = (preds & masks).float().sum((1, 2, 3))
    union = (preds | masks).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


@torch.no_grad()
def calculate_dice(preds: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> float:
    """Dice score (F1) trung bình trên batch."""
    preds = torch.sigmoid(preds) > threshold
    masks = masks > threshold
    intersection = (preds & masks).float().sum((1, 2, 3))
    denom = preds.float().sum((1, 2, 3)) + masks.float().sum((1, 2, 3))
    dice = (2 * intersection + 1e-6) / (denom + 1e-6)
    return dice.mean().item()
