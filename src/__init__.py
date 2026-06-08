"""UNet-Lite Human Segmentation — package refactor.

Pipeline phân vùng người (COCO 2017) với kiến trúc UNet-Lite (khối MobileNetV2),
huấn luyện from-scratch trên GPU phổ thông (Mixed Precision + Gradient Accumulation).

Các module chính:
    config     -- siêu tham số tập trung (Config dataclass)
    model      -- UNetLite + các khối Inverted Residual
    dataset    -- COCOPersonDataset + augmentation + dataloader
    losses     -- BCEDice / FocalDice / OHEMDice
    metrics    -- IoU, Dice
    engine     -- vòng lặp train/eval, AMP, grad-accum, early stopping, resume
    visualize  -- vẽ history & overlay dự đoán
"""
from __future__ import annotations

from .config import Config
from .model import UNetLite, build_model
from .dataset import COCOPersonDataset, build_dataloaders, denormalize
from .losses import BCEDiceLoss, FocalDiceLoss, OHEMDiceLoss, build_loss
from .metrics import calculate_iou, calculate_dice
from .engine import fit, train_one_epoch, evaluate, load_checkpoint

__all__ = [
    "Config",
    "UNetLite",
    "build_model",
    "COCOPersonDataset",
    "build_dataloaders",
    "denormalize",
    "BCEDiceLoss",
    "FocalDiceLoss",
    "OHEMDiceLoss",
    "build_loss",
    "calculate_iou",
    "calculate_dice",
    "fit",
    "train_one_epoch",
    "evaluate",
    "load_checkpoint",
]
