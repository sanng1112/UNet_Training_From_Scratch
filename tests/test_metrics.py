"""Unit tests cho các độ đo — dùng unittest (không phụ thuộc pytest)."""
from __future__ import annotations

import torch

from src.metrics import calculate_iou, calculate_dice


class TestCalculateIoU:
    def test_perfect_match(self):
        """IoU = 1 khi pred giống hệt mask."""
        pred = torch.sigmoid(torch.full((2, 1, 32, 32), 10.0))
        mask = torch.ones(2, 1, 32, 32)
        iou = calculate_iou(pred, mask)
        assert abs(iou - 1.0) < 0.02, f"Expected ~1.0, got {iou}"

    def test_no_match(self):
        """IoU = 0 khi không có overlap."""
        pred = torch.sigmoid(torch.full((2, 1, 32, 32), -10.0))
        mask = torch.ones(2, 1, 32, 32)
        iou = calculate_iou(pred, mask)
        assert abs(iou - 0.0) < 0.02, f"Expected ~0.0, got {iou}"

    def test_half_overlap(self):
        """IoU cho overlap 50%."""
        pred = torch.zeros(1, 1, 32, 32)
        pred[:, :, :16, :] = 10.0  # nửa trên là person
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, :, :16] = 1.0   # nửa trái là person
        # overlap = quarter (16x16), union = 3 quarters
        iou = calculate_iou(pred, mask)
        # intersection=16*16=256, union=32*32-16*16=768, IoU=256/768≈0.333
        assert abs(iou - 0.333) < 0.02, f"Expected ~0.333, got {iou}"

    def test_different_threshold(self):
        """Threshold khác 0.5 hoạt động."""
        pred = torch.sigmoid(torch.full((1, 1, 32, 32), 0.0))  # sigmoid(0) = 0.5
        mask = torch.ones(1, 1, 32, 32)
        iou_strict = calculate_iou(pred, mask, threshold=0.6)  # 0.5 < 0.6 → pred=0
        iou_lenient = calculate_iou(pred, mask, threshold=0.4)  # 0.5 > 0.4 → pred=1
        assert iou_strict < iou_lenient, \
            f"Strict ({iou_strict}) should be < lenient ({iou_lenient})"


class TestCalculateDice:
    def test_dice_equals_iou_when_perfect(self):
        """Khi perfect match, Dice = 1, IoU = 1."""
        pred = torch.sigmoid(torch.full((2, 1, 32, 32), 10.0))
        mask = torch.ones(2, 1, 32, 32)
        dice = calculate_dice(pred, mask)
        iou = calculate_iou(pred, mask)
        assert abs(dice - 1.0) < 0.02, f"Dice: {dice}"
        assert abs(iou - 1.0) < 0.02, f"IoU: {iou}"

    def test_dice_higher_than_iou(self):
        """Dice >= IoU trong mọi trường hợp."""
        pred = torch.randn(4, 1, 32, 32)
        mask = (torch.rand(4, 1, 32, 32) > 0.7).float()
        dice = calculate_dice(pred, mask)
        iou = calculate_iou(pred, mask)
        assert dice >= iou - 1e-6, f"Dice ({dice}) < IoU ({iou})"
