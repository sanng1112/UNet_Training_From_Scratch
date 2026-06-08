"""Unit tests cho các hàm mất mát — dùng unittest (không phụ thuộc pytest)."""
from __future__ import annotations

import torch

from src.losses import BCEDiceLoss, FocalDiceLoss, OHEMDiceLoss, build_loss
from src.config import Config


B, H, W = 2, 64, 64


class TestBCEDiceLoss:
    def __init__(self):
        self.pred_logits = torch.randn(B, 1, H, W)
        self.target_perfect = torch.zeros(B, 1, H, W)
        self.target_perfect[:, :, 10:30, 10:30] = 1.0
        self.target_empty = torch.zeros(B, 1, H, W)

    def test_output_positive(self):
        """Loss phải là số dương."""
        loss_fn = BCEDiceLoss()
        loss = loss_fn(self.pred_logits, self.target_perfect)
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    def test_perfect_prediction_low_loss(self):
        """Khi prediction hoàn hảo, loss phải rất nhỏ."""
        loss_fn = BCEDiceLoss(bce_weight=1.0)
        # pred giống hệt target (logit từ xác suất)
        pred = torch.log(self.target_perfect + 1e-7) - torch.log(1 - self.target_perfect + 1e-7)
        loss = loss_fn(pred, self.target_perfect)
        assert loss.item() < 0.1, f"Perfect prediction loss too high: {loss.item()}"


class TestFocalDiceLoss:
    def __init__(self):
        self.pred_logits = torch.randn(B, 1, H, W)
        self.target_perfect = torch.zeros(B, 1, H, W)
        self.target_perfect[:, :, 10:30, 10:30] = 1.0

    def test_focal_lower_than_bce(self):
        """Focal loss phải thấp hơn BCE trên cùng input (do giảm trọng số pixel dễ)."""
        focal = FocalDiceLoss(alpha=0.25, gamma=2.0, bce_weight=1.0)
        bce = BCEDiceLoss(bce_weight=1.0)
        loss_f = focal(self.pred_logits, self.target_perfect)
        loss_b = bce(self.pred_logits, self.target_perfect)
        assert loss_f.item() <= loss_b.item() * 1.1, \
            f"Focal ({loss_f.item()}) should be <= BCE ({loss_b.item()}) * 1.1"


class TestOHEMDiceLoss:
    def __init__(self):
        self.pred_logits = torch.randn(B, 1, H, W)
        self.target_perfect = torch.zeros(B, 1, H, W)
        self.target_perfect[:, :, 10:30, 10:30] = 1.0

    def test_ohem_selects_hard_pixels(self):
        """OHEM chỉ phạt top-k pixel khó nhất, loss phải khác BCE."""
        ohem = OHEMDiceLoss(top_k_ratio=0.25, bce_weight=1.0)
        loss_o = ohem(self.pred_logits, self.target_perfect)
        assert loss_o.item() > 0, f"Expected positive OHEM loss, got {loss_o.item()}"

    def test_top_k_1_percent(self):
        """top_k_ratio=0.01 vẫn hoạt động."""
        ohem = OHEMDiceLoss(top_k_ratio=0.01, bce_weight=1.0)
        loss = ohem(self.pred_logits, self.target_perfect)
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"


class TestBuildLoss:
    def test_build_all_loss_names(self):
        """build_loss phải tạo được tất cả các loại loss."""
        for name in ["focal_dice", "bce_dice", "ohem_dice"]:
            cfg = Config()
            cfg.loss_name = name
            loss_fn = build_loss(cfg)
            assert loss_fn is not None, f"Failed to build loss: {name}"

    def test_invalid_loss_name(self):
        """build_loss với tên không hợp lệ phải raise ValueError."""
        cfg = Config()
        cfg.loss_name = "invalid_loss"
        try:
            build_loss(cfg)
            assert False, "Expected ValueError for invalid loss name"
        except ValueError:
            pass
