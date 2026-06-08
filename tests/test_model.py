"""Unit tests cho UNet-Lite model — dùng unittest (không phụ thuộc pytest)."""
from __future__ import annotations

import torch

from src.model import UNetLite, build_model, count_parameters
from src.config import Config


class DummyOpts:
    """Giả lập opts cho cv_nets layers."""
    pass


class TestUNetLite:
    """Test UNetLite model forward/backward/params."""

    def __init__(self):
        self.model = UNetLite(opts=DummyOpts(), num_classes=1)
        self.sample_input = torch.randn(2, 3, 320, 320)

    def test_forward_shape(self):
        """Đầu ra phải có shape [B, 1, H, W]."""
        output = self.model(self.sample_input)
        expected = (2, 1, 320, 320)
        assert output.shape == expected, f"Expected {expected}, got {output.shape}"

    def test_output_range_before_sigmoid(self):
        """Logit đầu ra là số thực, không nằm gọn trong [0,1]."""
        output = self.model(self.sample_input)
        assert not (output < 0).all(), "Logits should not be all negative"
        assert not (output > 1).all(), "Logits should not be all in [0,1]"

    def test_gradient_flow(self):
        """Loss backward phải tạo gradient cho tất cả tham số."""
        output = self.model(self.sample_input)
        loss = output.sum()
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients flowing"
        has_nonzero = False
        for g in grads:
            if (g != 0).any():
                has_nonzero = True
                break
        assert has_nonzero, "All gradients are zero"

    def test_count_parameters(self):
        """count_parameters trả về số nguyên dương."""
        n = count_parameters(self.model)
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert n > 0, f"Expected positive, got {n}"
        assert n < 10_000_000, f"Too many parameters: {n}"

    def test_build_model_from_config(self):
        """build_model với Config phải tạo model thành công."""
        cfg = Config()
        m = build_model(cfg)
        assert isinstance(m, UNetLite), f"Expected UNetLite, got {type(m)}"

    def test_different_features(self):
        """Model hoạt động với features tuples khác nhau."""
        m = UNetLite(opts=DummyOpts(), features=(16, 32, 64, 128))
        x = torch.randn(1, 3, 320, 320)
        out = m(x)
        assert out.shape == (1, 1, 320, 320), f"Got {out.shape}"

    def test_quant_stubs_present(self):
        """Model phải có QuantStub và DeQuantStub cho QAT sau này."""
        assert hasattr(self.model, 'quant'), "Missing quant"
        assert hasattr(self.model, 'dequant'), "Missing dequant"
        assert hasattr(self.model, 'f_cat'), "Missing f_cat"
