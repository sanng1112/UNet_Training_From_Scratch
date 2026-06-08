"""Cấu hình tập trung cho pipeline Human Segmentation (UNet-Lite).

Mọi siêu tham số được gom về một `dataclass` duy nhất để dễ tái lập thí nghiệm.
Giá trị mặc định bám sát báo cáo seminar (COCO 2017, 320x320, RTX 4050 - 6GB).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple

import torch


@dataclass
class Config:
    # ----- Dữ liệu -----
    train_img_dir: str = "./data/COCO/train2017"
    train_ann_file: str = "./data/COCO/annotations/instances_train2017.json"
    val_img_dir: str = "./data/COCO/val2017"
    val_ann_file: str = "./data/COCO/annotations/instances_val2017.json"
    image_size: Tuple[int, int] = (320, 320)
    copy_paste_prob: float = 0.5

    # ----- DataLoader -----
    batch_size: int = 16
    num_workers: int = 16
    prefetch_factor: int = 2

    # ----- Mô hình -----
    num_classes: int = 1                       # nhị phân: person / nền (1 logit)
    features: Tuple[int, ...] = (8, 16, 32, 64, 128)
    expand_ratio: int = 2

    # ----- Huấn luyện -----
    epochs: int = 30
    lr: float = 1e-3
    eta_min: float = 1e-6                       # CosineAnnealingLR
    accumulation_steps: int = 16               # batch hiệu dụng = 16 x 16 = 256
    amp: bool = True                           # Mixed Precision (fp16)
    channels_last: bool = True
    cudnn_benchmark: bool = True

    # ----- Hàm mất mát -----
    # 'focal_dice' | 'bce_dice' | 'ohem_dice'
    loss_name: str = "focal_dice"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    bce_weight: float = 0.5
    ohem_top_k: float = 0.25

    # ----- Early stopping / checkpoint -----
    patience: int = 10                         # số epoch chịu đựng val_loss không giảm
    min_epochs_before_stop: int = 15           # chỉ xét early stopping sau mốc này
    save_dir: str = "./models/unet_base"
    ckpt_prefix: str = "unet_base"

    # ----- Kiến trúc -----
    use_attention: bool = False               # nếu True, dùng AttentionBottleneck thay DoubleConvBlock

    # ----- Khác -----
    seed: int = 42
    threshold: float = 0.5                      # ngưỡng nhị phân hoá khi tính metric

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def best_ckpt(self) -> str:
        import os
        return os.path.join(self.save_dir, f"best_{self.ckpt_prefix}.pth")

    @property
    def last_ckpt(self) -> str:
        import os
        return os.path.join(self.save_dir, f"last_{self.ckpt_prefix}.pth")

    @property
    def history_file(self) -> str:
        import os
        return os.path.join(self.save_dir, "training_history.json")

    def to_dict(self) -> dict:
        return asdict(self)
