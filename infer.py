"""Suy luận UNet-Lite trên một ảnh đơn lẻ và xuất overlay mask.

Ví dụ:
    python infer.py --image path/to/photo.jpg --ckpt ./models/unet_base/best_unet_base.pth --out out.png
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src import Config, build_model, load_checkpoint
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


def preprocess(image: Image.Image, size):
    short_edge = min(size)
    img = TF.resize(image, short_edge, interpolation=InterpolationMode.BILINEAR)
    img = TF.center_crop(img, size)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0), np.array(img) / 255.0


def overlay(img_np, mask, color=(1.0, 0.0, 0.0), alpha=0.5):
    out = img_np.copy()
    for c in range(3):
        out[:, :, c] = np.where(mask > 0, img_np[:, :, c] * (1 - alpha) + alpha * color[c], img_np[:, :, c])
    return np.clip(out, 0, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Infer UNet-Lite Human Segmentation")
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", default=None, help="mặc định: best checkpoint trong Config")
    p.add_argument("--out", default="prediction.png")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    device = cfg.device

    model = build_model(cfg).to(device).eval()
    ckpt_path = args.ckpt or cfg.best_ckpt
    load_checkpoint(ckpt_path, model, map_location=device)
    print(f"Đã nạp checkpoint: {ckpt_path}")

    image = Image.open(args.image).convert("RGB")
    tensor, img_np = preprocess(image, cfg.image_size)
    tensor = tensor.to(device)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
        logits = model(tensor)
    mask = (torch.sigmoid(logits)[0, 0] > args.threshold).cpu().numpy()

    result = (overlay(img_np, mask) * 255).astype(np.uint8)
    Image.fromarray(result).save(args.out)
    print(f"Đã lưu kết quả: {args.out}  (tỉ lệ pixel person: {mask.mean():.2%})")


if __name__ == "__main__":
    main()
