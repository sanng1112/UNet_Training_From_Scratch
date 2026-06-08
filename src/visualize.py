"""Trực quan hoá: dữ liệu mẫu, đường cong huấn luyện, dự đoán & overlay màu."""
from __future__ import annotations

import json
import math
import os
from typing import Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from .dataset import denormalize


def _grid(num_samples: int, cols_per_sample: int):
    """Tạo lưới subplot: mỗi sample chiếm `cols_per_sample` cột, 2 sample/hàng."""
    samples_per_row = 2
    nrows = math.ceil(num_samples / samples_per_row)
    ncols = samples_per_row * cols_per_sample
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 3.5 * nrows), squeeze=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    return fig, axes, samples_per_row


def _to_img(image_tensor: torch.Tensor) -> np.ndarray:
    """Tensor CHW đã chuẩn hoá -> ảnh HWC numpy [0,1]."""
    return denormalize(image_tensor.cpu()).permute(1, 2, 0).numpy()


def apply_color_mask(image, mask, color=(0.0, 1.0, 0.0), alpha=0.5):
    """Phủ lớp màu bán trong suốt lên vùng mask > 0."""
    out = image.copy()
    for c in range(3):
        out[:, :, c] = np.where(mask > 0, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])
    return np.clip(out, 0, 1)


def show_batch(dataloader, num_samples: int = 8):
    """Hiển thị ảnh gốc + ground-truth từ dataloader (kiểm tra dữ liệu)."""
    fig, axes, spr = _grid(num_samples, cols_per_sample=2)
    shown = 0
    for images, masks in dataloader:
        for i in range(images.size(0)):
            if shown >= num_samples:
                break
            row, col = shown // spr, (shown % spr) * 2
            axes[row, col].imshow(_to_img(images[i])); axes[row, col].set_title(f"Sample {shown+1}"); axes[row, col].axis("off")
            axes[row, col + 1].imshow(masks[i].squeeze().numpy(), cmap="gray"); axes[row, col + 1].set_title("Ground Truth"); axes[row, col + 1].axis("off")
            shown += 1
        if shown >= num_samples:
            break
    plt.suptitle("Dữ liệu mẫu (Input | Ground Truth)", fontsize=16)
    plt.show()


@torch.no_grad()
def show_predictions(model, dataloader, cfg, num_samples: int = 8, colored: bool = False):
    """Hiển thị Input | Ground Truth | Prediction.

    colored=False: mask xám. colored=True: overlay GT(xanh) / Pred(đỏ) lên ảnh.
    """
    device = cfg.device
    model.eval()
    fig, axes, spr = _grid(num_samples, cols_per_sample=3)
    shown = 0
    for images, masks in dataloader:
        imgs = images.to(device)
        if cfg.channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            outputs = model(imgs)
        preds = (torch.sigmoid(outputs) > cfg.threshold).cpu()

        for i in range(images.size(0)):
            if shown >= num_samples:
                break
            row, col = shown // spr, (shown % spr) * 3
            img = _to_img(images[i])
            gt = masks[i].squeeze().numpy()
            pr = preds[i].squeeze().numpy()

            axes[row, col].imshow(img); axes[row, col].set_title(f"Sample {shown+1}"); axes[row, col].axis("off")
            if colored:
                axes[row, col + 1].imshow(apply_color_mask(img, gt, (0, 1, 0))); axes[row, col + 1].set_title("Truth (Green)")
                axes[row, col + 2].imshow(apply_color_mask(img, pr, (1, 0, 0))); axes[row, col + 2].set_title("Pred (Red)")
            else:
                axes[row, col + 1].imshow(gt, cmap="gray"); axes[row, col + 1].set_title("Ground Truth")
                axes[row, col + 2].imshow(pr, cmap="gray"); axes[row, col + 2].set_title("Prediction")
            axes[row, col + 1].axis("off"); axes[row, col + 2].axis("off")
            shown += 1
        if shown >= num_samples:
            break
    plt.suptitle("Đánh giá Segmentation" + (" (Overlay màu)" if colored else " (Input | GT | Pred)"), fontsize=16)
    plt.show()


def plot_history(history_source: Union[str, dict]):
    """Vẽ Loss / Learning Rate / IoU / Dice từ dict history hoặc file JSON."""
    if isinstance(history_source, str) and os.path.exists(history_source):
        with open(history_source) as f:
            history = json.load(f)
    elif isinstance(history_source, dict):
        history = history_source
    else:
        raise ValueError("history_source phải là dict hoặc đường dẫn file JSON tồn tại.")

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history.get("train_loss", []), "b-", label="Train Loss", linewidth=2)
    if history.get("val_loss"):
        plt.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, ls="--", alpha=0.7)

    if history.get("lr"):
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history["lr"], "k-", linewidth=2)
        plt.title("Learning Rate"); plt.xlabel("Epoch"); plt.yscale("log"); plt.grid(True, ls="--", alpha=0.7)

    if history.get("val_iou"):
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history["val_iou"], "g-", linewidth=2)
        plt.title("Validation IoU"); plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.grid(True, ls="--", alpha=0.7)

    if history.get("val_dice"):
        plt.subplot(2, 2, 4)
        plt.plot(epochs, history["val_dice"], "m-", linewidth=2)
        plt.title("Validation Dice"); plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.grid(True, ls="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
