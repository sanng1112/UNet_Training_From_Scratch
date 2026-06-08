"""Vòng lặp huấn luyện/đánh giá cho UNet-Lite.

Tích hợp: Mixed Precision (AMP fp16), channels_last, Gradient Accumulation,
CosineAnnealingLR, Early Stopping và lưu/khôi phục checkpoint.

So với notebook gốc, các lỗi sau đã được sửa:
  * Cửa sổ gradient-accumulation cuối (khi số batch không chia hết) trước đây bị
    bỏ qua -> nay luôn step phần dư.
  * `min_epochs_before_stop == epochs` khiến early stopping không bao giờ chạy
    -> tách thành tham số độc lập, mặc định 15.
  * Dùng `torch.amp.GradScaler('cuda')` / `torch.autocast` (API mới, torch>=2.x).
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import calculate_iou, calculate_dice


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _empty_history() -> Dict[str, list]:
    return {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "lr": []}


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, epoch: int) -> float:
    """Chạy một epoch huấn luyện, trả về train loss trung bình."""
    device = cfg.device
    model.train()
    running = 0.0
    accum = cfg.accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]", unit="batch")
    for i, (images, masks) in enumerate(bar):
        images = images.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.to(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss / accum).backward()

        if (i + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    # Step phần dư nếu số batch không chia hết cho accumulation_steps.
    if len(loader) % accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, cfg) -> Dict[str, float]:
    """Đánh giá trên val set, trả về dict {loss, iou, dice}."""
    device = cfg.device
    model.eval()
    val_loss = val_iou = val_dice = 0.0

    bar = tqdm(loader, desc="[Val]", unit="batch", leave=False)
    for images, masks in bar:
        images = images.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.to(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        val_loss += loss.item()
        val_iou += calculate_iou(outputs, masks, cfg.threshold)
        val_dice += calculate_dice(outputs, masks, cfg.threshold)

    n = max(1, len(loader))
    return {"loss": val_loss / n, "iou": val_iou / n, "dice": val_dice / n}


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, epoch: int,
                    metrics: Dict[str, float], history: Dict[str, list], cfg) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": metrics["loss"],
        "val_iou": metrics["iou"],
        "val_dice": metrics["dice"],
        "history": history,
        "config": cfg.to_dict(),
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu") -> dict:
    """Nạp checkpoint. Trả về dict checkpoint (gồm 'history', 'epoch', ...)."""
    # weights_only=False vì checkpoint chứa cả 'config'/'history' (object Python),
    # không chỉ tensor. Cần thiết cho torch >= 2.6 (mặc định đã đổi sang True).
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def fit(model, train_loader, val_loader, criterion, cfg, resume: Optional[str] = None) -> Dict[str, list]:
    """Huấn luyện đầy đủ với early stopping & checkpoint. Trả về history."""
    device = cfg.device
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    history = _empty_history()
    best_val_loss = float("inf")
    start_epoch = 1
    early_stop_counter = 0

    if resume and os.path.exists(resume):
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, scaler, map_location=device)
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
        print(f"==> Resume từ epoch {start_epoch} (best val_loss={best_val_loss:.4f})")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # TensorBoard
    tb_log_dir = os.path.join(cfg.save_dir, "tb_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg, epoch)
        metrics = evaluate(model, val_loader, criterion, cfg)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_iou"].append(metrics["iou"])
        history["val_dice"].append(metrics["dice"])
        history["lr"].append(current_lr)

        print(f"-> Epoch {epoch} | Train {train_loss:.4f} | Val {metrics['loss']:.4f} "
              f"| IoU {metrics['iou']:.4f} | Dice {metrics['dice']:.4f} | LR {current_lr:.2e}")

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", metrics["loss"], epoch)
        writer.add_scalar("Metrics/IoU", metrics["iou"], epoch)
        writer.add_scalar("Metrics/Dice", metrics["dice"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        save_checkpoint(cfg.last_ckpt, model, optimizer, scheduler, scaler, epoch, metrics, history, cfg)

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            early_stop_counter = 0
            save_checkpoint(cfg.best_ckpt, model, optimizer, scheduler, scaler, epoch, metrics, history, cfg)
            print(f"   ✓ Lưu mô hình tốt nhất: {cfg.best_ckpt}")
        elif epoch >= cfg.min_epochs_before_stop:
            early_stop_counter += 1
            print(f"   [EarlyStopping] val_loss không giảm: {early_stop_counter}/{cfg.patience}")
            if early_stop_counter >= cfg.patience:
                print(f"==== EARLY STOPPING tại epoch {epoch} ====")
                break

        scheduler.step()

    writer.close()

    with open(cfg.history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"\nHoàn tất huấn luyện. History -> {cfg.history_file}")
    return history
