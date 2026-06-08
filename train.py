"""Entry point huấn luyện UNet-Lite cho Human Segmentation.

Ví dụ:
    python train.py                       # train với cấu hình mặc định (báo cáo)
    python train.py --epochs 50 --loss ohem_dice
    python train.py --resume ./models/unet_base/last_unet_base.pth
"""
from __future__ import annotations

import argparse

from src import Config, build_model, build_dataloaders, build_loss, fit
from src.model import count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UNet-Lite Human Segmentation")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--accumulation-steps", type=int, default=None)
    p.add_argument("--loss", dest="loss_name", type=str, default=None,
                   choices=["focal_dice", "bce_dice", "ohem_dice"])
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision")
    p.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để train tiếp")
    p.add_argument("--use-attention", action="store_true", help="Dùng AttentionBottleneck thay DoubleConvBlock")
    p.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint để train tiếp")
    p.add_argument("--profile", action="store_true", help="Chạy benchmark dataloader & model")
    p.add_argument("--subset", type=int, default=None, help="Dùng N ảnh đầu để debug nhanh")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    for key in ["epochs", "batch_size", "lr", "num_workers", "accumulation_steps", "loss_name", "save_dir"]:
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)
    if args.no_amp:
        cfg.amp = False
    if args.subset is not None:
        cfg.subset = args.subset
    if args.use_attention:
        cfg.use_attention = True
    if args.subset is not None:
        cfg.subset = args.subset
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    print(f"Thiết bị: {cfg.device}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} ảnh | Val: {len(val_loader.dataset)} ảnh")

    model = build_model(cfg)
    print(f"Tham số học được: {count_parameters(model):,}")

    # Profiling mode
    if args.profile:
        from src.profiler import benchmark_dataloader, benchmark_model_forward
        print("\n=== Profiling DataLoader ===")
        dl_stats = benchmark_dataloader(train_loader, num_batches=20)
        for k, v in dl_stats.items():
            print(f"  {k}: {v:.2f}")
        print("\n=== Profiling Model Forward ===")
        model = model.to(cfg.device)
        model_stats = benchmark_model_forward(
            model, (cfg.batch_size, 3, cfg.image_size[0], cfg.image_size[1])
        )
        for k, v in model_stats.items():
            print(f"  {k}: {v:.2f}")
        print("=== Profiling complete ===\n")
        return

    criterion = build_loss(cfg)
    fit(model, train_loader, val_loader, criterion, cfg, resume=args.resume)


if __name__ == "__main__":
    main()
