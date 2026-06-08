# UNet-Lite — Human Segmentation (COCO 2017)

Phân vùng người (person/nền) huấn luyện **from scratch** bằng UNet-Lite (khối MobileNetV2),
tối ưu cho GPU phổ thông (RTX 4050 6 GB): Mixed Precision + Gradient Accumulation +
Cosine LR + Early Stopping. **≈ 1,65M tham số.**

📄 Báo cáo chi tiết: [BAO_CAO.md](BAO_CAO.md)

## Cấu trúc

```
src/            package chính (config, model, dataset, losses, metrics, engine, visualize)
train.py        CLI huấn luyện
infer.py        CLI suy luận 1 ảnh → overlay mask
unet_base.ipynb notebook điều phối mỏng (chỉ gọi API src/)
cv_nets/        thư viện lớp nền (Conv2d, Norm, Activation)
```

## Chạy

> Chạy từ **thư mục gốc dự án** (để import `cv_nets`), môi trường conda `vision_env`.

```bash
python train.py                                    # train cấu hình mặc định
python train.py --epochs 50 --loss ohem_dice       # tùy chỉnh
python train.py --resume ./models/unet_base/last_unet_base.pth
python infer.py --image anh.jpg --out ket_qua.png  # suy luận
```

Mọi siêu tham số nằm trong `src/config.py` (`Config` dataclass).
