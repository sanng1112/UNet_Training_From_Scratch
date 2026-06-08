# Báo cáo Seminar — UNet-Lite cho bài toán Phân vùng Người (Human Segmentation)

**Tập dữ liệu:** COCO 2017 (lớp `person`) · **Phần cứng:** NVIDIA RTX 4050 Laptop (6 GB) · **Khung:** PyTorch 2.4, dựa trên thư viện `cv_nets`.

---

## 1. Tóm tắt

Dự án xây dựng và huấn luyện **từ đầu (from scratch)** một mạng phân vùng ngữ nghĩa nhị phân
(person / nền) trên ảnh đời thực COCO 2017. Mô hình **UNet-Lite** kế thừa triết lý U-Net
(encoder–decoder + skip-connection) nhưng thay các khối tích chập nặng bằng khối
**Inverted Residual** của MobileNetV2, nhờ đó chỉ còn **≈ 1,65 triệu tham số** — đủ nhẹ để
huấn luyện trọn vẹn trên GPU 6 GB.

Toàn bộ chương trình đã được **refactor** từ một notebook đơn khối (monolithic) thành một
package Python (`src/`) có cấu trúc rõ ràng, kèm hai entry-point dòng lệnh (`train.py`,
`infer.py`) và một notebook "mỏng" chỉ gọi API. Quá trình refactor đồng thời **sửa ba lỗi
thực sự** trong vòng lặp huấn luyện gốc.

---

## 2. Bài toán & dữ liệu

- **Mục tiêu:** với mỗi pixel của ảnh, dự đoán xác suất pixel đó thuộc về người. Đầu ra là
  một bản đồ logit 1 kênh, nhị phân hoá ở ngưỡng 0.5.
- **Dữ liệu:** COCO 2017, chỉ giữ những ảnh có chứa annotation `person`
  (≈ 64k ảnh train / ≈ 2,7k ảnh val). Mask ground-truth là **hợp (union)** của tất cả
  instance person trong ảnh — bài toán semantic, không phân biệt từng cá thể.
- **Tiền xử lý:** resize cạnh ngắn → cắt về `320×320`, chuẩn hoá theo thống kê ImageNet
  (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`).
- **Thách thức chính:** mất cân bằng lớp nghiêm trọng (vùng người thường nhỏ so với nền),
  và sự đa dạng về tư thế/kích thước/che khuất.

### Tăng cường dữ liệu (chỉ áp dụng cho tập train)

| Phép biến đổi | Mục đích |
|---|---|
| Resize cạnh ngắn + `RandomResizedCrop` (scale 0.5–1.0) | bất biến tỉ lệ & vị trí |
| Horizontal flip (p=0.5) | bất biến trái–phải |
| **Copy-Paste** (p=0.5) | dán người từ ảnh khác → tăng mật độ đối tượng dương |
| `ColorJitter` (p=0.5) | bất biến ánh sáng/màu |
| `GaussianBlur` (p=0.3) | bất biến độ nét |

Tập val dùng tiền xử lý **xác định** (resize + center-crop) để đánh giá ổn định.

---

## 3. Kiến trúc UNet-Lite

```
Input 3×320×320
   │
 [Encoder] 5 tầng — mỗi tầng: MV2DoubleBlock → downsample (Conv stride 2)
   │  features = (8, 16, 32, 64, 128)        (skip-connection lưu lại mỗi tầng)
   ▼
 [Bottleneck] DoubleConvBlock  128 → 256
   │
 [Decoder] 5 tầng — mỗi tầng: ConvTranspose2d ×2 → cat(skip) → MV2DoubleBlock
   ▼
 final 1×1 Conv → 1 logit/pixel  →  1×320×320
```

**Các khối thành phần** (`src/model.py`):

- **`InvertedResidual`** — khối cơ sở MobileNetV2: pointwise *expand* (×`expand_ratio`) →
  depthwise 3×3 → pointwise *project* (tuyến tính), có residual khi `stride=1` và số kênh
  vào = ra. Giảm mạnh tham số/FLOPs so với Conv 3×3 thường.
- **`MV2DoubleBlock`** — hai `InvertedResidual` nối tiếp, dùng ở mọi tầng encoder/decoder.
- **`DoubleConvBlock`** — hai (Conv3×3 → Norm → Act), chỉ dùng cho bottleneck.
- **Quant/DeQuant stubs + `FloatFunctional`** — giữ sẵn móc nối cho lượng tử hoá (QAT) về sau.

Các lớp `Conv2d`, `ConvTranspose2d`, normalization (BatchNorm), activation (LeakyReLU) đều lấy
từ `cv_nets` — do đó **phải chạy chương trình từ thư mục gốc dự án** để import được.

**Quy mô:** 1.647.816 tham số học được (đã xác minh bằng `count_parameters`).

---

## 4. Hàm mất mát & độ đo

Để xử lý mất cân bằng lớp, dự án cung cấp 3 hàm loss (chọn qua `Config.loss_name`), tất cả
đều là **tổ hợp một thành phần pixel-wise + Dice**:

| Loss | Thành phần pixel | Khi nào dùng |
|---|---|---|
| `bce_dice` | BCE chuẩn | baseline |
| **`focal_dice`** *(mặc định)* | Focal (α=0.25, γ=2) — hạ trọng số pixel dễ | nền áp đảo |
| `ohem_dice` | OHEM — chỉ phạt top-25% pixel khó nhất | người rất nhỏ |

Thành phần **Dice** (`1 − Dice`) tối ưu trực tiếp độ chồng lấp hình dạng, bù cho điểm yếu của
loss pixel-wise khi đối tượng nhỏ. Trọng số trộn mặc định `bce_weight = 0.5`.

**Độ đo đánh giá** (`src/metrics.py`): **IoU** (Jaccard) và **Dice (F1)** trung bình trên batch,
nhị phân hoá ở ngưỡng 0.5.

---

## 5. Quy trình huấn luyện (`src/engine.py`)

| Kỹ thuật | Cấu hình | Lý do |
|---|---|---|
| Optimizer | Adam, `lr=1e-3` | hội tụ nhanh, ổn định |
| LR scheduler | `CosineAnnealingLR`, `eta_min=1e-6` | giảm LR mượt → hội tụ tốt hơn |
| **Mixed Precision (AMP fp16)** | `torch.autocast` + `GradScaler` | tiết kiệm VRAM, tăng tốc trên GPU 6 GB |
| `channels_last` | bật | tối ưu kernel tích chập |
| **Gradient Accumulation** | batch 16 × 16 bước = **batch hiệu dụng 256** | mô phỏng batch lớn trên VRAM nhỏ |
| **Early Stopping** | `patience=10`, chỉ xét sau epoch 15 | tránh overfit, tiết kiệm thời gian |
| Checkpoint | lưu `best_` & `last_` + `history`/`config` | tái lập & train tiếp (`--resume`) |
| Tái lập | cố định seed (NumPy/torch/cuda) | thí nghiệm lặp lại được |

---

## 6. Refactor: từ notebook đơn khối → package có cấu trúc

### 6.1. Trước (notebook gốc `unetlite.ipynb`)

- ~24 ô gộp toàn bộ: định nghĩa model, dataset, loss, vòng lặp train, vẽ hình.
- Siêu tham số là **biến toàn cục** rải rác (`BATCH_SIZE`, `EPOCH=150`, …).
- **Đường dẫn hard-code** theo Kaggle (`/kaggle/input/...`).
- Logic resume checkpoint viết tay, lặp lại; khó tái sử dụng/khó test.

### 6.2. Sau (package `src/`)

```
src/
├── config.py     # Config dataclass — gom toàn bộ siêu tham số
├── model.py      # UNetLite + InvertedResidual + build_model + count_parameters
├── dataset.py    # COCOPersonDataset + augmentation + build_dataloaders
├── losses.py     # BCEDice / FocalDice / OHEMDice + build_loss
├── metrics.py    # calculate_iou / calculate_dice
├── engine.py     # fit / train_one_epoch / evaluate / save|load_checkpoint
└── visualize.py  # show_batch / show_predictions / plot_history
train.py          # CLI huấn luyện (argparse, có --resume)
infer.py          # CLI suy luận 1 ảnh → overlay mask
unet_base.ipynb   # notebook "mỏng": chỉ gọi API của src/
```

**Lợi ích:** mỗi module một trách nhiệm; siêu tham số tập trung trong một `dataclass`
(dễ tái lập); chạy được cả từ CLI lẫn notebook; tách bạch logic → dễ kiểm thử và mở rộng.

### 6.3. Ba lỗi đã sửa trong quá trình refactor

1. **Bỏ sót cửa sổ gradient-accumulation cuối:** khi số batch không chia hết cho
   `accumulation_steps`, phần dư trước đây không bao giờ được `optimizer.step()` → mất gradient.
   *Sửa:* luôn step phần dư ở cuối epoch.
2. **Early Stopping không bao giờ kích hoạt:** `min_epochs_before_stop` bị đặt **bằng** số epoch
   tổng → điều kiện không bao giờ thỏa. *Sửa:* tách thành tham số độc lập (mặc định 15).
3. **API AMP lỗi thời:** thay `torch.cuda.amp.GradScaler` bằng `torch.amp.GradScaler("cuda")`
   / `torch.autocast` (chuẩn torch ≥ 2.x).

*(Bổ sung lần này)* **`load_checkpoint` an toàn tương lai:** truyền `weights_only=False` tường
minh để vẫn nạp được checkpoint chứa `config`/`history` trên torch ≥ 2.6 (nơi mặc định đổi sang
`True`).

---

## 7. Cách chạy

> Luôn chạy từ **thư mục gốc dự án** (để import `cv_nets`) và trong môi trường `vision_env`.

```bash
# Huấn luyện (cấu hình mặc định trong src/config.py)
python train.py
python train.py --epochs 50 --loss ohem_dice          # đổi loss / số epoch
python train.py --resume ./models/unet_base/last_unet_base.pth   # train tiếp

# Suy luận 1 ảnh → xuất overlay
python infer.py --image anh.jpg --ckpt ./models/unet_base/best_unet_base.pth --out ket_qua.png

# Hoặc dùng notebook điều phối mỏng
jupyter notebook unet_base.ipynb
```

---

## 8. Kết quả & hướng phát triển

- Pipeline đã **xác minh chạy đúng**: forward `(2,3,320,320) → (2,1,320,320)`, ba hàm loss và
  hai độ đo hoạt động; checkpoint cũ nạp lại tương thích state_dict.
- Checkpoint hiện có mới ở giai đoạn rất sớm (epoch ~2): `val_IoU ≈ 0.37`, `val_Dice ≈ 0.49` —
  cần huấn luyện đủ số epoch để đánh giá đầy đủ.

**Hướng phát triển:**
- Huấn luyện trọn 30+ epoch và báo cáo đường cong Loss/IoU/Dice (`viz.plot_history`).
- Thử `ohem_dice` cho ảnh nhiều người nhỏ; quét ngưỡng nhị phân hoá trên tập val.
- Lượng tử hoá (QAT — đã có sẵn Quant/DeQuant stubs) để triển khai trên thiết bị biên.
- Bổ sung test đơn vị cho `losses`/`metrics` và đo throughput dữ liệu.
