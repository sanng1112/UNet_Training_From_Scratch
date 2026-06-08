# UNet-Lite Optimization Implementation Plan

> **For agentic workers:** COMPLETED — All tasks have been implemented and verified.

**Goal:** Tối ưu hiệu suất UNet-Lite Human Segmentation — thêm unit tests, cải tiến model architecture (AttentionBottleneck), tối ưu data pipeline (PersonStampPool), profiling tools, TensorBoard logging.

**Architecture:** 
- **blocks.py**: Tách building block khỏi model.py — DoubleConvBlock, InvertedResidual, MV2DoubleBlock, AttentionBottleneck, LinearSelfAttention
- **dataset.py**: Thêm PersonStampPool để Copy-Paste không cần load I/O + subset support
- **model.py**: Refactor import blocks + use_attention flag
- **engine.py**: TensorBoard logging; fix duplicate imports
- **train.py**: --profile, --subset, --use-attention CLI flags
- **tests/**: 20 unit tests không phụ thuộc pytest

**Tech Stack:** Python 3.11, PyTorch 2.4, torchvision 0.19, pycocotools, tensorboard 2.20

---

## Task Completion

### Task 1: Unit Tests ✅
**Files:** `tests/__init__.py`, `tests/test_model.py`, `tests/test_losses.py`, `tests/test_metrics.py`, `run_tests.py`
- 7 tests cho model (forward shape, gradient flow, param count, features, QAT stubs)
- 7 tests cho losses (BCE, Focal, OHEM positive, build_loss all names, invalid name error)
- 6 tests cho metrics (IoU perfect/no match/half overlap/threshold, Dice perfect/Dice≥IoU)
- Custom test runner không phụ thuộc pytest

### Task 2: Blocks Module + AttentionBottleneck ✅
**Files:** `src/blocks.py` (mới), `src/model.py` (sửa)
- DoubleConvBlock: 2× (Conv3×3→BN→LeakyReLU)
- InvertedResidual: expand→depthwise→project, residual khi stride=1 & in_ch=out_ch
- MV2DoubleBlock: 2× InvertedResidual nối tiếp
- LinearSelfAttention: patch unfold→scaled dot-product→fold, xử lý input không chia hết
- AttentionBottleneck: MV2DoubleBlock→LSA→MV2DoubleBlock
- Base: 1,647,816 params | Attention: 1,934,280 params (+17%)

### Task 3: PersonStampPool ✅
**Files:** `src/dataset.py` (sửa)
- Precompute 500 person stamps từ COCO, filter stamp quá nhỏ (min_person_ratio=0.005)
- __getitem__: ưu tiên pool, fallback load ảnh B cũ
- build_dataloaders: tự động tạo pool cho train dataset

### Task 4: Profiling Tools ✅
**Files:** `src/profiler.py` (mới), `train.py` (sửa)
- benchmark_dataloader: avg_batch_time_ms, images_per_sec
- benchmark_model_forward: avg_forward_ms, fps
- train.py --profile: chạy benchmark và in kết quả, không train

### Task 5: TensorBoard Logging ✅
**Files:** `src/engine.py` (sửa)
- SummaryWriter log Loss/train, Loss/val, Metrics/IoU, Metrics/Dice, LR
- writer.close() sau training

### Task 6: CLI Improvements ✅
**Files:** `train.py` (sửa), `src/dataset.py` (sửa)
- --subset N: train/eval trên N ảnh đầu (Subset dataset)
- --use-attention: bật AttentionBottleneck
- --profile: benchmark mode

### Bug Fixes ✅
- engine.py: xóa 6 dòng import duplicate
- config.py: section headers đúng thứ tự
- test_metrics.py: fix double sigmoid (truyền raw logits thay vì sigmoid output)

## Final File Structure
```
src/
├── __init__.py     # Public API
├── blocks.py       # Building blocks (MỚI)
├── config.py       # Config dataclass
├── dataset.py      # PersonStampPool + COCOPersonDataset (SỬA)
├── engine.py       # fit/train/evaluate/checkpoint + TensorBoard (SỬA)
├── losses.py       # BCEDice/FocalDice/OHEMDice
├── metrics.py      # calculate_iou/calculate_dice
├── model.py        # UNetLite + build_model (SỬA)
├── profiler.py     # benchmark functions (MỚI)
└── visualize.py    # plotting + overlay
tests/
├── __init__.py     # (MỚI)
├── test_model.py   # 7 tests (MỚI)
├── test_losses.py  # 7 tests (MỚI)
└── test_metrics.py # 6 tests (MỚI)
run_tests.py        # Test runner (MỚI)
train.py            # CLI entry (SỬA)
infer.py            # Inference
```

## Verification Results
| Check | Result |
|---|---|
| Unit tests | 20/20 ✅ |
| Blocks verification | 10/10 ✅ |
| Base model forward | 1,647,816 params ✅ |
| Attention model forward | 1,934,280 params ✅ |
| Checkpoint round-trip | epoch/history/config ✅ |
| DataLoader init | subset=5 works ✅ |
| CLI flags | --use-attention, --subset, --profile ✅ |
| Bug fixes | 3 bugs fixed ✅ |

