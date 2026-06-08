# UNet-Lite Optimization Implementation Plan

> **For agentic workers:** Completed via inline execution.

**Goal:** Tối ưu hiệu suất UNet-Lite Human Segmentation — data pipeline, model architecture, testing, tooling.

**Architecture:** Giữ nguyên package `src/` hiện tại, bổ sung unit tests (20 tests), tối ưu Copy-Paste augmentation với PersonStampPool, thêm AttentionBottleneck tùy chọn (+286K params), TensorBoard logging, profiling tools, subset flag.

**Tech Stack:** Python 3.11, PyTorch 2.4, torchvision 0.19, pycocotools, tensorboard 2.20

---

## Summary of Changes

### Files Created
| File | Purpose |
|---|---|
| `tests/__init__.py` | Test package |
| `tests/test_model.py` | 7 tests: forward shape, gradient flow, param count, features, QAT stubs |
| `tests/test_losses.py` | 7 tests: BCE, Focal, OHEM, build_loss |
| `tests/test_metrics.py` | 6 tests: IoU, Dice, thresholds |
| `run_tests.py` | Custom test runner (no pytest dependency) |
| `src/blocks.py` | Refactored blocks: DoubleConvBlock, InvertedResidual, MV2DoubleBlock, AttentionBottleneck, LinearSelfAttention |
| `src/profiler.py` | DataLoader throughput & model forward benchmark |

### Files Modified
| File | Changes |
|---|---|
| `src/model.py` | Import blocks from `src.blocks`; add `use_attention` option |
| `src/config.py` | Add `use_attention` field |
| `src/dataset.py` | Add `PersonStampPool` class; update Copy-Paste to use pool; add subset support |
| `src/engine.py` | Add TensorBoard `SummaryWriter` logging |
| `train.py` | Add `--profile`, `--subset` flags |

### Key Metrics
- **Unit tests:** 20/20 passing (model, losses, metrics)
- **Attention model:** 1,934,280 params (+286K vs base)
- **Base model:** 1,647,816 params (unchanged)

## Implementation Log

1. **Task 1:** Unit tests for model, losses, metrics → 20 tests, all pass
2. **Task 2:** Refactor blocks into `src/blocks.py` + add AttentionBottleneck
3. **Task 3:** Profiling tools (DataLoader throughput, model forward benchmark)
4. **Task 4:** TensorBoard logging in training loop
5. **Task 5:** PersonStampPool for faster Copy-Paste augmentation
6. **Task 6:** `--subset` flag for debug training
7. **Bug fix:** Metrics tests passed sigmoid output instead of raw logits
