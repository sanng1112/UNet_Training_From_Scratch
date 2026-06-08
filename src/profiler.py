"""Công cụ profiling cho training pipeline UNet-Lite."""
from __future__ import annotations

import time
from typing import Dict

import torch


@torch.no_grad()
def benchmark_dataloader(loader, num_batches: int = 50) -> Dict[str, float]:
    """Đo throughput của DataLoader (images/second)."""
    times = []
    total_images = 0
    loader_iter = iter(loader)
    for _ in range(num_batches):
        start = time.perf_counter()
        try:
            images, masks = next(loader_iter)
        except StopIteration:
            break
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_images += images.size(0)

    if not times:
        return {"avg_batch_time_ms": 0, "images_per_sec": 0, "avg_batch_size": 0, "num_batches": 0}

    avg_time = sum(times) / len(times)
    avg_batch_size = total_images / len(times)
    return {
        "avg_batch_time_ms": avg_time * 1000,
        "images_per_sec": total_images / sum(times),
        "avg_batch_size": avg_batch_size,
        "num_batches": len(times),
    }


@torch.no_grad()
def benchmark_model_forward(model, input_shape: tuple,
                            num_warmup: int = 10, num_iter: int = 50) -> Dict[str, float]:
    """Đo thời gian forward của model (ms) và FPS."""
    device = next(model.parameters()).device
    x = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(num_warmup):
        _ = model(x)

    # Measure
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(num_iter):
        start = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = sum(times) / len(times) * 1000
    return {
        "avg_forward_ms": avg_ms,
        "fps": 1000 / avg_ms,
        "num_iter": num_iter,
    }
