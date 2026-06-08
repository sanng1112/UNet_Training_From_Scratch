#!/usr/bin/env python3
"""Test runner cho UNet-Lite — không phụ thuộc pytest."""
from __future__ import annotations

import importlib
import sys
import time
from typing import List, Tuple


def run_tests_in_module(module_name: str) -> Tuple[int, int, float]:
    """Chạy tất cả test classes trong module, trả về (pass, fail, time_sec)."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"  FAIL: Không thể import {module_name}: {e}")
        return 0, 1, 0

    passed = 0
    failed = 0
    start = time.time()

    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if not isinstance(obj, type):
            continue
        if not attr_name.startswith("Test"):
            continue

        # Khởi tạo test instance
        try:
            instance = obj()
        except Exception as e:
            print(f"  FAIL: {module_name}.{attr_name} không thể khởi tạo: {e}")
            failed += 1
            continue

        # Chạy từng method test_*
        for method_name in sorted(dir(obj)):
            if not method_name.startswith("test_"):
                continue
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✓ {module_name}.{attr_name}.{method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {module_name}.{attr_name}.{method_name}: {e}")
                failed += 1

    elapsed = time.time() - start
    return passed, failed, elapsed


def main():
    modules = [
        "tests.test_model",
        "tests.test_losses",
        "tests.test_metrics",
    ]

    total_passed = 0
    total_failed = 0
    total_start = time.time()

    print(f"{'='*60}")
    print(f"  UNet-Lite Test Suite")
    print(f"{'='*60}\n")

    for mod_name in modules:
        print(f"[{mod_name}]")
        p, f, t = run_tests_in_module(mod_name)
        total_passed += p
        total_failed += f
        status = "✅" if f == 0 else "❌"
        print(f"  → {p} passed, {f} failed in {t:.2f}s {status}\n")

    total_elapsed = time.time() - total_start
    print(f"{'='*60}")
    print(f"  Total: {total_passed} passed, {total_failed} failed in {total_elapsed:.2f}s")
    print(f"{'='*60}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
