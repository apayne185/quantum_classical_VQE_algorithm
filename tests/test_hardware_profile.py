"""Smoke test for the HardwareProfile auto-detection layer.

Run standalone (no MPI needed):
    python3 tests/test_hardware_profile.py

Also exercises env-var overrides to confirm researcher hooks work.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.hardware import HardwareProfile


def _banner(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_autodetect():
    _banner("Auto-detect (no overrides)")
    for k in ("VQE_PRECISION", "VQE_BACKEND", "USE_GPU"):
        os.environ.pop(k, None)
    hw = HardwareProfile.detect()
    print(hw.describe())
    for n in hw.notes:
        print(f"  note: {n}")

    # Precision recommendation must return a valid tag for a range of sizes.
    for n in (4, 12, 14, 20, 24, 30):
        p = hw.recommend_precision(n)
        assert p in {"fp32", "fp64", "mixed"}, f"bad precision {p} for n={n}"
        print(f"  n={n:<3d} → precision={p}, fits≈{hw.max_qubits_fit(p)} qubits")


def test_override_precision():
    _banner("Override VQE_PRECISION=fp64")
    os.environ["VQE_PRECISION"] = "fp64"
    hw = HardwareProfile.detect()
    assert hw.recommend_precision(4) == "fp64"
    assert hw.recommend_precision(30) == "fp64"
    print("  fp64 forced across all problem sizes [PASS]")
    del os.environ["VQE_PRECISION"]


def test_override_gpu_off():
    _banner("Override USE_GPU=no")
    os.environ["USE_GPU"] = "no"
    hw = HardwareProfile.detect()
    assert hw.want_gpu() is False
    assert hw.recommend_precision(4) == "fp64"
    print("  GPU disabled via env [PASS]")
    del os.environ["USE_GPU"]


def test_override_backend():
    _banner("Override VQE_BACKEND=aer_cpu")
    os.environ["VQE_BACKEND"] = "aer_cpu"
    hw = HardwareProfile.detect()
    assert hw.recommend_backend() == "aer_cpu"
    print("  backend forced to aer_cpu [PASS]")
    del os.environ["VQE_BACKEND"]


if __name__ == "__main__":
    test_autodetect()
    test_override_precision()
    test_override_gpu_off()
    test_override_backend()
    print("\nAll HardwareProfile tests [PASS]")