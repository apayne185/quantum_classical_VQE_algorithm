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


def test_max_qubits_fit_accounts_for_ranks_sharing_a_gpu():
    """Regression test for the CO2/A100 incident (2026-08-06): a 30-qubit
    problem at P=2 on a single-GPU instance ran for over an hour without
    completing an iteration. Root cause: max_qubits_fit() budgeted for one
    simulation, but round-robin CUDA device assignment (rank % gpu_count)
    put both ranks on the same physical GPU, each redundantly building its
    own full statevector. This asserts the fix: the budget must shrink when
    more ranks share a GPU, and recover when ranks map 1:1 to separate GPUs.
    """
    hw = HardwareProfile()
    hw.gpu_memory_gb = 39.6
    hw.gpu_count = 1

    one_rank = hw.max_qubits_fit("fp64", mpi_size=1)
    two_ranks_one_gpu = hw.max_qubits_fit("fp64", mpi_size=2)
    assert two_ranks_one_gpu < one_rank, \
        "two ranks sharing one GPU must get a smaller budget than one rank alone"

    hw.gpu_count = 2
    two_ranks_two_gpus = hw.max_qubits_fit("fp64", mpi_size=2)
    assert two_ranks_two_gpus == one_rank, \
        "one rank per GPU (no sharing) should recover the single-simulation budget"

    fp32_budget = HardwareProfile(gpu_memory_gb=39.6, gpu_count=1).max_qubits_fit("fp32", mpi_size=1)
    assert fp32_budget > one_rank, "fp32 (8 bytes/amplitude) must fit more qubits than fp64 (16 bytes)"

    # The actual CO2 failure: 30 qubits, P=2, 1 physical GPU -- must now be
    # flagged as exceeding the adjusted (not the single-sim) budget.
    co2_qubits = 30
    real_hw = HardwareProfile(gpu_memory_gb=39.6, gpu_count=1)
    assert co2_qubits > real_hw.max_qubits_fit("fp64", mpi_size=2), \
        "the CO2 P=2 case must be flagged as over budget by the rank-aware check"
    print("  MPI-aware max_qubits_fit correctly reduces budget when ranks share a GPU [PASS]")


if __name__ == "__main__":
    test_autodetect()
    test_override_precision()
    test_override_gpu_off()
    test_override_backend()
    test_max_qubits_fit_accounts_for_ranks_sharing_a_gpu()
    print("\nAll HardwareProfile tests [PASS]")