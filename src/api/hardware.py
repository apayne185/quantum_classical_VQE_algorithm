"""Hardware auto-detection and policy layer.

Probes GPU vendor, class, memory, and MPI availability at stack init.
Derived decisions (precision, backend, qubit limits) are exposed via
helper methods on HardwareProfile so the rest of the stack doesn't need
to know about hardware details.

Researchers override auto-detection via env vars:
  VQE_PRECISION  = auto | fp32 | fp64 | mixed
  VQE_BACKEND    = auto | cuStateVec | aer_gpu | aer_cpu
  USE_GPU        = yes  | no    (existing; takes precedence over detection)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


# Known GPU families: (substring, class, est. fp64:fp32 ratio)
# Datacenter cards have full FP64 and do not benefit from mixed precision.
_GPU_DATABASE: list[tuple[str, str, float]] = [
    ("A100", "datacenter", 1 / 2),
    ("H100", "datacenter", 1 / 2),
    ("V100", "datacenter", 1 / 2),
    ("A40", "workstation", 1 / 32),
    ("RTX 6000 Ada", "workstation", 1 / 64),
    ("RTX A6000", "workstation", 1 / 32),
    ("RTX 4090", "consumer", 1 / 64),
    ("RTX 4080", "consumer", 1 / 64),
    ("RTX 3090", "consumer", 1 / 64),
    ("RTX 3080", "consumer", 1 / 64),
    ("GTX 1650", "consumer", 1 / 32),
    ("GTX 1660", "consumer", 1 / 32),
]


@dataclass
class HardwareProfile:
    has_cuda: bool = False
    gpu_name: str = ""
    gpu_class: str = "unknown"       # datacenter | workstation | consumer | unknown
    gpu_memory_gb: float = 0.0
    fp64_ratio: float = 1 / 32        # estimated FP64:FP32 throughput
    compute_capability: tuple[int, int] | None = None
    has_cuquantum: bool = False
    has_aer_gpu: bool = False
    has_mpi: bool = False
    mpi_size: int = 1

    # User overrides captured from env vars
    override_precision: str = "auto"
    override_backend: str = "auto"
    override_use_gpu: str = ""

    notes: list[str] = field(default_factory=list)

    @classmethod
    def detect(cls) -> HardwareProfile:
        p = cls()
        p.override_precision = os.environ.get("VQE_PRECISION", "auto").strip().lower()
        p.override_backend = os.environ.get("VQE_BACKEND", "auto").strip().lower()
        p.override_use_gpu = os.environ.get("USE_GPU", "").strip().lower()
        p._detect_gpu()
        p._detect_libs()
        p._detect_mpi()
        return p

    # ---------- probes ----------

    def _detect_gpu(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.notes.append("nvidia-smi unavailable; assuming CPU-only")
            return

        if out.returncode != 0 or not out.stdout.strip():
            self.notes.append("nvidia-smi found no GPU")
            return

        first = out.stdout.strip().splitlines()[0].split(",")
        self.has_cuda = True
        self.gpu_name = first[0].strip()
        try:
            self.gpu_memory_gb = float(first[1].strip()) / 1024.0
        except (ValueError, IndexError):
            pass
        try:
            major, minor = first[2].strip().split(".")
            self.compute_capability = (int(major), int(minor))
        except (ValueError, IndexError):
            pass

        for sub, cls_, ratio in _GPU_DATABASE:
            if sub in self.gpu_name:
                self.gpu_class = cls_
                self.fp64_ratio = ratio
                return
        self.notes.append(f"GPU '{self.gpu_name}' not in database; "
                          "conservative defaults applied")

    def _detect_libs(self) -> None:
        try:
            from qiskit_aer import AerSimulator
            try:
                AerSimulator(method="statevector", device="GPU")
                self.has_aer_gpu = True
            except Exception:
                pass
        except ImportError:
            pass
        try:
            import cuquantum  # noqa: F401
            self.has_cuquantum = True
        except ImportError:
            pass

    def _detect_mpi(self) -> None:
        try:
            from mpi4py import MPI
            self.has_mpi = True
            # MPI may not be initialized yet (interface.py defers init to hpc_core.init_mpi).
            # Only query the communicator size if MPI is already up; otherwise the C-level
            # MPI_Comm_size call aborts before MPI_Init has run.
            if MPI.Is_initialized():
                self.mpi_size = MPI.COMM_WORLD.Get_size()
        except ImportError:
            pass

    # ---------- policy decisions ----------

    def want_gpu(self) -> bool:
        """Whether the stack should attempt to use the GPU path."""
        if self.override_use_gpu == "no":
            return False
        if self.override_use_gpu == "yes":
            return self.has_cuda
        return self.has_cuda

    def recommend_precision(self, num_qubits: int) -> str:
        """Pick fp32 or fp64 based on hardware class and problem size.

        Rules:
          - user override wins (fp32 | fp64)
          - no GPU → fp64 (CPU doesn't benefit from reduced precision)
          - fp32 only when: consumer/workstation GPU AND < 20 qubits AND user explicitly asked for it
          - everything else → fp64 (required for chemical accuracy)
        """
        if self.override_precision in {"fp32", "fp64"}:
            return self.override_precision
        return "fp64"

    def recommend_backend(self) -> str:
        """Pick a simulator backend implementation."""
        if self.override_backend in {"custatevec", "aer_gpu", "aer_cpu"}:
            return self.override_backend
        if self.has_cuda and self.has_cuquantum and self.has_aer_gpu:
            return "custatevec"
        if self.has_cuda and self.has_aer_gpu:
            return "aer_gpu"
        return "aer_cpu"

    def max_qubits_fit(self, precision: str = "fp64") -> int:
        """Upper bound on qubit count that fits in GPU memory."""
        # fp64 statevector: 16 bytes/amplitude (complex128)
        # fp32 statevector:  8 bytes/amplitude (complex64)
        bytes_per_amp = 16 if precision == "fp64" else 8
        if self.gpu_memory_gb <= 0:
            return 0
        # Reserve ~25% for gates / Qiskit overhead
        usable = self.gpu_memory_gb * 0.75 * (1024 ** 3)
        n = 0
        while (2 ** (n + 1)) * bytes_per_amp <= usable:
            n += 1
        return n

    def describe(self) -> str:
        gpu = (
            f"{self.gpu_name} ({self.gpu_class}, {self.gpu_memory_gb:.1f} GB, "
            f"fp64:fp32≈{self.fp64_ratio:.3f})"
            if self.has_cuda else "no GPU (CPU only)"
        )
        libs = []
        if self.has_aer_gpu: libs.append("aer-gpu")
        if self.has_cuquantum: libs.append("cuquantum")
        libs_str = ", ".join(libs) if libs else "none"
        return (
            f"[hw] {gpu} | libs: {libs_str} | "
            f"MPI={self.mpi_size if self.has_mpi else 'off'} | "
            f"backend={self.recommend_backend()}"
        )