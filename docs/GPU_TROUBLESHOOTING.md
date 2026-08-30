# GPU Troubleshooting

Working notes from getting Qiskit Aer's GPU backend + cuStateVec + MPI running on
various machines (IE University Slurm cluster, Lambda Cloud A100). Kept for the
next time a fresh instance turns into "GPU says CUDA OK but Aer refuses to load."

For live cluster-specific setup steps see `docs/CLUSTER_SETUP.md`.

---

## What is BROKEN (recurring failure modes)

### 1. `libcusparse.so.12: undefined symbol __nvJitLinkGetErrorLogSize_12_9`

```
libcusparse.so.12: undefined symbol: __nvJitLinkGetErrorLogSize_12_9,
version libnvJitLink.so.12
```

Root cause: conda pulled mismatched CUDA library versions — libcusparse expects
CUDA 13 nvJitLink, env has CUDA 12. Or, symmetrically, pip's
`nvidia-cusparse-cu12` wheel conflicts with the CUDA libs that another package
already pulled in.

Verified fix (locks the pip nvidia-cu12 set to a matched combo, then reinstalls
qiskit-aer-gpu on top):

```bash
pip uninstall -y qiskit-aer-gpu nvidia-cusparse-cu12 nvidia-cublas-cu12 \
    nvidia-cusolver-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12
pip install --no-cache-dir qiskit-aer-gpu
```

Resulting matched set (pinned in `environment.yml` pip section):
- nvidia-cublas-cu12==12.9.2.10
- nvidia-cuda-runtime-cu12==12.9.79
- nvidia-cuda-nvrtc-cu12==12.9.86
- nvidia-cusparse-cu12==12.5.10.65
- nvidia-cusolver-cu12==11.7.5.82
- nvidia-nvjitlink-cu12==12.9.86

### 2. `/usr/local/cuda/lib64` shadowing pip nvidia libs

Slurm scripts `slurm_trial.sh`/`slurm_gpu.sh` used to PREPEND `/usr/local/cuda/lib64`
to `LD_LIBRARY_PATH`, which made Aer load the system CUDA 12.4 libs instead of
the CUDA 12.9 pip wheels. Fix: the prepend line is removed/commented out
(committed in `228e1e4`). If you re-add anything CUDA-related to
`LD_LIBRARY_PATH`, verify Aer still loads by running the smoke test in
"Diagnostic commands" below.

### 3. conda-forge UCX 1.20.0: `libuct.so.0` references unexported symbol

`ucs_netif_is_ipoib` is unexported in 1.20. Fix: pin `ucx<1.20` in
`environment.yml`, plus auto-recovery in `install_native.sh`. However the
auto-recovery import check fails on some clusters (haskell) because it does not
set `UCX_TLS=sm,self` first — patch pending.

### 4. InfiniBand/IRDMA kernel bug (haskell, likely other IE nodes)

```
irdma0: iface failed to create UD QP TX wr:256 sge:6 inl:64 resp:0
RX wr:4096 sge:1 resp:0 failed: Invalid argument
```

Permanent workaround: force shared-memory only via
`UCX_TLS=sm,self UCX_NET_DEVICES=`. Already set in `slurm_*.sh` runtime scripts.
Not set in `install_native.sh`'s import check.

### 5. Login node `rust` CPU is too old for NumPy 2.x baseline

X86_V2 fault. Never run trial/benchmark interactively on rust — always submit
via Slurm so it dispatches to haskell. Corollary: `mpirun -bootstrap fork ...`
from rust fails with X86_V2 error.

---

## Things tried for the CUDA library mismatch, and why they didn't stick

### Attempt 1: `--force-reinstall ucx=1.18.1`

conda metadata said 1.18.1, but `nm -D libucp.so.0` still showed undefined
`ucs_global_opts_set_value_modifiable`. The on-disk binary was from 1.20.0 —
conda skipped the actual reinstall because metadata thought it was already
1.18.1.

### Attempt 2: `conda clean --all` + `--force-reinstall`

Same outcome. Cache clear didn't help — the package wasn't actually re-downloaded
because metadata version matched the spec.

### Attempt 3: Nuke env + rebuild via `install_native.sh`

Got a fresh env but discovered two additional issues:
- (a) Conda pulled GCC 15.2.0, too new for CUDA 12.4 nvcc → CMake fell back to
  CPU-only build (`kernel_stub.cpp`, no GPU dispatcher).
- (b) qiskit-aer-gpu installed but its libcusparse expects nvJitLink 12.9, env
  has older nvJitLink → AerSimulator GPU fails to load.

### Attempt 4: Pin libnvjitlink + libcusparse to 12.6

Conda solver refused:

```
libcusparse =12.6 requires cuda-version >=13.0
libnvjitlink =12.6 requires cuda-version 12.x
→ incompatible
```

conda-forge labels libcusparse 12.6.x as needing CUDA 13 metadata for the latest
revision, even though it's a CUDA 12.x library. Solver can't reconcile.

### Why this isn't a 5-minute fix

The conda-forge CUDA library tree is fundamentally inconsistent for users who
- have a system CUDA (12.4) at `/usr/local/cuda`,
- want conda to manage Python deps (qiskit-aer-gpu, cupy), and
- get pulled-in CUDA libs from multiple conda packages, each pinned differently.

System CUDA plus conda's qiskit-aer-gpu pulls a conflicting cusparse from pip's
`nvidia-cusparse-cu12` wheel — that's the thing that actually breaks.

---

## What to try NEXT for GPU (priority order)

### A. Force system CUDA via `LD_LIBRARY_PATH`

The system CUDA at `/usr/local/cuda` is consistent. Force qiskit-aer to use
system libs instead of pip's bundled ones:

```bash
# in srun bash on haskell:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# remove pip's bundled nvidia-* libs from sys.path
python -c "
import sys
sys.path = [p for p in sys.path if 'nvidia' not in p]
from qiskit_aer import AerSimulator
sim = AerSimulator(method='statevector', device='GPU')
print('GPU OK')
"
```

If this works → fix is to set `LD_LIBRARY_PATH` in the slurm scripts so the
system CUDA libs shadow pip's.

### B. Pin qiskit-aer-gpu to a known-good CUDA pip combo

```bash
pip uninstall -y qiskit-aer-gpu nvidia-cusparse-cu12 nvidia-cublas-cu12 \
    nvidia-cusolver-cu12 nvidia-nvjitlink-cu12
pip install "qiskit-aer-gpu==0.13.3"   # older, tied to CUDA 11.8 libs
```

### C. Build qiskit-aer-gpu from source against system CUDA

Slowest but bulletproof:

```bash
pip uninstall -y qiskit-aer-gpu
AER_THRUST_BACKEND=CUDA pip install --no-binary qiskit-aer qiskit-aer
```

Compiles against `/usr/local/cuda`, never touches pip's bundled libs. ~10–15 min
to compile.

### D. Publication fallback — accept April RTX 6000 data as authoritative

The April runs in `results/rtx-6000-ada-generation/` already demonstrate GPU
cuStateVec, and the August A100 runs (`results/a100-sxm4-40gb/`) supersede them
for the scaling story. If a reproducibility rerun on a specific hardware target
gets stuck in library hell, that's a valid outcome to note in the paper — don't
burn another week chasing it.

---

## CRITICAL "do not do this" list

1. **Don't run `python tests/test_layers_run.py` interactively on rust.** Login
   node has no x86-v2 CPU. Use `make slurm-trial`.
2. **Don't `conda update`** anything in the hybrid-vqe env. Conda will pull in
   UCX 1.20 or break CUDA libs again.
3. **Don't trust `conda list ucx`** when diagnosing UCX. Always check the actual
   binary — see the diagnostic command below.
4. **Don't use `set -euo pipefail`** in Slurm scripts that source conda. Conda's
   activate scripts hit unset vars. Use `set -eo pipefail`.
5. **Don't pipe `nvidia-smi | head`** in a Slurm script with `set -e`. SIGPIPE =
   exit 13 = script aborts before doing anything. Already fixed in `fd3ccf6`.
6. **Don't run `install_native.sh` from rust.** It needs nvcc at
   `/usr/local/cuda/bin/nvcc` which only exists on haskell.

---

## Diagnostic commands

```bash
# Is libucp.so.0 actually broken (vs metadata claim)?
nm -D ~/miniforge3/envs/hybrid-vqe/lib/libucp.so.0 | grep ucs_global_opts_set_value_modifiable
# U = bad (undefined), T = good

# Test mpi4py with proper UCX vars (always use these on haskell)
UCX_TLS=sm,self UCX_NET_DEVICES= \
    python -c "from mpi4py import MPI; print('ok', MPI.COMM_WORLD.Get_size())"

# Test if qiskit-aer GPU backend loads at all
python -c "from qiskit_aer import AerSimulator; AerSimulator(method='statevector', device='GPU'); print('GPU OK')"

# Check ALL UCX-related conda packages
conda list | grep -iE "ucx|ucs|ucm|mpi"

# Check CUDA library symbol consistency for Aer
ldd ~/miniforge3/envs/hybrid-vqe/lib/python3.11/site-packages/qiskit_aer/backends/aer_compiler.cpython-*.so \
    2>&1 | grep -i "not found\|undefined"
```
