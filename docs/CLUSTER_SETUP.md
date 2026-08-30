# Cluster Setup — IE University Slurm (`rust` / `haskell`)

Reference for the IE University Slurm cluster used in the thesis's GPU
troubleshooting phase (June 2026). All active benchmark work since has moved to
Lambda Cloud / local Docker; this file exists so the cluster path can be
revived without re-deriving everything.

For failure modes and diagnostic commands see `docs/GPU_TROUBLESHOOTING.md`.

---

## Topology

- `rust` — login node. CPU is too old for NumPy 2.x baseline (X86_V2 fault).
  **Never run Python trials interactively on rust.** Only submit jobs.
- `haskell` — GPU compute node (RTX 6000 Ada 48GB). All real work happens here
  via `srun` or a batch job.
- Conda env name: `hybrid-vqe`.

## Connect

```bash
ssh -J capstone21@ssh.iesci.tech capstone21@10.205.20.10
```

## Sanity — env healthy?

```bash
conda activate hybrid-vqe                        # on haskell, not rust
python -c "from mpi4py import MPI; print('OK')"
```

## Interactive GPU session

```bash
srun --partition=interactive --gres=gpu:1 --cpus-per-task=4 --mem=16G \
     --time=01:00:00 --pty bash
```

Once inside, verify GPU Aer loads:

```bash
python -c "from qiskit_aer import AerSimulator; \
    AerSimulator(method='statevector', device='GPU'); print('GPU OK')"
```

## Submit a Slurm trial (from rust)

```bash
make slurm-trial
squeue -u $USER
ls -lt results/slurm/ | head -3      # find the latest log
```

Grep for GPU markers to confirm cuStateVec actually engaged:

```bash
grep -E "\[hw\]|cuStateVec|GPU=|SV_backend|\[LAYER|Tests passed|Path:" \
    results/slurm/vqe-trial_<JOBID>.log | head -25
```

Success markers to look for:
- `libs: aer-gpu, cuquantum`
- `GPU=enabled, SV_backend=GPU (cuStateVec)`
- `Path: GPU-cuStateVec MPI-distributed (2 ranks)` per iteration

---

## Working `cmake` invocation for `hpc_core`

Once GCC 13 is available in the env (see below), build from
`~/quantum_classical_VQE_algorithm/build` inside `srun` on haskell:

```bash
/usr/bin/cmake .. \
    -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DCMAKE_CXX_COMPILER=<conda gcc 13 g++ path> \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=<conda gcc 13 g++ path> \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DMPI_CXX_SKIP_MPICXX=TRUE \
    -DMPI_CXX_HEADER_DIR=$CONDA_PREFIX/include \
    -DMPI_CXX_LIB_NAMES=mpi \
    -DMPI_mpi_LIBRARY=$CONDA_PREFIX/lib/libmpi.so
```

Verify the resulting build actually has CUDA support:

```bash
PYTHONPATH=./build:. python -c "import hpc_core; print('CUDA build:', hpc_core.cuda_build())"
# Expected: CUDA build: True
```

---

## Known blockers

### GCC 15.2 vs CUDA 12.4 nvcc (as of 2026-06-19)

Both system `/usr/bin/g++` and conda's default `x86_64-conda-linux-gnu-g++` are
GCC 15.2 on this cluster. CUDA 12.4 hard-requires GCC ≤ 13. Without a GCC 13
in-env, `hpc_core` builds CPU-only via `kernel_stub.cpp`, then
`hpc_core.set_cuda_device(rank)` throws at runtime → `use_gpu=False` → no GPU.

Attempted fix: `conda install -c conda-forge "gcc_linux-64=13.*" "gxx_linux-64=13.*" -y`
— but conda binary paths were showing "No such file or directory" after install.
Needs verification of actual paths in `$CONDA_PREFIX/bin/`.

### NFS mount failures on `haskell` (as of 2026-08-06)

Last observed status. Blocks all use of the cluster until IT resolves. Reason
this cluster stopped being the active platform; work migrated to Lambda Cloud.

### CUDA library mismatch — see `docs/GPU_TROUBLESHOOTING.md`

libcusparse/nvJitLink version conflicts between conda-forge, pip nvidia wheels,
and the system CUDA at `/usr/local/cuda`. Full history and attempted fixes are
in the troubleshooting doc.
