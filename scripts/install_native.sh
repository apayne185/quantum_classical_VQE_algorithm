#!/usr/bin/env bash
# Native (bare-metal / HPC) install for the hybrid VQE stack.
# Use this on clusters where Docker is unavailable (Slurm HPCs).
# For local / reproducible runs, prefer the Docker path: `make build`.
#
# Usage:
#   bash scripts/install_native.sh
#   SCRATCH=/scratch/$USER bash scripts/install_native.sh
#
# Key overrides:
#   SCRATCH      scratch filesystem root     (default: /scratch/$USER)
#   ENV_PATH     full path for conda env     (default: $SCRATCH/hybrid-vqe)
#   MINIFORGE    conda root                  (default: $HOME/miniforge3)
#   CUDA_MODULE  exact cluster module name   (default: auto-detect)
#   CUDA_ARCH    semicolon-separated SM list (default: 70;75;80;86;89;90)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-/scratch/$USER}"
ENV_PATH="${ENV_PATH:-$SCRATCH/hybrid-vqe}"
ENV_NAME="hybrid-vqe"
MINIFORGE="${MINIFORGE:-$HOME/miniforge3}"
CUDA_ARCH="${CUDA_ARCH:-70;75;80;86;89;90}"

echo "[install] Repo root: $REPO_ROOT"
echo "[install] Env path:  $ENV_PATH"

# ------------------------------------------------------------------
# Step 1: Load CUDA module (needed for nvcc + libcudart at build time)
# Override with CUDA_MODULE=<name> if auto-detect picks the wrong one.
# ------------------------------------------------------------------
_cuda_loaded=0
if [ -n "${CUDA_MODULE:-}" ]; then
    module load "$CUDA_MODULE"
    _cuda_loaded=1
elif command -v module &>/dev/null; then
    for _mod in \
        "cuda/12.6" "cuda/12.4" "cuda/12.2" "cuda/12.0" "cuda/11.8" \
        "CUDA/12.6"  "CUDA/12.4"  "CUDA/12.2"  "CUDA/12.0" \
        "cuda" "CUDA"
    do
        if module load "$_mod" 2>/dev/null; then
            echo "[install] Loaded CUDA module: $_mod"
            _cuda_loaded=1
            break
        fi
    done
fi
[ "$_cuda_loaded" -eq 0 ] && \
    echo "[install] WARN: no CUDA module loaded — build will fail without nvcc"

# ------------------------------------------------------------------
# Step 2: Create or update conda env on scratch
# The SLURM scripts activate $ENV_PATH first, then fall back to the
# named env "hybrid-vqe" — so we always write to $ENV_PATH.
# ------------------------------------------------------------------
if [ ! -d "$MINIFORGE" ]; then
    echo "[install] ERROR: miniforge not found at $MINIFORGE"
    echo "  Install with:"
    echo "    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "    bash Miniforge3-Linux-x86_64.sh -b -p $MINIFORGE"
    exit 1
fi

# shellcheck source=/dev/null
source "$MINIFORGE/bin/activate"

if [ -d "$ENV_PATH" ]; then
    echo "[install] Updating existing env at $ENV_PATH ..."
    conda env update --prefix "$ENV_PATH" \
        --file "$REPO_ROOT/environment.yml" --prune -q
else
    echo "[install] Creating new env at $ENV_PATH ..."
    mkdir -p "$(dirname "$ENV_PATH")"
    conda env create --prefix "$ENV_PATH" \
        --file "$REPO_ROOT/environment.yml" -q
fi

conda activate "$ENV_PATH"

# ------------------------------------------------------------------
# Step 3: Verify toolchain
# conda provides cmake, mpicxx (mpich), and python.
# nvcc must come from the CUDA module loaded above.
# ------------------------------------------------------------------
echo "[install] Toolchain check:"
command -v nvcc   >/dev/null 2>&1 && echo "  nvcc:  $(nvcc --version | grep release)" \
    || echo "  nvcc:  NOT FOUND — GPU build will fail"
command -v mpicxx >/dev/null 2>&1 && echo "  mpi:   $(mpicxx --version 2>&1 | head -1)" \
    || echo "  mpi:   NOT FOUND"
command -v cmake  >/dev/null 2>&1 && echo "  cmake: $(cmake --version | head -1)" \
    || echo "  cmake: NOT FOUND"

# ------------------------------------------------------------------
# Step 4: Build hpc_core.so
# conda provides pybind11, mpich, cmake, and python.
# nlohmann_json is fetched automatically by CMake FetchContent.
# ------------------------------------------------------------------
echo "[install] Building hpc_core via CMake (archs: $CUDA_ARCH) ..."
mkdir -p "$REPO_ROOT/build"
cd "$REPO_ROOT/build"

cmake "$REPO_ROOT" \
    -DPython_EXECUTABLE="$(which python)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

make -j"$(nproc)"

# ------------------------------------------------------------------
# Smoke test — import only, no MPI calls (must run outside mpirun)
# ------------------------------------------------------------------
cd "$REPO_ROOT"
echo "[install] Smoke testing import ..."
PYTHONPATH="$REPO_ROOT/build:$REPO_ROOT" python -c "
import hpc_core
print('[install] hpc_core import OK —', [x for x in dir(hpc_core) if not x.startswith('_')])
"

SO_FILE="$(ls "$REPO_ROOT/build"/hpc_core*.so 2>/dev/null | head -1)"
echo ""
echo "=== Install complete ==="
echo "  Module: $SO_FILE"
echo ""
echo "Next steps:"
echo "  Diagnostic (no SLURM):  REPO_ROOT=$REPO_ROOT make native-trial NP=2"
echo "  SLURM trial job:        make slurm-trial"
echo "  Full scaling sweep:     make slurm-scaling"
