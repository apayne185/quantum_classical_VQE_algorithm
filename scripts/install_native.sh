#!/usr/bin/env bash
# Native (bare-metal / HPC) install for the hybrid VQE stack.
# Use this on clusters where Docker is unavailable (Slurm HPCs).
# For local / reproducible runs, prefer the Docker path: `make build`.
#
# Usage:
#   bash scripts/install_native.sh                    # installs conda env + builds C++ module
#   SCRATCH=/scratch/$USER bash scripts/install_native.sh   # install env to scratch (HPC)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="hybrid-vqe"
SCRATCH="${SCRATCH:-}"

echo "[install] Repo root: $REPO_ROOT"

# Conda check
if ! command -v conda >/dev/null 2>&1; then
  echo "[install] ERROR: conda not found. Install miniforge first:"
  echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  echo "  bash Miniforge3-Linux-x86_64.sh -b -p \$HOME/miniforge3"
  echo "  eval \"\$(\$HOME/miniforge3/bin/conda shell.bash hook)\""
  exit 1
fi

# Create conda env (on scratch if requested, otherwise default location)
if [ -n "$SCRATCH" ]; then
  ENV_PREFIX="$SCRATCH/$ENV_NAME"
  echo "[install] Creating env at $ENV_PREFIX (scratch, fast SSD)"
  conda env create -f "$REPO_ROOT/environment.yml" --prefix "$ENV_PREFIX" || \
    conda env update -f "$REPO_ROOT/environment.yml" --prefix "$ENV_PREFIX"
  ACTIVATE="conda activate $ENV_PREFIX"
else
  echo "[install] Creating env '$ENV_NAME' in default conda location"
  conda env create -f "$REPO_ROOT/environment.yml" || \
    conda env update -f "$REPO_ROOT/environment.yml"
  ACTIVATE="conda activate $ENV_NAME"
fi

# Activate for this shell
eval "$(conda shell.bash hook)"
if [ -n "$SCRATCH" ]; then
  conda activate "$ENV_PREFIX"
else
  conda activate "$ENV_NAME"
fi

# Verify toolchain
echo "[install] Checking CUDA + MPI toolchain..."
command -v nvcc   >/dev/null 2>&1 && nvcc --version | tail -1   || echo "  WARN: nvcc not found (GPU build will be skipped)"
command -v mpicxx >/dev/null 2>&1 && mpicxx --showme:version    || echo "  WARN: mpicxx not found"
command -v cmake  >/dev/null 2>&1 && cmake --version | head -1

# Build the C++/CUDA hpc_core pybind11 module
echo "[install] Building hpc_core via CMake..."
cd "$REPO_ROOT"
mkdir -p build && cd build
cmake .. \
  -DPython_EXECUTABLE="$(which python)" \
  -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

# Smoke test
cd "$REPO_ROOT"
echo "[install] Smoke testing..."
PYTHONPATH="$REPO_ROOT/build:$REPO_ROOT" python -c "
import hpc_core
from src.api.interface import HPCHybridStack
print('[install] hpc_core + HPCHybridStack import OK')
"

echo ""
echo "[install] DONE. To use:"
echo "  $ACTIVATE"
echo "  export PYTHONPATH=$REPO_ROOT/build:$REPO_ROOT"
echo "  mpirun -np 2 python tests/test_layers_run.py"
