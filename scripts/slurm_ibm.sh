#!/usr/bin/env bash
# Slurm batch script — IBM Quantum QPU run (H2 ground state)
# Submit with: make slurm-ibm   OR   sbatch scripts/slurm_ibm.sh
#
# Requires .env in the repo root with IBM credentials:
#   cp .env.example .env   # then fill in your token and instance

#SBATCH --job-name=vqe-ibm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/vqe-ibm_%j.log

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/quantum_classical_VQE_algorithm}"
ENV_PATH="${ENV_PATH:-/scratch/$USER/hybrid-vqe}"

mkdir -p "$REPO_ROOT/results/slurm"
cd "$REPO_ROOT"

# Load credentials from .env
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.env"
    set +a
else
    echo "[ERROR] No .env file found at $REPO_ROOT/.env"
    echo "  Run: cp .env.example .env"
    echo "  Then fill in your IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE"
    exit 1
fi

# Validate credentials are real (not placeholder values)
if [ -z "${IBM_QUANTUM_TOKEN:-}" ] || [ "$IBM_QUANTUM_TOKEN" = "your_token_here" ]; then
    echo "[ERROR] IBM_QUANTUM_TOKEN is not set in .env"
    echo "  Get your token at: https://quantum.cloud.ibm.com"
    exit 1
fi
if [ -z "${IBM_QUANTUM_INSTANCE:-}" ] || [ "$IBM_QUANTUM_INSTANCE" = "ibm-q/open/main" ]; then
    echo "[ERROR] IBM_QUANTUM_INSTANCE is not set in .env"
    exit 1
fi

echo "=== IBM QPU Job ==="
echo "Host:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-n/a}"
echo "Backend:   ${IBM_QUANTUM_BACKEND:-ibm_marrakesh}"
echo "MPI ranks: ${SLURM_NTASKS:-2}"

# Activate conda env
if [ -d "$ENV_PATH" ]; then
    source "$HOME/miniforge3/bin/activate" "$ENV_PATH"
else
    source "$HOME/miniforge3/bin/activate" hybrid-vqe
fi

export PYTHONPATH="$REPO_ROOT/build:$REPO_ROOT:${PYTHONPATH:-}"
export BACKEND="ibm_cloud"
export USE_GPU="yes"

# CUDA path
export PATH="/usr/local/cuda/bin:$PATH"
# LD_LIBRARY_PATH not modified - rely on conda env + pip nvidia-* wheels for runtime CUDA libs

NP="${SLURM_NTASKS:-2}"
export UCX_TLS=sm,self
export UCX_NET_DEVICES=

nvidia-smi || true

mpirun -bootstrap fork -n "$NP" python benchmarks/ibm_test_run.py