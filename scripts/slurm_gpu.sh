#!/usr/bin/env bash
# Slurm batch script - full simulator benchmark on 1 GPU.
# Submit with: sbatch scripts/slurm_gpu.sh
# Capstone account limit: max 1 GPU per job.

#SBATCH --job-name=vqe-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/vqe-gpu_%j.log

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/quantum_classical_VQE_algorithm}"
ENV_PATH="${ENV_PATH:-/scratch/$USER/hybrid-vqe}"
NP="${NP:-2}"

mkdir -p "$REPO_ROOT/results/slurm"
cd "$REPO_ROOT"

echo "=== Slurm VQE GPU job ==="
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-n/a}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "MPI ranks:  $NP"

# Activate conda env (prefer scratch install)
if [ -d "$ENV_PATH" ]; then
  source "$HOME/miniforge3/bin/activate" "$ENV_PATH"
else
  source "$HOME/miniforge3/bin/activate" hybrid-vqe
fi

export PYTHONPATH="$REPO_ROOT/build:$REPO_ROOT:${PYTHONPATH:-}"
export BACKEND="simulator"
export USE_GPU="yes"

# CUDA path (no module system on this cluster)
export PATH="/usr/local/cuda/bin:$PATH"
# LD_LIBRARY_PATH not modified - rely on conda env + pip nvidia-* wheels for runtime CUDA libs

nvidia-smi

NP="${SLURM_NTASKS:-2}"
export UCX_TLS=sm,self          # single-node: shared memory only, skip RDMA
export UCX_NET_DEVICES=          # disable network devices entirely
mpirun -bootstrap fork -n "$NP" python benchmarks/local_test_run.py
