#!/usr/bin/env bash
# Slurm batch script - full simulator benchmark on 1 GPU.
# Submit with: sbatch scripts/slurm_gpu.sh
# Capstone account limit: max 1 GPU per job.

#SBATCH --job-name=vqe-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/vqe-gpu_%j.log

set -euo pipefail

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

nvidia-smi

mpirun -np "$NP" python benchmarks/local_test_run.py
