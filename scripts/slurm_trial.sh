#!/usr/bin/env bash
# Slurm batch script - 7-layer diagnostic on 1 GPU (smoke test)
# Submit with: sbatch scripts/slurm_trial.sh

#SBATCH --job-name=vqe-trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/vqe-trial_%j.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/quantum_classical_VQE_algorithm}"
ENV_PATH="${ENV_PATH:-/scratch/$USER/hybrid-vqe}"
NP="${NP:-2}"

mkdir -p "$REPO_ROOT/results/slurm"
cd "$REPO_ROOT"

if [ -d "$ENV_PATH" ]; then
  source "$HOME/miniforge3/bin/activate" "$ENV_PATH"
else
  source "$HOME/miniforge3/bin/activate" hybrid-vqe
fi

export PYTHONPATH="$REPO_ROOT/build:$REPO_ROOT:${PYTHONPATH:-}"
export BACKEND="simulator"
export USE_GPU="yes"

mpirun -np "$NP" python tests/test_layers_run.py
