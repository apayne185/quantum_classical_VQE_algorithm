#!/usr/bin/env bash
# Slurm scaling-sweep helper - submits 4 jobs for P in {1,2,4,8}.
# Each job runs the full benchmark (strong + weak scaling are both
# executed inside benchmarks/local_test_run.py) with a different rank count.
# Capstone limit: 1 GPU per job, so ranks share the single GPU.
#
# Usage:
#   bash scripts/slurm_scaling.sh            # default P set: 1 2 4 8
#   RANKS="2 4"    bash scripts/slurm_scaling.sh
#   JOB_PREFIX=ws  bash scripts/slurm_scaling.sh    # custom job name prefix

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RANKS="${RANKS:-1 2 4 8}"
JOB_PREFIX="${JOB_PREFIX:-vqe-scale}"

mkdir -p "$REPO_ROOT/results/slurm"
cd "$REPO_ROOT"

for P in $RANKS; do
  echo "[scaling] Submitting P=$P ..."
  sbatch \
    --ntasks="$P" \
    --job-name="${JOB_PREFIX}-P${P}" \
    --output="results/slurm/${JOB_PREFIX}-P${P}_%j.log" \
    scripts/slurm_gpu.sh
done

echo ""
echo "[scaling] Submitted. Track with: squeue -u \$USER"
echo "[scaling] Logs:   results/slurm/${JOB_PREFIX}-P*.log"
