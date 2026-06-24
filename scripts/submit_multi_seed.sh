#!/usr/bin/env bash
# Submit multi-seed simulator benchmark jobs for publication statistics.
# Each seed runs all molecules in benchmarks/local_test_run.py sequentially.
#
# Usage:
#   bash scripts/submit_multi_seed.sh                # default: seeds 42 43 44
#   SEEDS="42 43 44 45 46" bash scripts/submit_multi_seed.sh
#   MOLECULES="H2 LiH" SEEDS="42 43" bash scripts/submit_multi_seed.sh

set -eo pipefail

SEEDS="${SEEDS:-42 43 44}"
MOLECULES="${MOLECULES:-}"   # empty = default 4 molecules

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/results/slurm"
cd "$REPO_ROOT"

echo "[multi-seed] Submitting seeds: $SEEDS"
echo "[multi-seed] Molecules: ${MOLECULES:-<default: H2 LiH BeH2 H2O>}"

for s in $SEEDS; do
    extra_env=""
    [ -n "$MOLECULES" ] && extra_env="$extra_env,MOLECULES=$MOLECULES"

    jobid=$(sbatch --parsable \
                   --job-name="vqe-seed${s}" \
                   --output="results/slurm/vqe-seed${s}_%j.log" \
                   --export="ALL,SEED=$s$extra_env" \
                   scripts/slurm_gpu.sh)
    echo "  seed=$s  -> job $jobid"
done

echo ""
echo "[multi-seed] All jobs queued. Watch with:"
echo "  squeue -u \$USER"
echo "[multi-seed] After completion, aggregate with:"
echo "  python benchmarks/aggregate_seeds.py"