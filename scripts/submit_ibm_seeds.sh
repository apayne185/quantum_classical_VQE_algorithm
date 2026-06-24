#!/usr/bin/env bash
# Submit multi-seed IBM QPU runs for publication statistics.
#
# Usage:
#   bash scripts/submit_ibm_seeds.sh                        # default: seeds 42 43 44, 10 iters each
#   SEEDS="42 43 44" MAX_ITERS=10 bash scripts/submit_ibm_seeds.sh
#   SEEDS="42" MAX_ITERS=5 bash scripts/submit_ibm_seeds.sh   # smoke test
#
# IBM budget estimate:
#   ~1-3s billed QPU per iteration (4-qubit H2, 4096 shots, 2 PUBs/iter)
#   10 iters x 3 seeds = ~30 jobs = ~30-90s billed total

set -eo pipefail

SEEDS="${SEEDS:-42 43 44}"
MAX_ITERS="${MAX_ITERS:-10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/results/slurm"
cd "$REPO_ROOT"

echo "[ibm-seeds] Submitting seeds: $SEEDS"
echo "[ibm-seeds] Iterations per run: $MAX_ITERS"

for s in $SEEDS; do
    jobid=$(sbatch --parsable \
                   --job-name="vqe-ibm-s${s}" \
                   --output="results/slurm/vqe-ibm-s${s}_%j.log" \
                   --export="ALL,SEED=$s,MAX_ITERS=$MAX_ITERS" \
                   scripts/slurm_ibm.sh)
    echo "  seed=$s  -> job $jobid"
done

echo ""
echo "[ibm-seeds] All jobs queued. Watch with:"
echo "  squeue -u \$USER"
echo "[ibm-seeds] After completion, aggregate with:"
echo "  python benchmarks/aggregate_seeds.py --backend ibm"