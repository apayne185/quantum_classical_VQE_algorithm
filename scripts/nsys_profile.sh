#!/usr/bin/env bash
# Capture an Nsight Systems timeline of one short VQE run on H2O.
# Design + rationale: docs/PROFILING.md.
#
# Assumes: running inside vqe-mpi-gpu:profiling (or a host with nsys
# installed + this repo checked out). GPU visible. Conda env active if not
# using the container.
#
# Output: results/profiling/<timestamp>.nsys-rep, opened later in the
# Nsight Systems GUI locally.

set -eo pipefail

MOLECULE="${MOLECULE:-H2O}"
MAX_ITERS="${MAX_ITERS:-10}"
NP="${NP:-2}"
OUT_DIR="${OUT_DIR:-results/profiling}"

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
REPORT="${OUT_DIR}/${MOLECULE}_np${NP}_iters${MAX_ITERS}_${TS}"

echo "[nsys] profiling ${MOLECULE} NP=${NP} MAX_ITERS=${MAX_ITERS}"
echo "[nsys] output → ${REPORT}.nsys-rep"

# Env vars read by benchmarks/local_test_run.py — export before the mpirun
# subprocess so both ranks see them.
export MOLECULES="$MOLECULE"
export MAX_ITERS="$MAX_ITERS"

# --trace: what to record.
#   cuda      — CUDA API calls + kernels (the core of what we want).
#   nvtx      — hot-region annotations from src/api/interface.py (once wired).
#   mpi       — MPI calls, so scaling-limit diagnosis is possible from the
#               timeline. Requires OpenMPI's PMPI hooks — Lambda AMI has this.
#   osrt      — OS runtime (thread state); modest cost, useful for spotting
#               Python-GIL contention.
# --sample=cpu: also sample the CPU-side call stack at 1kHz. Cheap and lets
#               you spot Python overhead between kernel launches.
# --stats=true: print a summary table at the end so you don't need the GUI
#               to sanity-check the run before rsync-ing.
nsys profile \
    --trace=cuda,nvtx,mpi,osrt \
    --sample=cpu \
    --stats=true \
    --output "$REPORT" \
    --force-overwrite=true \
    mpirun -np "$NP" \
        python -m benchmarks.local_test_run

echo ""
echo "[nsys] done."
echo "[nsys] report: ${REPORT}.nsys-rep"
echo "[nsys] rsync back to laptop before terminating the instance:"
echo "  rsync -av <instance>:$(pwd)/${OUT_DIR}/ ${OUT_DIR}/"
echo "[nsys] open with: nsight-sys ${REPORT}.nsys-rep"
