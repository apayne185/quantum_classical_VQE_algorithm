# GPU-Native Expectation Fix (feature/gpu-expectation-fix)

**Status**: implemented + local correctness verified; awaiting cloud-GPU
wall-clock validation before merge.

**Motivation**: baseline comparison from the 2026-08-31 session showed
Pennylane Lightning-GPU winning wall-clock at BeH2 (1.61×) and H2O
(1.79×) vs HPCHybridStack on the same A100. Root cause identified in
`_evaluate_distributed_statevector`: after Aer built the statevector on
GPU, the code pulled the full 2^n array to CPU via `get_statevector`,
wrapped it in `qiskit.quantum_info.Statevector`, and computed the Pauli
expectation via numpy — a pure CPU code path. Lightning-GPU stayed on
GPU end-to-end. The **1.79× gap is entirely explained by the GPU→CPU
statevector copy + CPU-side expectation**.

## The fix

New method `_expectation_on_gpu(bound_plus, bound_minus, local_terms)`
in `src/api/interface.py`. Uses Aer's `save_expectation_value`
instruction so the Pauli expectation runs on the same GPU state the
circuit produced — no round-trip:

```python
local_op = SparsePauliOp.from_list(local_terms)
bound_plus.save_expectation_value(local_op, qubit_range, label="ev")
r_plus = sim.run(bound_plus).result()
e_plus_local = float(r_plus.data(0)["ev"])
```

`_evaluate_distributed_statevector` dispatches to the new path by
default when GPU is available, falls back to the old numpy path when:
- GPU is not available (`_gpu_sv` is False), OR
- The rank has zero local Pauli terms after partitioning, OR
- `VQE_LEGACY_EXPECT=1` is set (explicit A/B override)

The path taken is stamped into the per-iter log as
`GPU-cuStateVec+native-expect` vs `GPU-cuStateVec+cpu-expect` (or
`CPU-Statevector` when there's no GPU) so the log is self-diagnosing.

## Correctness verification (local, CPU)

Under Docker on the laptop, without GPU access, an A/B test on a
representative 4-qubit ansatz + fake H2 Hamiltonian:

```
OLD PATH (SV -> numpy):     E = -0.451491937550357
NEW PATH (save_expect):     E = -0.451491937550356
|delta|:                       5.55e-17  (machine epsilon)
```

The last-digit difference is from floating-point summation order in the
two implementations — the same reason SPSA already has stochastic
divergence between rank counts. Not a numerics regression.

## Wall-clock validation plan (cloud GPU, ~$1, ~15 min)

Run the same baseline sweep as 2026-08-30, but this time with the fix.
The critical comparison is against the committed baseline table under
`results/baseline_comparison/paper_table.md`.

### Step 1 — On the instance, in tmux

```bash
git fetch --all
git checkout feature/gpu-expectation-fix
make build   # ~5-10 min if the Aer-from-source layer needs rebuild
```

### Step 2 — Run hpchybrid with the new path (default)

```bash
for m in H2 LiH BeH2 H2O; do
    docker run --rm --gpus all \
        -v $(pwd)/results:/workspace/results \
        vqe-mpi-gpu \
        python3 -m benchmarks.baseline_comparison \
            --backend hpchybrid --molecule $m --max-iters 100 \
            --out-dir results/baseline_comparison_gpuexpect
done
```

### Step 3 — Compare wall times

Expected outcome (predictions to falsify):

| Molecule | Old hpchybrid (s) | Predicted new (s) | Predicted vs old |
|---|---:|---:|---:|
| H2   |  0.56 | 0.4–0.6  | flat (kernel-launch dominates at 4q)  |
| LiH  |  7.33 | 5.5–6.5  | ~15–25% faster |
| BeH2 | 12.13 | 7.5–9    | ~35% faster (matches or beats Lightning) |
| H2O  | 18.69 | 10.5–12  | ~40% faster (matches Lightning's 10.44s) |

If the H2O run comes in at ≤ 12s, the fix has succeeded — the paper
table becomes a clean win for hpchybrid on every non-trivial molecule.

### Step 4 — A/B against the legacy path

Sanity-check that `VQE_LEGACY_EXPECT=1` reproduces the old times:

```bash
docker run --rm --gpus all \
    -e VQE_LEGACY_EXPECT=1 \
    -v $(pwd)/results:/workspace/results \
    vqe-mpi-gpu \
    python3 -m benchmarks.baseline_comparison \
        --backend hpchybrid --molecule H2O --max-iters 100 \
        --out-dir results/baseline_comparison_legacy_check
```

Should reproduce ~18.7s H2O. If it does, we have proof the wall-clock
delta is entirely from the code-path change and not something else that
drifted on the instance (Docker rebuild, Aer version, etc.).

### Step 5 — Numeric parity check

Both runs must produce H2O energy within 1 mHa of each other at
seed=42, 100 iters. If they diverge by more than that, something is
wrong with the new path — do NOT merge.

## What to do based on the result

- **If Fix A works** (H2O ≤12s): merge to main, regenerate the paper
  table with the new hpchybrid numbers, and the paper narrative shifts
  from "Lightning wins at H2O" to "hpchybrid competitive across all
  4 canonical molecules, wins at BeH2/H2O with the GPU-native
  expectation path."
- **If Fix A partially works** (some improvement, not full parity):
  document the residual as a follow-up in `docs/FUTURE_WORK.md` and
  merge anyway — a 20% win is still a win.
- **If Fix A does not work** (no measurable improvement or numeric
  regression): do NOT merge. The `VQE_LEGACY_EXPECT` flag guarantees
  the fallback stays available; drop the branch and pursue Fix B
  (transpile caching) instead.

## IBM triple-integration validation (optional, ~$0 on open-plan)

The fix also applies to `_evaluate_ibm_estimator` — the classical
"T_accel" work done in parallel with QPU submission uses the same
build-SV + numpy-expectation pattern. Ported in the same commit with
the same env-flag semantics (`VQE_LEGACY_EXPECT=1` reverts both paths).

**Why this matters for the masking metric**: T_accel drops, T_comm
(QPU RTT, 32–60s) stays the same. The masking ratio M = T_accel /
T_comm was already 0.5–1.0 on the H2 IBM run per the thesis; with the
fix it drops further, meaning the classical work no longer fully masks
the QPU wait. This is a **paper-relevant finding** — either:
- Frame as "the improvement is only meaningful for larger molecules
  where T_accel remains substantial," or
- Extend the classical overlap to fill T_quant with additional useful
  work (e.g., statevector-based observables beyond the Hamiltonian).

**IBM validation (only after the simulator validation above passes)**:
```bash
# On the instance, in tmux -- IBM QPU can take 30-60s per iter:
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/.env:/workspace/.env \
    vqe-mpi-gpu \
    python3 benchmarks/ibm_test_run.py
```

Expected: 10-iter H2 run completes (matches prior IBM triple-integration
runs). Per-iter masking metric M should be visible in the log; compare
against `results/ibm/ibm_cloud_20260727_220130.json`'s M values to
quantify how much T_accel dropped.

Free-tier caveat: IBM open plan allows one 10-min slot per month; use
this validation carefully. If the simulator validation looks good, this
becomes optional.

## Follow-up items unblocked by this fix

- Nsight re-profile: with the GPU-native path, the timeline will show
  cuStateVec kernels dominating instead of the CPU-side numpy call —
  makes the T_accel/T_comm story cleaner.
- CO2 retry: this fix does NOT solve CO2 (the bandwidth-bound
  expectation at 30q remains the dominant cost), but the reduced
  Python-side overhead should shave meaningful time off each iteration.
  Worth one more `timeout 1200` attempt after this fix lands.
- Hot-path optimizations 7.1 and 7.2 in `docs/FUTURE_WORK.md` become
  the natural next round of wall-clock improvements if Fix A validates.
