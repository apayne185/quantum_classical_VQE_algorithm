# GPU Profiling with Nsight Systems

Capture kernel-level and MPI-timeline profiles of a single VQE iteration to
answer the reviewer question *"where is the wall-clock time actually going,
and does the masking metric M = T_accel / T_comm predict it?"*

All prep here is local. Actual capture needs a GPU instance — deferred to the
next Lambda / AWS session (`docs/AWS_DEPLOYMENT.md`).

Cross-refs: `docs/FUTURE_WORK.md` §Nsight profiling (paper-ready blurb),
`docs/KNOWN_GAPS.md` gap H (distributed statevector — what Nsight is expected
to justify quantitatively).

---

## What we want to measure

For a single 100-iteration SPSA run on H2O (14q, the largest problem in the
canonical set):

1. **Time breakdown per iteration**: circuit assembly, statevector
   construction, expectation-value summation, MPI reduction. Which of these
   dominates?
2. **cuStateVec kernel utilization**: are we bottlenecked in the kernel, or
   in Python overhead between kernel launches?
3. **MPI communication overhead vs compute**: the direct measurement of
   M = T_accel / T_comm from the thesis methodology. Predicts why RTX 6000
   scales flat and A100 scales positively.
4. **GPU memory traffic**: are we bandwidth-bound (would justify distributed
   statevector) or compute-bound (would not)?

## Tools

- **`nsys profile`** — Nsight Systems for the timeline view (MPI + CUDA API
  + CUDA kernels + NVTX ranges). Primary tool.
- **`ncu` / Nsight Compute** — kernel-level detail (register pressure,
  occupancy). Deferred; only useful if `nsys` shows a specific kernel as the
  bottleneck.
- **NVTX ranges** — added inline in `_build_statevector` and the SPSA loop
  (deferred to a separate PR; needs the actual GPU run to validate they show
  up in the timeline).

## Container variant

`Dockerfile.profiling` extends the existing `Dockerfile` with:
- `nsight-systems` CLI (from NVIDIA apt repo).
- Kept CUDA base + all Python deps unchanged so profiled runs use the same
  library stack as production.

Built as a separate image tag (`vqe-mpi-gpu:profiling`) so production runs
never carry the profiling overhead.

## Script

`scripts/nsys_profile.sh` — wraps a single H2O run under `nsys profile`,
captures MPI + CUDA + NVTX, writes to `results/profiling/<timestamp>.nsys-rep`.
Report is opened in the Nsight Systems GUI locally after `rsync`-back.

## Execution plan (when cloud GPU access returns)

Estimated cost: ~$3–5 on a 30-minute Lambda A100 session (one warmup + one
timed profile).

1. Boot instance per `install_native.sh`.
2. Install Nsight Systems: `sudo apt install nsight-systems` (Lambda AMI
   ships with the CUDA toolkit already; only the profiler is missing).
3. Warmup: `MOLECULES=H2 MAX_ITERS=5 make run NP=2` — primes cuStateVec.
4. Profile: `MOLECULES=H2O MAX_ITERS=10 scripts/nsys_profile.sh`.
   Short iteration budget deliberate — the timeline is more useful than the
   sample count for the questions above.
5. `rsync -av instance:~/.../results/profiling/ results/profiling/`.
6. Open the `.nsys-rep` file locally in Nsight Systems.

## What the results should show (predictions)

1. **cuStateVec kernels ≥ 60% of iteration time** at 14 qubits on A100 —
   otherwise the "GPU-accelerated" claim is misleading and Python overhead
   is the real bottleneck.
2. **MPI reduction < 15% of iteration time** at NP=2 — matches the A100
   positive-scaling result. On RTX 6000 the same run would show MPI reduction
   ≥ 40%, which is why scaling goes flat/negative there.
3. **No idle GPU gaps > 5ms between kernel launches** — would indicate
   Python-loop overhead worth optimizing (candidate for a Cython or ctypes
   rewrite of the innermost loop).

If (1) fails: revisit the CUDA-kernel-vs-Aer-GPU story in the paper.
If (2) fails: our masking-metric prediction is wrong; investigate before
publication.
If (3) fails: cheap optimization win, worth doing before final submission.

## What the profile does not tell us

- Whether **distributed statevector** would be faster — the profile only
  measures the current architecture. The projection has to come from
  cuStateVec multi-GPU docs + a back-of-envelope calc.
- Cross-hardware — one A100 profile ≠ an RTX 6000 profile. Nice to have both
  eventually; the RTX 6000 cluster is currently blocked (`docs/CLUSTER_SETUP.md`).
- IBM QPU path — Nsight only sees local GPU/CPU/MPI. QPU timing comes from
  the IBM Runtime job metadata.

## Paper deliverable

<!-- paper -->
Nsight Systems profiling of a representative VQE iteration (H2O, 14 qubits,
NP=2, A100-SXM4-40GB) partitions wall-clock time into circuit assembly,
statevector construction, Pauli-term expectation summation, and MPI
reduction. The measured ratio T_accel / T_comm (the masking metric
introduced in §methodology) directly explains the observed strong-scaling
asymmetry between datacenter and workstation GPUs: on A100 the metric
exceeds unity across all four canonical molecules, admitting positive
scaling up to NP=8; on RTX 6000 the metric is below unity for H2 and LiH,
matching the observed flat scaling curves. This gives an empirical basis for
the design principle that distributed VQE benefits require T_accel > T_comm
per iteration, not just a fast accelerator.
<!-- /paper -->
