# Paper Alignment — QAAS Journal Submission

Maps every claim the QAAS-middleware paper will make to the artifact that
backs it. Prevents scope drift between "what the thesis showed" and "what
the paper will show," and pins down what still needs re-running vs what
is already committed.

**Framing shift**: this work was originally the BCSAI thesis
"Hybrid Quantum-Classical Software Stack for HPC Algorithm Acceleration"
(Payne, IE University, 2026), evaluated on a GTX 1650 with n=1 seed. The
journal submission repositions it as **Quantum-as-a-Service (QAAS) middleware**
— reusable infrastructure other researchers adopt. That reframe promotes
three thesis limitations to first-class paper features (below) and demotes
one thesis contribution (chemistry accuracy, which was already tabled as
future work in the thesis's own §7).

Cross-refs: `docs/RELATED_WORK.md`, `docs/BASELINE_COMPARISON.md`,
`docs/FUTURE_WORK.md`, `docs/KNOWN_GAPS.md`.

---

## Framing shift — thesis → QAAS journal

| Concern | Thesis frame | QAAS journal frame |
|---|---|---|
| **Primary contribution** | Novel triple-heterogeneous execution model | Reusable middleware other researchers adopt unchanged |
| **Success criterion** | 1.5× wall-clock vs serial (achieved: 1.51×) | Vendor-portability + baseline-competitive + reproducible |
| **Chemistry accuracy** | 1.6 mHa target (partial: 0.13 mHa best-seed) | Explicitly out-of-scope — orthogonal to middleware |
| **Ansatz choice** | HWE limitation is a paper limitation | HWE is one plugin option; UCCSD/others in ansatz tier system |
| **Hardware coverage** | GTX 1650 (thesis) | GTX 1650 + RTX 6000 Ada + A100 (validated); AWS A10G (planned) |
| **Seed count** | n=1 (seed=42), fixed | n=5 canonical, n=10 for LiH — statistical methodology built in |
| **Comparison targets** | Serial Qiskit only | Serial Qiskit + Pennylane Lightning + Qiskit Aer MPI + openQSE-surveyed stacks |

The **middleware contributions are orthogonal to ansatz + chemistry accuracy**
(the exact framing already used at the end of `docs/FUTURE_WORK.md` §1).
The paper leans into that: contribution is the QAAS layer, not the
chemistry science.

---

## Priority order for the QAAS paper (revised)

Elevated from the thesis's original future-work list because they matter
more when the audience is other middleware adopters, not chemistry
readers:

1. **Distributed statevector** (was gap H / thesis future-work item 3).
   Newly promoted to a *near-term* item — the Aer-MPI baseline
   (`benchmarks/baseline_comparison.py --backend aer-mpi`, with
   `blocking_enable=True`) now **directly measures the competitive
   penalty** of not having this. The paper's honest positioning is:
   *"HPCHybridStack achieves competitive wall-clock at ≤14 qubits vs
   distributed-statevector baselines; scaling ceiling is bounded by the
   replicated-SV design, and a distributed-SV rearchitecture via
   cuStateVec multi-GPU is the leading follow-up work
   (`docs/FUTURE_WORK.md` §2)."*
2. **Baseline comparison table** vs Pennylane Lightning + Qiskit Aer MPI.
   Anchor of the related-work section.
3. **Multi-cloud validation** on AWS A10G. One reproduced seed=42 sweep
   from a different vendor's silicon = the vendor-portability claim.
4. **Nsight profiling** for T_accel/T_comm attribution — kernel-level
   confirmation of the masking metric already computed per-iteration in
   `interface.py`.

Deferred (still important, still in `docs/FUTURE_WORK.md`, not blockers):

- UCCSD chemistry accuracy
- ZNE/PEC error mitigation
- QPU backend plugin system (Braket, IonQ, Azure)
- Multi-node MPI on InfiniBand

---

## Claim-to-artifact map

Every paper claim → concrete artifact that reproduces it. Broken into
**Committed** (data already exists), **Locally regenerable** (aggregators
run on committed JSONs), and **Needs cloud GPU** (has to be produced in
the next session).

### §Contribution — "auto-detected CPU/GPU/QPU triple integration"

| Claim | Artifact | Status |
|---|---|---|
| Same Python code runs on laptop, cloud GPU, IBM QPU unchanged | `src/api/interface.py` + `src/api/hardware.py` (`HardwareProfile.detect()`) | Committed |
| End-to-end triple integration proven on real QPU | `results/ibm/ibm_cloud_20260727_220130.json` | Committed |
| Sub-1000 LOC middleware; no fork of Qiskit or Aer | Repo LOC count + `README.md` architecture section | Committed |

### §Scaling — "A100 real strong scaling, RTX 6000 flat/negative"

| Claim | Artifact | Status |
|---|---|---|
| A100 H2O 3.77× at P=8 | `results/a100-sxm4-40gb/scaling/*.json` | Committed |
| RTX 6000 flat/negative strong scaling | `results/rtx-6000-ada-generation/scaling/*.json` | Committed |
| Cross-hardware speedup at matched P=8: H2O 2.83×, BeH2 3.28× | `benchmarks/aggregate_scaling.py --hw a100-sxm4-40gb` and `--hw rtx-6000-ada-generation` | Locally regenerable |
| Masking metric M > 1 across all simulator P | `results/*/simulator/*.json` `masking_metric` field | Committed |

### §Statistical robustness — "best-of-N hits chemical accuracy"

| Claim | Artifact | Status |
|---|---|---|
| Best-of-5 seeds: H2 0.2 mHa, BeH2 0.7 mHa | `benchmarks/aggregate_seeds.py --hw a100-sxm4-40gb` best-of-N summary | Locally regenerable |
| Median ± range table across 5 canonical seeds | Same aggregator, median block | Locally regenerable |
| LiH bimodal spread (n=10 → 492 mHa median, 0.6 mHa best-of-10) | Existing n=5 → **need n=10** | **Needs cloud GPU** (5 more LiH seeds) |

### §Related work + Baseline comparison — new for the journal paper

| Claim | Artifact | Status |
|---|---|---|
| Positioning vs openQSE reference architecture | `docs/RELATED_WORK.md` | Committed |
| Positioning vs JHPC-Quantum, Qristal, Tierkreis, Lightning, Aer MPI | `docs/RELATED_WORK.md` | Committed |
| Baseline table: hpchybrid vs Lightning-GPU vs Aer MPI (H2/LiH/BeH2/H2O) | `benchmarks/baseline_comparison.py` + `aggregate_baseline.py` | **Needs cloud GPU** (CLI + aggregator done, dry-run OK locally on CPU) |
| Distributed-SV competitive gap quantified | Aer-MPI wall-clock vs hpchybrid wall-clock at H2O (same table above) | **Needs cloud GPU** |

### §QPU integration — "one full pipeline exercise"

| Claim | Artifact | Status |
|---|---|---|
| Full triple-integration executes end-to-end without failure | `results/ibm/ibm_cloud_20260727_220130.json` (CPU+GPU+QPU) | Committed |
| PUB batching halves QPU round trips | `interface.py:_evaluate_ibm_estimator` (2 PUBs per iter) | Committed |
| Downward-trending energy on real QPU (H2, 10 iters) | Same JSON, `history` field | Committed |
| ZNE/PEC integration | Deferred to `docs/FUTURE_WORK.md` §3 | Not a paper claim |

### §Resilience — thesis Experiment 6 (paper differentiator)

| Claim | Artifact | Status |
|---|---|---|
| Checkpoint restart from mid-run failure | `tests/test_layers_run.py` layer 7 + committed checkpoint files | Committed |
| QPU latency spike (0.5–2s asymmetric delays) does not deadlock MPI | Same, layer 6 | Committed |
| Recovery time < 60s | Same, timing field | Committed |
| Distinguishing feature vs Lightning/Aer MPI (neither has checkpointing) | `docs/RELATED_WORK.md` "resilience" row in comparison table | Should add — currently missing from RELATED_WORK.md |

### §Multi-cloud portability — vendor-lock-in claim

| Claim | Artifact | Status |
|---|---|---|
| Same code, same seed=42, agrees to <1 mHa across Lambda A100 and AWS A10G | `scripts/aws_deploy.sh` + resulting `results/g5-a10g/simulator/*.json` | **Needs cloud GPU** (script wired, cleanup deliberate) |

### §Nsight profiling — attribution of T_accel/T_comm

| Claim | Artifact | Status |
|---|---|---|
| Per-iteration time breakdown: SV build / Pauli sum / MPI reduce | `scripts/nsys_profile.sh` + resulting `.nsys-rep` | **Needs cloud GPU** (script + Dockerfile.profiling + NVTX ranges done) |
| Empirical M = T_accel / T_comm matches per-iter simulator M values | Nsight timeline vs `masking_metric` field in `results/*/simulator/*.json` | **Needs cloud GPU** |

---

## What needs cloud GPU vs what is already provable now

**Provable NOW locally (no GPU, no spend):**
- All committed A100 + RTX 6000 + IBM data — regenerate all tables via
  `aggregate_seeds.py`, `aggregate_scaling.py`, `run_analysis.py`.
- All related-work + positioning content in `docs/`.
- Reproducibility claim via `make trial NP=2` (Docker CPU fallback) —
  passes 7/7 on any laptop.
- Statistical methodology (best-of-N + median±range) on existing JSONs.

**Needs the next cloud-GPU session (~$10, ~2 hours):**
1. Retry CO2 with the **new safety recipe**:
   `VQE_PRECISION=fp32 MOLECULES=CO2 MAX_ITERS=10 NP=1 make run` —
   validates the pre-flight warning added `interface.py` and the
   `reps=1` registry fix.
2. Baseline comparison: 3 backends × 4 molecules × 1 seed. Populates
   `results/baseline_comparison/` for `aggregate_baseline.py`.
3. Nsight profile: one H2O 10-iter run under `scripts/nsys_profile.sh`.
4. AWS multi-cloud spot check: `scripts/aws_deploy.sh` + one seed=42
   sweep on g5.xlarge.
5. Optional: 5 additional LiH seeds to complete the n=10 bimodal
   demonstration (thesis had n=10 for LiH; A100 currently only has n=5).

---

## Sections that are ready to draft now

The paper can be substantively started this week using only committed
artifacts + docs/. Section-by-section readiness:

- **Abstract** — draft; leave the "3-backend baseline" sentence pending
- **1. Introduction + related work** — ready (`docs/RELATED_WORK.md`)
- **2. Architecture** — ready (`README.md` architecture section + `docs/API.md`)
- **3. Methodology** — ready (thesis §3 + updates for auto-detection layer)
- **4. Results — scaling** — ready (A100 committed data)
- **4. Results — accuracy** — ready (best-of-5 numbers)
- **4. Results — QPU** — ready (IBM triple-integration JSON)
- **4. Results — baseline comparison** — pending cloud GPU
- **4. Results — profiling** — pending cloud GPU
- **4. Results — multi-cloud portability** — pending cloud GPU
- **5. Discussion + limitations** — ready (`docs/KNOWN_GAPS.md` + honest gap H framing)
- **6. Future work** — ready (`docs/FUTURE_WORK.md`, priority reordered per §above)
- **7. Conclusion** — draft alongside abstract
