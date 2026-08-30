# Known Gaps

Findings from the 2026-08-06 "is this stack coordinated, organized, prod-ready" repo
review. Roughly priority order. Items marked **FIXED** have been resolved since;
everything else is open and needs a decision before the next round of work.

Cross-references: `docs/FUTURE_WORK.md` covers the deferred *feature* work
(UCCSD, distributed statevector, error mitigation, QPU plugin system); this file
covers *maintenance* debt.

---

## A. C++/CUDA dispatcher computes wrong energies if ever used

`src/classical/gpu/dispatcher.cpp` / `kernel.cu` use a mean-field product-state
approximation (`⟨X⟩=⟨Y⟩=sin(θ)`), not physically self-consistent for entangled
circuits. Currently harmless — `vqe_optimize()` only ever branches to
`backend="simulator"` / `"ibm_cloud"`, never to the C++ dispatcher. It is dead
code in every real run, only reached by the non-assertive Layer-3 diagnostic test.
**Landmine** if anyone ever refactors the backend branching without fixing the
physics first.

## B. Dependency drift — PARTIALLY FIXED

Fixed 2026-08-06: `qiskit-aer` added to `requirements.txt`, `matplotlib`/`rdkit`
added to `Dockerfile`. **Not build-tested** — Docker daemon wasn't running when
the change was made; confirm the next `make build` still succeeds.

Still open:
- Dockerfile pip installs are otherwise completely unpinned (non-reproducible
  builds over time — contradicts the README's reproducibility claims).
- `pyscf` / `qiskit-nature` version floors still disagree between
  `requirements.txt` and `environment.yml`.
- `qiskit-algorithms` still declared in `environment.yml` but never imported.

## C. `.git` is 155 MB

Almost entirely a 156.8 MB Miniconda installer committed and later deleted but
never purged from history. Fix needs `git filter-repo` / BFG — a **history
rewrite** that invalidates every existing clone. Not touched; needs explicit
go-ahead (and a broadcast to anyone else who has cloned).

## D. CodeQL only scans Python

The C++/CUDA layer (manual `cudaMalloc`, a curl-based REST client) gets zero
static analysis, despite being the part most likely to have memory-safety issues.

## E. Silent CUDA kernel-failure swallowing

`src/classical/gpu/kernel.cu:130-137`: a kernel-launch failure is logged to
stderr but the function still returns as if it succeeded (typically `0.0`) — no
exception, no error code back to Python. Inert today since gap A means this path
is unused, but the pattern is wrong.

## F. `src/classical/cpu/kernel_stub.cpp` — orphaned function

Defines a never-called function (dead code from an earlier refactor). Safe to
delete.

## G. `docs/notes/notes-docker.txt` — stray key output

Gitignored, so not a real hygiene issue. Has stray local GPG/SSH keygen terminal
output. Fine to delete locally, not thesis-relevant.

## H. Distributed statevector — the big unfixed speedup lever

Every MPI rank independently builds the **full** 2ⁿ statevector; only the
Pauli-term *summation* is distributed, not statevector construction itself
(explicit in the thesis methodology text). This is exactly why RTX 6000 shows
flat/negative strong scaling and why A100's scaling plateaus rather than keeps
climbing. Real fix = a genuinely distributed statevector via cuStateVec's
multi-GPU/multi-node API — a rearchitecture, not a patch. See
`docs/FUTURE_WORK.md` for the full write-up (this is the leading paper follow-up
item).

## I. Two cheap `_build_statevector` speedup wins — FIXED

2026-08-06: (1) `sim.set_options(precision=...)` hoisted from per-evaluation to
per-`vqe_optimize()` call. (2) Redundant `bound_circuit.copy()` removed — every
real caller passes a circuit fresh from `assign_parameters()`. Deliberately did
**not** apply to `problem.ansatz_circuit` itself (the shared reused template):
that would leak the Aer-only `save_statevector` instruction into the IBM
transpilation path and likely break real QPU submission. Untested for actual
speedup impact — confirm on the next GPU session.

## J. Chemical accuracy: SPSA on noiseless simulator is a poor default

SPSA is used even on the noiseless simulator backend, where its noise-robustness
earns nothing and its slower/noisier convergence directly costs accuracy — LiH
(336 mHa) and H2O (220 mHa) median errors look like SPSA convergence artifacts,
not fundamental ansatz limits.

Proposed: an exact/analytic-gradient optimizer specifically for
`backend="simulator"` (keep SPSA for the IBM QPU path, where noise-robustness is
real). Lighter middle ground than full UCCSD: a particle-number-conserving HWE
variant (Givens-rotation gates) — avoids UCCSD's tied-parameter /
QASM-decompose problem entirely, since there's no shared-parameter structure to
break.

## K. UCCSD — tabled for a future trial

Root blocker traced 2026-08-06: it's **not** the QASM/C++ dispatcher path
(unreachable for `simulator` / `ibm_cloud` backends anyway, see gap A) — no log
anywhere in the repo shows `"UCCSD built"` ever printing, meaning it's most
likely never been run end-to-end at all.

Structurally: auto-tier-selection never routes any benchmarked molecule there
(all score 0.34–0.50, below the 0.55 threshold), and it's hard-capped to ≤12
qubits regardless. `force_tier="uccsd"` on H2 (backend=simulator) is the
concrete next step. Deeper write-up in `docs/FUTURE_WORK.md`.

## L. `benchmarks/run_analysis.py` missing `--hw` guard — FIXED

2026-08-06: same `--hw` guard as the two `aggregate_*.py` scripts added. While
adding it, found `_pick_latest_sim()` had the *identical* molecule-clobbering
bug already fixed in the aggregation scripts (`candidates[-1]`, no
molecule-count awareness) — fixed the same way. This was live and wrong for
every plot function (convergence, wall-time, accuracy, speedup-by-molecule)
until the fix.

## M. Zero test coverage

No coverage for:
- `src/api/results.py`
- `src/api/log.py`
- `FinanceProblem` (test explicitly commented out)
- The C++/CUDA layer beyond the non-assertive Layer-3 check
- `_evaluate_ibm_estimator` (only exercised by the manual-`workflow_dispatch`
  CI job)

---

## Results layout — quick reference

Where each hardware's raw data lives (all under `results/`):

- `rtx-6000-ada-generation/` — April, university cluster, complete P∈{1,2,4,8}
  strong+weak scaling, no seed sweep (predates that feature), base 4-molecule
  set only.
- `a100-sxm4-40gb/` — July–Aug, Lambda Cloud, complete P∈{1,2,4,8} at seed=42
  plus seeds 42–46 at P=2, NH3/N2 single runs. CO2 attempted and cancelled
  (>1hr, never finished ansatz setup — dropped from scope, root cause found
  and fixed but not retried yet).
- `cpu-only/` — serial baseline plus old CPU-only dev/IBM runs.
- `gtx1650-reference/` — README only, no raw data (never committed, presumed
  lost; the thesis PDF §4.4.6, §4.7 is the only surviving source). Excluded
  from the published GPU comparison by design — it was the preliminary trial,
  not part of the final result set.
