# Post-audit TODO — 2026-09-07 sync

Mirror of the pending-items block in the paper repo's
`/home/annap/Desktop/THESIS/publish/qaas-publication-/NOTES.md`.
Update BOTH files when items land — the paper repo has more prose
context, this file is the code-side checklist.

The publication paper is a rework of the BCSAI thesis; large parts of the
prose read like the thesis and that's intentional (scaffold reuse). Do
NOT rewrite existing paper text just because it "sounds old" — only patch
where a data change or disclosure gap requires it.

## Status

- Live sweep: `results/2026-09-07-legacy-scaling-nh3-fixed` branch pending
  (created after Sept 7 A100 sweep rsyncs back). Cloud instance:
  Lambda A100-SXM4-40GB, `132.145.129.97`, container `vqe-mpi-gpu:latest`.
- Main branch state: has the `_best_physical_energy` reset fix (commit
  eabcaef, merged as PR #37).

## Code-side items — resolved

- [x] `_best_physical_energy` cross-molecule leak — commit 274ed44 →
      merged to main via PR #37. Every NH3 result JSON produced BEFORE
      this fix from a multi-molecule process is contaminated (silently
      reports the preceding molecule's energy).
- [x] `install_native.sh` doesn't work on plain Ubuntu Lambda AMI; the
      canonical path is `docker build -t vqe-mpi-gpu .` from `~/vqe`
      then run everything inside the container. Documented for future
      cloud sessions.

## Code-side items — pending

- [ ] **Sept 7 sweep completion + rsync + commit**. Once "ALL DONE"
      lands, rsync to `results/a100-sxm4-40gb-2026-09-07/`, commit on
      the results-branch above, PR to main.
- [ ] **scaling_P2.txt provenance**. Main branch has
      `T_total=0.4879, final_E=-3.561186` for `scaling_P2.txt`. July 27
      commit `08c5c4b` has `-5.001826`. Working tree today's
      `run_scaling_local` writes yet another value. Which one does the
      paper's Table 5 (`table:scaling_molecules` in `4_results.tex`)
      actually cite? Answer that before final submission, then commit
      the correct file to main and delete the other on the results
      branch or add a README explaining the divergence.
- [ ] **`Dockerfile` line 68**: `ENV IBM_QUANTUM_TOKEN=...` bakes the
      token into every image layer. Not blocking today (token has been
      rotated and image is only used locally on cloud instances), but
      before any `docker push` or public image publication, switch to
      build-time secret or runtime `-e IBM_QUANTUM_TOKEN=$IBM_QUANTUM_TOKEN`.
- [ ] **N2 legacy full-convergence** is not part of the Sept 7 sweep.
      Documented as "prohibitive single-CPU cost at P=1 legacy path."
      Only path to include N2 legacy in the paper is either (a) budget
      a dedicated overnight P=1 legacy run, or (b) run P≥2 only for N2
      and disclose no P=1 baseline. Neither is scheduled; the paper
      currently just excludes N2 from strong-scaling tables.

## Paper-side items — what the code changes require

See `qaas-publication-/NOTES.md` for the full checklist. High-level:

1. Experiment 2 (`3_experiments.tex:50`) — un-comment out the TODO,
   update `scaling_molecules` + `scaling_efficiency` tables in
   `4_results.tex`, add legacy-path disclosure sentence.
2. Experiment 4 (`3_experiments.tex:100`) — un-comment out the TODO,
   disclose GPU-native expectation path (Fix A) + the MPI regression
   that routes NP≥2 to legacy.
3. Experiment 1's `distributed_results` table (`4_results.tex:14`) —
   add NH3, N2 rows if we're extending Experiment 1 to 6 molecules;
   otherwise state explicitly that Experiment 1 covers 4 molecules
   with NH3/N2 relegated to Experiments 2/7.
4. Experiment 7 (`3_experiments.tex:172+`) — verify all Lightning
   numbers are from AFTER the 2026-09-03 ansatz-parity fix.

## Anything not in this list

...is not paper-blocking. Experiments 3, 5, 6 and all masking-metric
tables are unaffected by the recent bug fix and sweep.

---

## 2026-09-07 fact-check audit results (mirror)

Full details in the paper repo's `NOTES.md` under the *"2026-09-07
fact-check audit"* section. This mirror is code-side only — what the
stack itself needs to do for the paper to be defensible.

### Stack code claims — verified accurate

- `MPI_Ibcast` (C++) + `comm.Bcast`/`Allreduce` (Python) — real, in
  `src/dispatcher/dispatcher.cpp:154-155` and
  `src/api/interface.py:526-527`.
- C++ dispatcher local compute = mean-field approximation, NOT valid
  for entangled circuits — explicitly commented, and the paper's
  disclosure of this limitation is honest
  (`src/dispatcher/dispatcher.cpp:31,62,75`).
- `std::async` + REST client + exponential backoff for QPU polling —
  real, in `qpu_client.cpp:201-208`.
- Auto FP32/FP64 precision policy — real, in `hardware.py:154-183`.
- `adaptive_reps = min(reps+1, 3)` at `hwe_adaptive` tier and
  `entanglement="full"` for n≤20 — real, in `problems.py:85,91`.
- Serial baseline path parity (`ChemistryProblem` + `MoleculeResolver`)
  — real in code as of commit dcb8994. Note: paper Table 4 data
  (March 19 file) pre-dates this fix — see paper NOTES.md for options.

### Stack-side gaps that surface in the paper

- **NH3 and N2 have never appeared in a `corr_score=` log line** across
  all runs in `results/`. Paper claims "all 6 molecules corr_score
  0.34-0.50" but only 3 unique values are logged (0.342, 0.501, 0.502)
  and all are for H2/LiH/BeH2/H2O. To fix: run `local_test_run.py`
  once with MOLECULES="NH3 N2" and MAX_ITERS=1 to just get the tier-
  selection log lines, then quote real numbers.
- **Fix A (GPU-native expectation) is not disclosed anywhere in the
  paper's methodology.** Docs live at `docs/GPU_EXPECTATION_FIX.md`
  but paper reader has no way to know the stack has two paths and a
  runtime policy for choosing between them. See paper NOTES.md item
  20 for what to add.
- **GTX 1650 legacy data** — paper Table 8/9/10 (Results) cite this,
  but `results/gtx1650-reference/README.md` explicitly says the data
  is lost and excluded. Not blocking today; will need to be replaced
  when Results is rewritten.

### Paper prose issues (informational, code-side won't touch)

Typos and small drifts logged in the paper repo's NOTES.md — user will
fix in their next `.tex` edit pass.

### Nothing to do differently for today's sweep

The 2026-09-07 A100 legacy sweep in progress does not need to change
based on this audit. All it needs to do: complete, land clean NH3 data
(no cross-molecule contamination — bug is fixed on main), and produce
6-molecule × 4-P wall-clocks that then feed Experiment 2 + Experiment 4.
