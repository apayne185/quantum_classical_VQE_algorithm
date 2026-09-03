# Lightning results — INVALIDATED pending rerun (2026-09-03)

All Lightning JSONs in this directory were produced with an
ansatz-parity bug in `benchmarks/baseline_comparison.py::_lightning_ansatz`:

- Used `entanglement="linear"` instead of the Qiskit path's `"full"`
- Used raw `problem.reps` instead of `adaptive_reps = min(reps+1, 3)`

Combined effect: Lightning ran on a ~25% smaller ansatz (fewer params)
with ~85% fewer entangling gates than hpchybrid at n=14. Every
wall-clock number here is measured on an easier problem than the
hpchybrid comparison side of the same table.

**Do not cite these Lightning numbers in the paper.** Rerun with the
fixed code in `feature/gpu-expectation-fix` after 2026-09-03 commit,
then delete this README.

The hpchybrid/ and aer-mpi/ directories in the sibling paths are
unaffected — the bug was Lightning-side only. Their JSONs remain
authoritative.
