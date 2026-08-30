"""Baseline comparison entry point: HPCHybridStack vs Pennylane Lightning-GPU
vs Qiskit Aer MPI on the same VQE problem.

Design and rationale: docs/BASELINE_COMPARISON.md.

Only the `hpchybrid` backend is fully wired today. The `lightning` and
`aer-mpi` paths are deliberate stubs — they raise NotImplementedError with
the concrete next step so this file locks in the CLI contract before the
cloud-GPU run.

Usage (from repo root, inside the container/env):
    python -m benchmarks.baseline_comparison --backend hpchybrid \
        --molecule H2 --max-iters 100 --seed 42
    mpirun -np 2 python -m benchmarks.baseline_comparison \
        --backend lightning --molecule LiH --max-iters 100
"""

import argparse
import os
import sys
import time
from datetime import datetime

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "build"))

BACKENDS = ("hpchybrid", "lightning", "aer-mpi")
DEFAULT_MOLECULES = ("H2", "LiH", "BeH2", "H2O")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--molecule", default="H2",
                   help="Single molecule ID (H2 / LiH / BeH2 / H2O). "
                        "Loop externally for the full set.")
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", action="store_true",
                   help="Run one throwaway H2 iteration to prime caches "
                        "before the timed run.")
    p.add_argument("--out-dir", default="results/baseline_comparison")
    return p.parse_args()


def _run_hpchybrid(molecule, max_iters, seed, out_dir):
    """Time HPCHybridStack (Aer-GPU, per-rank replicated statevector).

    Mirrors benchmarks/local_test_run.py's flow but strips the multi-molecule
    loop and stamps a `baseline_backend` field so aggregation does not
    collide with real HPCHybridStack runs.
    """
    from src.api.interface import HPCHybridStack
    from src.api.molecule_resolver import MoleculeResolver
    from src.api.results import save_results
    from src.api.hardware import HardwareProfile

    resolver = MoleculeResolver(max_qubits=30, allow_network=True,
                                cache_dir=".pubchem_cache")
    problem = resolver.resolve(molecule, freeze_core=True).to_chemistry_problem()
    problem.prepare()

    hw = HardwareProfile.detect()
    stack = HPCHybridStack(hw=hw)

    t0 = time.perf_counter()
    result = stack.run(problem, max_iters=max_iters, seed=seed)
    wall = time.perf_counter() - t0

    payload = {
        "baseline_backend": "hpchybrid-aer-gpu",
        "molecule": molecule,
        "max_iters": max_iters,
        "seed": seed,
        "wall_seconds": wall,
        "wall_per_iter": wall / max_iters,
        "energy_ha": getattr(result, "energy", None),
        "iterations": getattr(result, "iterations", max_iters),
    }
    _save(payload, out_dir, "hpchybrid-aer-gpu", molecule)
    return payload


def _run_lightning(molecule, max_iters, seed, out_dir):
    """Pennylane LightningGPU + Lightning-MPI path.

    Deliberate stub — the ansatz has to be ported to Pennylane templates
    matching our HWE tier gate-for-gate before this runs. See
    docs/BASELINE_COMPARISON.md "Open questions" for the ansatz parity note.
    """
    raise NotImplementedError(
        "Lightning-GPU path not yet wired. Next step:\n"
        "  pip install pennylane pennylane-lightning-gpu pennylane-lightning[mpi]\n"
        "  verify: mpirun -np 2 python -c 'import pennylane as qml; print(qml.about())'\n"
        "then port the HWE ansatz from src/api/ansatz_builder.py to a Pennylane\n"
        "template. See docs/BASELINE_COMPARISON.md."
    )


def _run_aer_mpi(molecule, max_iters, seed, out_dir):
    """Qiskit Aer with distributed-statevector mode (blocking_enable=True).

    Deliberate stub — needs an Aer configuration path separate from the one
    HPCHybridStack uses today. Distinct AerSimulator instance with
    blocking_enable=True and blocking_qubits set appropriately for the
    problem size.
    """
    raise NotImplementedError(
        "Aer MPI distributed path not yet wired. Next step:\n"
        "  verify: python -c \"from qiskit_aer import AerSimulator; \\\n"
        "    AerSimulator(method='statevector', device='GPU', blocking_enable=True); \\\n"
        "    print('ok')\"\n"
        "then either add a `distributed_statevector=True` flag to HPCHybridStack\n"
        "or build a parallel entry point that constructs Aer directly. See\n"
        "docs/BASELINE_COMPARISON.md."
    )


_DISPATCH = {
    "hpchybrid": _run_hpchybrid,
    "lightning": _run_lightning,
    "aer-mpi": _run_aer_mpi,
}


def _save(payload, out_dir, backend_slug, molecule):
    os.makedirs(os.path.join(out_dir, backend_slug), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, backend_slug, f"{molecule}_{ts}.json")
    import json
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[baseline] wrote {path}")


def main():
    args = _parse_args()

    if args.warmup and args.molecule != "H2":
        print("[baseline] running warmup on H2 before timed run...")
        try:
            _DISPATCH[args.backend]("H2", max_iters=5, seed=0,
                                    out_dir=os.path.join(args.out_dir, "_warmup"))
        except NotImplementedError as e:
            print(f"[baseline] warmup skipped ({e.__class__.__name__})")

    print(f"[baseline] backend={args.backend} molecule={args.molecule} "
          f"iters={args.max_iters} seed={args.seed}")
    result = _DISPATCH[args.backend](args.molecule, args.max_iters,
                                     args.seed, args.out_dir)
    print(f"[baseline] done: wall={result['wall_seconds']:.2f}s "
          f"({result['wall_per_iter']:.4f}s/iter) "
          f"E={result.get('energy_ha')}")


if __name__ == "__main__":
    main()
