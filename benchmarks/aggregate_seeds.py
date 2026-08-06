"""Aggregate multi-seed VQE result JSONs into median ± IQR statistics.

Scans results/simulator/ for JSON files that have a "seed" field, groups by
molecule, and reports median/min/max across seeds for energy + wall time.

Run:
    python benchmarks/aggregate_seeds.py
    python benchmarks/aggregate_seeds.py --backend simulator --since 2026-06-23
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
from glob import glob


def load_seeded_results(backend: str, since: str | None,
                        ranks: int | None, hw: str | None = None) -> list[dict]:
    """Load seeded JSON result files; optionally filter by mpi_ranks.

    Deduplicates by (seed, mpi_ranks), keeping the most recent — guards against
    accidental contamination when scaling sweeps reuse SEED=42 default and
    produce JSONs at multiple P values.
    """
    pattern = os.path.join("results", hw or "*", backend, f"{backend}_*.json")
    files = sorted(glob(pattern))
    candidates = []
    for path in files:
        try:
            with open(path) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            print(f"[skip] {path}: invalid JSON", file=sys.stderr)
            continue
        if "seed" not in d:
            continue
        if since and d.get("timestamp", "") < since:
            continue
        if ranks is not None and d.get("mpi_ranks") != ranks:
            continue
        d["_path"] = path
        d["_hw_slug"] = path.split(os.sep)[1]
        candidates.append(d)

    # Dedup by (seed, mpi_ranks) - keep the most complete run (most molecules
    # covered, most recent as tiebreaker). A pure "most recent" rule silently
    # drops the real multi-molecule sweep whenever the same (seed, P) pair
    # gets reused later for an unrelated single-molecule probe (e.g. an
    # N2-only run at seed=42, P=2 clobbering the actual H2/LiH/BeH2/H2O
    # seed=42 run) -- same failure mode as aggregate_scaling.py's best_by_rank.
    by_key = {}
    for d in candidates:
        key = (d["seed"], d.get("mpi_ranks"))
        current = by_key.get(key)
        if current is None:
            by_key[key] = d
            continue
        n_mols = len(d.get("molecules", {}))
        current_n_mols = len(current.get("molecules", {}))
        if (n_mols, d.get("timestamp", "")) > (current_n_mols, current.get("timestamp", "")):
            by_key[key] = d
    return list(by_key.values())


def aggregate(runs: list[dict]) -> dict[str, dict]:
    """Group by molecule, collect (seed, energy, wall_time, iters) tuples.

    Supports two JSON shapes:
      - simulator runs: top-level "molecules" dict keyed by name
      - ibm runs: top-level "chemistry" with a "molecule" field naming the species
    """
    by_mol: dict[str, list[dict]] = {}
    for run in runs:
        seed = run["seed"]
        # Simulator shape
        for mol, data in run.get("molecules", {}).items():
            by_mol.setdefault(mol, []).append({
                "seed": seed,
                "energy": data["energy"],
                "fci": data.get("fci"),
                "wall_time": data.get("wall_time"),
                "iters": data.get("iters"),
            })
        # IBM shape — single chemistry record per run
        chem = run.get("chemistry")
        if chem and isinstance(chem, dict) and "energy" in chem:
            mol = chem.get("molecule", "H2")
            by_mol.setdefault(mol, []).append({
                "seed": seed,
                "energy": chem["energy"],
                "fci": chem.get("fci"),
                "wall_time": chem.get("wall_time"),
                "iters": chem.get("iterations"),
            })
    return by_mol


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    """Return (median, min, max) — IQR is min/max for small n."""
    s = sorted(values)
    return (statistics.median(s), s[0], s[-1])


def report(by_mol: dict[str, list[dict]]) -> None:
    print(f"\n{'Molecule':<8} {'n':<3} {'Seeds':<20} "
          f"{'Median E (Ha)':<16} {'E range':<22} "
          f"{'Median |err| (Ha)':<18} "
          f"{'Median T (s)':<14}")
    print("-" * 110)

    for mol, runs in sorted(by_mol.items()):
        seeds = sorted({r["seed"] for r in runs})
        energies = [r["energy"] for r in runs]
        fci = next((r["fci"] for r in runs if r["fci"] is not None), None)
        wts = [r["wall_time"] for r in runs if r["wall_time"]]

        med_e, min_e, max_e = median_iqr(energies)
        e_range = f"[{min_e:.4f}, {max_e:.4f}]"

        if fci is not None:
            errs = [abs(e - fci) for e in energies]
            med_err = statistics.median(errs)
            err_str = f"{med_err:.4f}"
        else:
            err_str = "N/A"

        med_t = statistics.median(wts) if wts else 0.0

        print(f"{mol:<8} {len(runs):<3} {str(seeds):<20} "
              f"{med_e:<16.6f} {e_range:<22} "
              f"{err_str:<18} {med_t:<14.2f}")

    print("\nNotes:")
    print(" - Median is taken across SPSA random seeds at fixed hyperparameters.")
    print(" - E range is [min, max] across seeds; reportable as median ± half-range.")
    print(" - For paper, this corresponds to 'n=N independent SPSA trajectories'.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="simulator",
                   help="results subdir: simulator | ibm | baseline (default: simulator)")
    p.add_argument("--since", default=None,
                   help="ISO timestamp prefix; ignore runs older than this (e.g. 2026-06-23)")
    p.add_argument("--ranks", type=int, default=2,
                   help="filter by mpi_ranks; default 2 (canonical config). "
                        "Use 0 to include all (e.g. for ibm backend).")
    p.add_argument("--hw", default=None,
                   help="restrict to one results/<hw-slug>/ folder (e.g. a100-sxm4-40gb). "
                        "Required if runs from more than one hardware slug are found "
                        "(wall-clock medians across GPUs are meaningless).")
    args = p.parse_args()

    ranks_filter = args.ranks if args.ranks > 0 else None
    runs = load_seeded_results(args.backend, args.since, ranks_filter, args.hw)
    if not runs:
        print(f"No seeded {args.backend} runs found "
              f"(looking for JSON files in results/*/{args.backend}/ with 'seed' field).")
        if args.since:
            print(f"Filter: timestamp >= {args.since}")
        if ranks_filter:
            print(f"Filter: mpi_ranks == {ranks_filter}")
        sys.exit(1)

    hw_slugs = {r["_hw_slug"] for r in runs}
    if len(hw_slugs) > 1:
        print(f"ERROR: runs span multiple hardware folders {sorted(hw_slugs)} -- "
              f"wall-clock medians mixing GPUs are meaningless. Re-run with --hw <slug>.")
        sys.exit(1)

    rank_label = f"P={ranks_filter}" if ranks_filter else "any P"
    print(f"Found {len(runs)} unique {args.backend} run(s) at {rank_label}, "
          f"hw={hw_slugs.pop() if hw_slugs else 'n/a'} (deduped by seed+rank):")
    for r in runs:
        print(f"  seed={r['seed']:<4} {r.get('timestamp', '?')[:19]}  {r['_path']}")

    by_mol = aggregate(runs)
    report(by_mol)


if __name__ == "__main__":
    main()
