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


def load_seeded_results(backend: str, since: str | None) -> list[dict]:
    pattern = os.path.join("results", backend, f"{backend}_*.json")
    files = sorted(glob(pattern))
    out = []
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
        d["_path"] = path
        out.append(d)
    return out


def aggregate(runs: list[dict]) -> dict[str, dict]:
    """Group by molecule, collect (seed, energy, wall_time, iters) tuples."""
    by_mol: dict[str, list[dict]] = {}
    for run in runs:
        seed = run["seed"]
        for mol, data in run.get("molecules", {}).items():
            by_mol.setdefault(mol, []).append({
                "seed": seed,
                "energy": data["energy"],
                "fci": data.get("fci"),
                "wall_time": data.get("wall_time"),
                "iters": data.get("iters"),
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
    args = p.parse_args()

    runs = load_seeded_results(args.backend, args.since)
    if not runs:
        print(f"No seeded {args.backend} runs found "
              f"(looking for JSON files in results/{args.backend}/ with 'seed' field).")
        if args.since:
            print(f"Filter: timestamp >= {args.since}")
        sys.exit(1)

    print(f"Found {len(runs)} {args.backend} run(s) with seed field:")
    for r in runs:
        print(f"  seed={r['seed']:<4} {r.get('timestamp', '?')[:19]}  {r['_path']}")

    by_mol = aggregate(runs)
    report(by_mol)


if __name__ == "__main__":
    main()
