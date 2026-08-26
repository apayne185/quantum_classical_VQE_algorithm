"""Build a strong-scaling table from results/simulator/*.json files.

Groups simulator runs by `mpi_ranks` (P=1,2,4,8 etc.), filters to a single seed
to keep apples-to-apples, and reports per-molecule wall time + speedup +
efficiency vs P=1.

Usage:
    python benchmarks/aggregate_scaling.py
    python benchmarks/aggregate_scaling.py --seed 42 --since 2026-06-24
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from glob import glob


def load_runs(backend: str, since: str | None, seed: int | None, hw: str | None) -> list[dict]:
    # results/<hardware-slug>/<backend>/<backend>_*.json -- the hw wildcard/filter
    # walks (or targets one of) every hardware folder (a100-sxm4-40gb,
    # rtx-6000-ada-generation, cpu-only, ...)
    pattern = os.path.join("results", hw or "*", backend, f"{backend}_*.json")
    out = []
    for path in sorted(glob(pattern)):
        try:
            with open(path) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        if "mpi_ranks" not in d:
            continue
        if seed is not None and d.get("seed") != seed:
            continue
        if since and d.get("timestamp", "") < since:
            continue
        d["_path"] = path
        d["_hw_slug"] = path.split(os.sep)[1]
        out.append(d)
    return out


def best_by_rank(runs: list[dict]) -> dict[int, dict]:
    """Keep the most complete run at each rank count (most molecules covered,
    most recent as tiebreaker).

    A "most recent wins" rule alone silently prefers a later single-molecule
    run (e.g. an NH3-only or N2-only probe) over an earlier full 4-molecule
    sweep at the same P, producing a scaling table missing most of its rows.
    Molecule count is a reasonable proxy for "this was the intended sweep
    run" without hardcoding a specific benchmark set.
    """
    by_rank: dict[int, dict] = {}
    for r in runs:
        P = r["mpi_ranks"]
        n_mols = len(r.get("molecules", {}))
        current = by_rank.get(P)
        if current is None:
            by_rank[P] = r
            continue
        current_n_mols = len(current.get("molecules", {}))
        if (n_mols, r.get("timestamp", "")) > (current_n_mols, current.get("timestamp", "")):
            by_rank[P] = r
    return dict(sorted(by_rank.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="simulator")
    p.add_argument("--seed", type=int, default=42,
                   help="filter by SEED (default 42 - matches scaling sweep default)")
    p.add_argument("--since", default=None,
                   help="ISO timestamp prefix; ignore runs older than this")
    p.add_argument("--hw", default=None,
                   help="restrict to one results/<hw-slug>/ folder (e.g. a100-sxm4-40gb). "
                        "Required if runs from more than one hardware slug are found.")
    args = p.parse_args()

    runs = load_runs(args.backend, args.since, args.seed, args.hw)

    hw_slugs = {r["_hw_slug"] for r in runs}
    if len(hw_slugs) > 1:
        print(f"ERROR: runs span multiple hardware folders {sorted(hw_slugs)} -- "
              f"a scaling table mixing GPUs is meaningless. Re-run with --hw <slug>.")
        sys.exit(1)

    by_rank = best_by_rank(runs)

    if not by_rank:
        print(f"No {args.backend} runs found with seed={args.seed}"
              + (f", hw={args.hw}" if args.hw else "") + ".")
        sys.exit(1)

    print(f"\nStrong scaling table  (seed={args.seed}, backend={args.backend}, "
          f"hw={hw_slugs.pop() if hw_slugs else 'n/a'})")
    for P, r in by_rank.items():
        print(f"  P={P:<3}  {r.get('timestamp', '')[:19]}  GPU={r.get('gpu')}  "
              f"{r['_path']}")

    # Collect molecule names across runs
    mols = set()
    for r in by_rank.values():
        mols.update(r.get("molecules", {}).keys())

    ranks = sorted(by_rank.keys())
    print()
    print(f"{'Molecule':<8} " + "".join(f"{'T(P=' + str(P) + ')':<12}" for P in ranks))
    print("-" * (8 + 12 * len(ranks)))
    for mol in sorted(mols):
        row = f"{mol:<8} "
        for P in ranks:
            data = by_rank[P].get("molecules", {}).get(mol)
            if data and data.get("wall_time"):
                row += f"{data['wall_time']:<12.2f}"
            else:
                row += f"{'-':<12}"
        print(row)

    # Speedup + efficiency vs P=1 (if present)
    base_P = ranks[0]
    if base_P != 1:
        print(f"\nNote: P=1 not present; speedup/efficiency referenced to P={base_P}")
    print(f"\nSpeedup vs P={base_P} (T_base / T_P):")
    print(f"{'Molecule':<8} " + "".join(f"{'S(P=' + str(P) + ')':<12}" for P in ranks))
    print("-" * (8 + 12 * len(ranks)))
    for mol in sorted(mols):
        row = f"{mol:<8} "
        base = by_rank[base_P].get("molecules", {}).get(mol, {}).get("wall_time")
        for P in ranks:
            T = by_rank[P].get("molecules", {}).get(mol, {}).get("wall_time")
            if base and T:
                row += f"{base / T:<12.2f}"
            else:
                row += f"{'-':<12}"
        print(row)

    print(f"\nEfficiency vs P={base_P}  (speedup / P):")
    print(f"{'Molecule':<8} " + "".join(f"{'E(P=' + str(P) + ')':<12}" for P in ranks))
    print("-" * (8 + 12 * len(ranks)))
    for mol in sorted(mols):
        row = f"{mol:<8} "
        base = by_rank[base_P].get("molecules", {}).get(mol, {}).get("wall_time")
        for P in ranks:
            T = by_rank[P].get("molecules", {}).get(mol, {}).get("wall_time")
            if base and T:
                eff = (base / T) / P * 100
                row += f"{eff:<12.1f}"
            else:
                row += f"{'-':<12}"
        print(row)


if __name__ == "__main__":
    main()