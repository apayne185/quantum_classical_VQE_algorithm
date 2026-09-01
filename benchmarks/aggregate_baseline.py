"""Aggregate baseline_comparison.py JSONs into the paper's Table 1.

Reads results/baseline_comparison/{hpchybrid,lightning,aer-mpi}/<mol>_*.json
and produces a molecule-by-backend wall-clock and accuracy table.

Refuses to average across mixed devices (e.g. a `lightning` CPU dry-run
next to a `hpchybrid` GPU run) -- flags them explicitly so a rushed
cloud session doesn't silently produce a misleading table.

Run:
    python benchmarks/aggregate_baseline.py
    python benchmarks/aggregate_baseline.py --require-device gpu
    python benchmarks/aggregate_baseline.py --out-dir results/baseline_comparison
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob

BACKENDS = ("hpchybrid", "lightning", "aer-mpi")
MOLECULES = ("H2", "LiH", "BeH2", "H2O", "NH3", "N2")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="results/baseline_comparison",
                   help="Root directory scanned for baseline JSONs.")
    p.add_argument("--require-device", choices=("gpu", "cpu"),
                   help="Fail if any run in the table uses the other device "
                        "class. Use gpu for the paper table; cpu for local "
                        "sanity checks.")
    p.add_argument("--markdown", action="store_true",
                   help="Emit as Markdown (default: plain text table).")
    return p.parse_args()


def _load(out_dir: str) -> dict[tuple[str, str], dict]:
    """Return {(backend, molecule): latest_run_dict} across the out-dir tree.

    Latest-wins on timestamp — matches the pattern in aggregate_seeds.py,
    intentionally simple: baseline_comparison.py stamps a fresh timestamp
    into every JSON, so the newest file is the newest run.
    """
    latest: dict[tuple[str, str], dict] = {}
    for backend in BACKENDS:
        for path in sorted(glob(os.path.join(out_dir, backend, "*.json"))):
            try:
                with open(path) as f:
                    d = json.load(f)
            except json.JSONDecodeError:
                print(f"[skip] {path}: invalid JSON", file=sys.stderr)
                continue
            mol = d.get("molecule")
            if not mol:
                continue
            d["_path"] = path
            key = (backend, mol)
            # Newest timestamp wins.
            if key not in latest or path > latest[key]["_path"]:
                latest[key] = d
    return latest


def _device_class(device_label: str) -> str:
    return "gpu" if device_label.startswith("gpu") else "cpu"


def _fmt(v, spec):
    return "--" if v is None else format(v, spec)


def _table(rows, headers, markdown: bool) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    if markdown:
        sep = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        div = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        body = "\n".join("| " + " | ".join(str(r[i]).ljust(w) for i, w in enumerate(widths)) + " |"
                         for r in rows)
        return f"{sep}\n{div}\n{body}"
    sep = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    body = "\n".join("  ".join(str(r[i]).ljust(w) for i, w in enumerate(widths))
                     for r in rows)
    return f"{sep}\n{'-' * len(sep)}\n{body}"


def main():
    args = _parse_args()
    runs = _load(args.out_dir)

    if not runs:
        print(f"[aggregate_baseline] no runs found under {args.out_dir}. "
              f"Run baseline_comparison.py first.")
        sys.exit(1)

    devices_seen = {_device_class(r["device"]) for r in runs.values()}
    if args.require_device and args.require_device not in devices_seen:
        print(f"[aggregate_baseline] --require-device={args.require_device} "
              f"but only saw {devices_seen}. Refusing to produce a mismatched table.",
              file=sys.stderr)
        sys.exit(2)
    if args.require_device and len(devices_seen) > 1:
        print(f"[aggregate_baseline] mixed devices detected ({devices_seen}); "
              f"table would compare CPU and GPU wall-clocks side-by-side "
              f"(dishonest). Drop --require-device to see the mixed table anyway.",
              file=sys.stderr)
        sys.exit(3)

    if len(devices_seen) > 1:
        print(f"[aggregate_baseline] WARNING: mixed devices ({devices_seen}) "
              f"in this table. Wall-clocks are NOT directly comparable across "
              f"CPU/GPU. Pass --require-device gpu when producing the paper table.")

    # Reference speedup vs hpchybrid (our stack), per molecule.
    header = ["Molecule", "n_qubits", "n_pauli", "Backend", "Device",
              "Wall (s)", "s/iter", "Iters", "E (Ha)", "|Δ FCI| (mHa)", "Speedup vs hpchybrid"]
    rows = []
    for mol in MOLECULES:
        hpc = runs.get(("hpchybrid", mol))
        hpc_wall = hpc["wall_seconds"] if hpc else None
        for backend in BACKENDS:
            r = runs.get((backend, mol))
            if not r:
                rows.append([mol, "--", "--", backend, "--", "--", "--", "--", "--", "--", "--"])
                continue
            err_mha = None
            if r.get("fci_ha") is not None and r.get("energy_ha") is not None:
                err_mha = abs(r["energy_ha"] - r["fci_ha"]) * 1000.0
            speedup = None
            if backend != "hpchybrid" and hpc_wall and r["wall_seconds"]:
                speedup = hpc_wall / r["wall_seconds"]
            rows.append([
                mol, r["num_qubits"], r["num_pauli"],
                backend, r["device"],
                _fmt(r["wall_seconds"], ".2f"),
                _fmt(r["wall_per_iter"], ".4f"),
                r["iters_completed"],
                _fmt(r.get("energy_ha"), ".4f"),
                _fmt(err_mha, ".2f"),
                (f"{speedup:.2f}x" if speedup else ("--" if backend != "hpchybrid" else "1.00x")),
            ])

    print(_table(rows, header, args.markdown))
    print(f"\n[aggregate_baseline] {len(runs)} runs loaded, "
          f"{len(devices_seen)} device class(es): {devices_seen}")


if __name__ == "__main__":
    main()
