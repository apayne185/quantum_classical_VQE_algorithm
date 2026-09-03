#!/usr/bin/env python3

import json
import os
import sys
import glob


RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

def load_results(hw: str | None = None):
    pattern = os.path.join(RESULTS_DIR, hw or "*", "**", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if not f.startswith(PLOTS_DIR)]
    runs = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] Skipping {f}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"  [warn] Skipping {f}: expected a JSON object", file=sys.stderr)
            continue
        data["_file"] = os.path.basename(f)
        data["_path"] = f
        data["_hw_slug"] = f.split(os.sep)[1] if os.sep in f else "?"
        runs.append(data)

    return runs


def print_summary(runs):
    if not runs:
        print("No results found within results/")
        return


    print(f"\n{'File':<36} {'Backend':<12} {'Ranks':<6} {'GPU':<5} {'Commit':<8} {'Timestamp':<20}")
    for r in runs:
        print(f"{r['_file']:<36} {r.get('backend','?'):<12} {r.get('mpi_ranks','?'):<6} "
              f"{str(r.get('gpu','?')):<5} {r.get('git_commit','?'):<8} {r.get('timestamp','?')[:19]:<20}")


    # Molecule details
    print(f"\n{'Run':<28} {'Molecule':<8} {'Energy (Ha)':<14} {'FCI':<14} {'Error (Ha)':<12} {'Tier':<18} {'Iters':<6} {'Time(s)':<8}")
    for r in runs:
        tag = r["_file"][:27]
        mols = r.get("molecules", {})

        chem = r.get("chemistry")  # IBM runs store results here instead of "molecules"
        if chem and isinstance(chem, dict) and "molecule" in chem:
            mol = chem["molecule"]
            energy = chem.get("energy", 0)
            fci = chem.get("fci")
            err = chem.get("error")
            err_str= f"{err:.4f}" if err is not None else "N/A"
            fci_str= f"{fci:.4f}" if fci is not None else "N/A"
            iters = chem.get("iterations",0)
            wall = chem.get("wall_time", 0)
            print(f"{tag:<28} {mol:<8} {energy:<14.6f} {fci_str:<14} {err_str:<12} {'—':<18} {iters:<6} {wall:<8.2f}")

        for mol, data in mols.items():
            energy = data.get("energy", 0)
            fci = data.get("fci")
            err = abs(energy - fci) if fci is not None else None
            err_str = f"{err:.4f}" if err is not None else "N/A"
            fci_str = f"{fci:.4f}" if fci is not None else "N/A"
            tier = data.get("tier","?")
            iters= data.get("iters", 0)
            wall= data.get("wall_time", 0)
            print(f"{tag:<28} {mol:<8} {energy:<14.6f} {fci_str:<14} {err_str:<12} {tier:<18} {iters:<6} {wall:<8.2f}")





# PLOTTING 

MOLECULES = ["H2", "LiH", "BeH2", "H2O", "NH3", "N2"]
MOL_LABELS = {"H2": "H₂", "LiH": "LiH", "BeH2": "BeH₂", "H2O": "H₂O", "NH3": "NH₃", "N2": "N₂"}
COLORS = {"H2": "#1f77b4", "LiH": "#ff7f0e", "BeH2": "#2ca02c", "H2O": "#d62728", "NH3": "#9467bd", "N2": "#8c564b"}


def _ensure_dirs():
    # Create the plot output directories 
    for sub in ["convergence", "scaling", "accuracy", "ibm"]:
        os.makedirs(os.path.join(PLOTS_DIR, sub), exist_ok=True)


def _pick_latest_sim(runs, ranks=2):
    # Return the most complete simulator run at the given rank count (most
    # molecules covered, most recent as tiebreaker). Plain "most recent"
    # would pick a later single-molecule probe (e.g. an N2-only run reusing
    # mpi_ranks=2) over the real multi-molecule sweep -- same failure mode
    # fixed in aggregate_scaling.py/aggregate_seeds.py's best_by_rank().
    candidates = [r for r in runs
                  if r.get("backend") == "simulator"
                  and r.get("mpi_ranks") == ranks
                  and "molecules" in r]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (len(r.get("molecules", {})), r.get("timestamp", "")))


def _pick_latest_baseline(runs):
    # Returns most recent serial baseline
    candidates = [r for r in runs if "baseline" in r["_path"]]
    return candidates[-1] if candidates else None

def _pick_ibm_runs(runs):
    return [r for r in runs if r.get("backend") == "ibm_cloud" and "chemistry" in r]


def plot_convergence_per_molecule(runs, plt):
    # One convergence plot per molecule for all available runs
    _ensure_dirs()
    sim_runs = [r for r in runs if "molecules" in r and r.get("backend") == "simulator"]

    for mol in MOLECULES:
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = False
        fci_val = None

        for r in sim_runs:
            data = r.get("molecules", {}).get(mol)
            if not data or "history" not in data:
                continue
            ranks = r.get("mpi_ranks", "?")
            gpu = "GPU" if r.get("gpu") else "CPU"
            ts = r.get("timestamp", "")[:10]
            label = f"P={ranks} {gpu} ({ts})"
            ax.plot(data["history"], label=label, linewidth=0.9, alpha=0.8)
            if data.get("fci") is not None:
                fci_val = data["fci"]
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        if fci_val is not None:
            ax.axhline(fci_val, color="black", linestyle="--", linewidth=1.2,
                        label=f"FCI = {fci_val:.4f} Ha")

        ax.set_xlabel("SPSA Iteration")
        ax.set_ylabel("Energy (Ha)")
        ax.set_title(f"VQE Convergence — {MOL_LABELS.get(mol, mol)}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        out = os.path.join(PLOTS_DIR, "convergence", f"convergence_{mol}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")



def plot_convergence_combined(runs, plt):
    """Single plot with the latest P=2 run for each molecule."""
    _ensure_dirs()
    run = _pick_latest_sim(runs, ranks=2)
    if not run:
        print("  [warn] plot_convergence_combined: no P=2 simulator run found, skipping", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for mol in MOLECULES:
        data = run.get("molecules", {}).get(mol)
        if not data or "history" not in data:
            continue
        ax.plot(data["history"], label=MOL_LABELS.get(mol, mol),
                color=COLORS.get(mol), linewidth=1.0)
        if data.get("fci") is not None:
            ax.axhline(data["fci"], color=COLORS.get(mol), linestyle="--",
                        linewidth=0.7, alpha=0.5)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("SPSA Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("VQE Convergence - All Molecules (P=2, Latest Run)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    out = os.path.join(PLOTS_DIR, "convergence", "convergence_all_molecules.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")





def plot_wall_time_comparison(runs, plt):
    # Bar chart - serial baseline vs distributed P=2 wall-clock times
    _ensure_dirs()
    baseline = _pick_latest_baseline(runs)
    sim = _pick_latest_sim(runs, ranks=2)
    if not baseline or not sim:
        print("  [warn] plot_wall_time_comparison: missing baseline or P=2 simulator run, skipping", file=sys.stderr)
        return

    mols_found = []
    serial_times = []
    dist_times = []
    for mol in MOLECULES:
        b = baseline.get(mol)
        s = sim.get("molecules", {}).get(mol)
        if b and s:
            mols_found.append(MOL_LABELS.get(mol, mol))
            serial_times.append(b.get("wall_time", 0))
            dist_times.append(s.get("wall_time", 0))

    if not mols_found:
        return

    import numpy as np
    x = np.arange(len(mols_found))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, serial_times, w, label="Serial Baseline", color="#7f8c8d")
    bars2 = ax.bar(x + w/2, dist_times, w, label="Distributed P=2", color="#2980b9")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(mols_found)
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_title("Serial Baseline vs Distributed Stack (P=2, CPU)")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    out = os.path.join(PLOTS_DIR, "scaling", "wall_time_serial_vs_distributed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")




def plot_strong_scaling(runs, plt):
    # Strong scaling - wall-clock time and efficiency across rank counts for each molecule
    _ensure_dirs()
    by_ranks = {}               # latest run per rank count, time sorted
    for r in runs:
        if r.get("backend") != "simulator" or "molecules" not in r:
            continue
        p = r.get("mpi_ranks")
        if p is not None:
            by_ranks[p] = r

    if len(by_ranks) < 2:
        return

    rank_counts = sorted(by_ranks.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for mol in MOLECULES:
        times = []
        ranks_with_data = []
        for p in rank_counts:
            data = by_ranks[p].get("molecules", {}).get(mol)
            if data and "wall_time" in data:
                times.append(data["wall_time"])
                ranks_with_data.append(p)
        if len(times) >= 2:
            ax1.plot(ranks_with_data, times, marker="o", label=MOL_LABELS.get(mol, mol),
                     color=COLORS.get(mol), linewidth=1.5)

    ax1.set_xlabel("MPI Ranks (P)")
    ax1.set_ylabel("Wall-Clock Time (s)")
    ax1.set_title("Strong Scaling - Wall-Clock Time")
    ax1.set_xticks(rank_counts)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    mol = "H2O"                              # efficiency panel uses largest molecule
    times = []
    ranks_with_data = []
    for p in rank_counts:
        data = by_ranks[p].get("molecules", {}).get(mol)
        if data and "wall_time" in data:
            times.append(data["wall_time"])
            ranks_with_data.append(p)

    if len(times) >= 2:
        t1 = times[0]
        efficiencies = [(t1 / (p * t)) * 100 for p, t in zip(ranks_with_data, times)]
        ideal = [100] * len(ranks_with_data)
        ax2.plot(ranks_with_data, efficiencies, marker="s", color=COLORS["H2O"],
                 linewidth=1.5, label="Measured")
        ax2.plot(ranks_with_data, ideal, linestyle="--", color="gray",
                 linewidth=1, label="Ideal (100%)")
        ax2.set_xlabel("MPI Ranks (P)")
        ax2.set_ylabel("Parallel Efficiency (%)")
        ax2.set_title(f"Strong Scaling Efficiency — {MOL_LABELS[mol]}")
        ax2.set_xticks(ranks_with_data)
        ax2.set_ylim(0, 110)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    out = os.path.join(PLOTS_DIR, "scaling", "strong_scaling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")




def plot_weak_scaling(plt):
    # Weak scaling from the scaling .txt file
    _ensure_dirs()
    weak_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "scaling", "weak_scaling_P*.txt")))
    if not weak_files:
        print("  [warn] plot_weak_scaling: no weak_scaling_P*.txt files found, skipping", file=sys.stderr)
        return

    ranks = []
    molecules = []
    total_times = []
    iter_times = []
    terms_per_rank = []

    for f in weak_files:
        with open(f) as fh:
            line = fh.read().strip()
        parts = {}
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v
        p = int(parts.get("P", 0))
        ranks.append(p)
        molecules.append(parts.get("mol", "?"))
        total_times.append(float(parts.get("T_total", 0)))
        iter_times.append(float(parts.get("T/iter", 0)))
        tr = parts.get("terms/rank", "0")
        terms_per_rank.append(float(tr))

    if len(ranks) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Total time
    ax1.bar(range(len(ranks)), total_times, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(ranks)])
    ax1.set_xticks(range(len(ranks)))
    ax1.set_xticklabels([f"P={p}\n{m}" for p, m in zip(ranks, molecules)], fontsize=9)
    ax1.set_ylabel("Total Wall-Clock Time (s)")
    ax1.set_title("Weak Scaling — Total Time (10 iterations)")
    ax1.grid(True, alpha=0.2, axis="y")
    for i, t in enumerate(total_times):
        ax1.text(i, t + 0.2, f"{t:.2f}s", ha="center", fontsize=8)

    # Per iteration time
    ax2.bar(range(len(ranks)), iter_times, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(ranks)])
    ax2.set_xticks(range(len(ranks)))
    ax2.set_xticklabels([f"P={p}\n{m}" for p, m in zip(ranks, molecules)], fontsize=9)
    ax2.set_ylabel("Time Per Iteration (s)")
    ax2.set_title("Weak Scaling — Per-Iteration Time")
    ax2.grid(True, alpha=0.2, axis="y")
    for i, t in enumerate(iter_times):
        ax2.text(i, t + 0.02, f"{t:.3f}s", ha="center", fontsize=8)

    out = os.path.join(PLOTS_DIR, "scaling", "weak_scaling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")



def plot_accuracy_comparison(runs, plt):
    #Bar chart of absolute energy error per molecule - serial vs distributed.
    _ensure_dirs()
    baseline = _pick_latest_baseline(runs)
    sim = _pick_latest_sim(runs, ranks=2)
    if not baseline or not sim:
        print("  [warn] plot_accuracy_comparison: missing baseline or P=2 simulator run, skipping", file=sys.stderr)
        return

    import numpy as np
    mols_found = []
    serial_err = []
    dist_err = []
    for mol in MOLECULES:
        b = baseline.get(mol)
        s = sim.get("molecules", {}).get(mol)
        if b and s and b.get("fci") is not None and s.get("fci") is not None:
            mols_found.append(MOL_LABELS.get(mol, mol))
            serial_err.append(abs(b["energy"] - b["fci"]))
            dist_err.append(abs(s["energy"] - s["fci"]))

    if not mols_found:
        return

    x = np.arange(len(mols_found))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, serial_err, w, label="Serial Baseline", color="#7f8c8d")
    ax.bar(x + w/2, dist_err, w, label="Distributed P=2", color="#2980b9")

    ax.axhline(1.6e-3, color="red", linestyle="--", linewidth=1, alpha=0.7,              # 1.6 mHa
               label="Chemical Accuracy (1.6 mHa)")

    ax.set_xticks(x)
    ax.set_xticklabels(mols_found)
    ax.set_ylabel("Absolute Energy Error (Ha)")
    ax.set_title("Energy Error vs FCI Reference")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    out = os.path.join(PLOTS_DIR, "accuracy", "error_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")




def plot_accuracy_final_vs_fci(runs, plt):
    # final energy vs FCI for each molecule from most recent P=2 run. 
    _ensure_dirs()
    sim = _pick_latest_sim(runs, ranks=2)
    if not sim:
        print("  [warn] plot_accuracy_final_vs_fci: no P=2 simulator run found, skipping", file=sys.stderr)
        return

    mols_found = []
    energies = []
    fcis = [] 
    for mol in MOLECULES:
        data = sim.get("molecules", {}).get(mol)
        if data and data.get("fci") is not None:
            mols_found.append(MOL_LABELS.get(mol, mol))
            energies.append(data["energy"])
            fcis.append(data["fci"])

    if not mols_found:
        return

    import numpy as np
    x = np.arange(len(mols_found))
    w = 0.35  
    fig, ax = plt.subplots(figsize=(8, 5))   

    ax.bar(x - w/2, fcis, w, label="FCI (Exact)", color="#27ae60", alpha=0.8)
    ax.bar(x + w/2, energies, w, label="VQE (Distributed P=2)", color="#2980b9", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mols_found)     
    ax.set_ylabel("Energy (Ha)")      
    ax.set_title("VQE Energy vs FCI Ground Truth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    out = os.path.join(PLOTS_DIR, "accuracy", "energy_vs_fci.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")



def plot_ibm_convergence(runs, plt):
    # IBM QPU iteration-by-iteration energy trajectory
    _ensure_dirs()
    ibm_runs = _pick_ibm_runs(runs)
    if not ibm_runs:
        print("  [warn] plot_ibm_convergence: no IBM cloud runs found, skipping", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for r in ibm_runs:
        chem = r["chemistry"]
        if "history" not in chem:
            continue
        ts = r.get("timestamp", "")[:10]
        label = f"IBM QPU ({ts})"
        iters = list(range(1, len(chem["history"]) + 1))
        ax.plot(iters, chem["history"], marker="o", linewidth=1.2, label=label)
        if chem.get("fci") is not None:
            ax.axhline(chem["fci"], color="green", linestyle="--", linewidth=1,
                        alpha=0.7, label=f"FCI = {chem['fci']:.4f} Ha")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    # Read backend name from result data rather than hardcoding it
    backends = list({r.get("backend", "ibm_cloud") for r in ibm_runs})
    backend_label = backends[0] if len(backends) == 1 else "IBM QPU"
    ax.set_xlabel("SPSA Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"IBM Quantum QPU - H₂ VQE Convergence ({backend_label})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out = os.path.join(PLOTS_DIR, "ibm", "ibm_qpu_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")




def plot_ibm_rtt(runs, plt):
    # IBM QPU round-trip time per iteration. 
    _ensure_dirs()
    ibm_runs = _pick_ibm_runs(runs)
    if not ibm_runs:
        print("  [warn] plot_ibm_rtt: no IBM cloud runs found, skipping", file=sys.stderr)
        return

    r = ibm_runs[-1]
    chem = r["chemistry"]
    wall = chem.get("wall_time", 0)
    iters = chem.get("iterations", 0)
    if iters == 0:
        return
    avg_rtt = wall / iters

    fig, ax = plt.subplots(figsize=(8, 4))
    if "history" in chem and iters > 0:
        iter_nums = list(range(1, iters + 1))
        ax.bar(iter_nums, [avg_rtt] * iters, color="#e74c3c", alpha=0.7)             # uniform, no per-iter RTT in JSON
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Avg Time Per Iteration (s)")
        ax.set_title(f"IBM QPU - Avg {avg_rtt:.1f}s/iter (dominated by QPU RTT)")
        ax.set_xticks(iter_nums)
        ax.grid(True, alpha=0.2, axis="y")

    out = os.path.join(PLOTS_DIR, "ibm", "ibm_qpu_rtt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")



def plot_speedup_by_molecule(runs, plt):
    # Speedup (serial/distributed) per molecule across all available rank counts.
    _ensure_dirs()
    baseline = _pick_latest_baseline(runs)
    if not baseline:
        print("  [warn] plot_speedup_by_molecule: no serial baseline found, skipping", file=sys.stderr)
        return

    by_ranks = {}
    for r in runs:
        if r.get("backend") != "simulator" or "molecules" not in r:
            continue
        p = r.get("mpi_ranks")
        if p is not None:
            by_ranks[p] = r

    rank_counts = sorted(by_ranks.keys())
    if len(rank_counts) < 2:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for mol in MOLECULES:
        serial_t = baseline.get(mol, {}).get("wall_time")
        if serial_t is None or serial_t == 0:
            continue
        speedups = []
        ranks_with_data = []
        for p in rank_counts:
            data = by_ranks[p].get("molecules", {}).get(mol)
            if data and "wall_time" in data and data["wall_time"] > 0:
                speedups.append(serial_t / data["wall_time"])
                ranks_with_data.append(p)
        if speedups:
            ax.plot(ranks_with_data, speedups, marker="o",
                    label=MOL_LABELS.get(mol, mol), color=COLORS.get(mol), linewidth=1.5)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Break-even (1.0x)")
    ax.set_xlabel("MPI Ranks (P)")
    ax.set_ylabel("Speedup vs Serial Baseline")
    ax.set_title("Speedup by Molecule Across Rank Counts")
    ax.set_xticks(rank_counts)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    out = os.path.join(PLOTS_DIR, "scaling", "speedup_by_molecule.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_all(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed - skipping plots.")
        return


    print("\nGenerating plots...")
    plot_convergence_per_molecule(runs, plt)
    plot_convergence_combined(runs, plt)
    plot_wall_time_comparison(runs, plt)
    plot_strong_scaling(runs, plt)
    plot_weak_scaling(plt)
    plot_speedup_by_molecule(runs, plt)
    plot_accuracy_comparison(runs, plt)
    plot_accuracy_final_vs_fci(runs, plt)
    plot_ibm_convergence(runs, plt)
    plot_ibm_rtt(runs, plt)

    print(f"\nAll plots saved to {PLOTS_DIR}/")



if __name__ == "__main__":
    hw = None
    if "--hw" in sys.argv:
        _idx = sys.argv.index("--hw")
        if _idx + 1 < len(sys.argv):
            hw = sys.argv[_idx + 1]

    runs = load_results(hw)

    hw_slugs = {r["_hw_slug"] for r in runs}
    if len(hw_slugs) > 1:
        print(f"ERROR: results span multiple hardware folders {sorted(hw_slugs)} -- "
              f"a combined summary/plot mixing GPUs is misleading. Re-run with --hw <slug>.",
              file=sys.stderr)
        sys.exit(1)

    print_summary(runs)

    if "--plot" in sys.argv:
        plot_all(runs)
