#!/usr/bin/env python3

import json
import os
import sys
import glob
   
  
RESULTS_DIR = "results"

def load_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    runs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            data["_file"] = os.path.basename(f)
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

        # Handles the ibm_cloud format (single chemistry result)
        chem = r.get("chemistry")
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
 
 

def plot_convergence(runs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plots creation.")
        return

    fig, ax = plt.subplots(figsize=(10,6))
    plotted = False 


    for r in runs:
        backend = r.get("backend", "?")
        mols = r.get("molecules", {})
                                     
        # IBM chemistry
        chem = r.get("chemistry")
        if chem and isinstance(chem, dict) and "history" in chem:
            mol = chem["molecule"]
            ax.plot(chem["history"], label=f"{mol} ({backend})", marker=".", markersize=3)
            if chem.get("fci"):
                ax.axhline(chem["fci"], linestyle="--", alpha=0.5)
            plotted = True
                  
        for mol, data in mols.items():
            if "history" in data:
                ax.plot(data["history"], label=f"{mol} ({backend})", marker=".", markersize=3)
                if data.get("fci"):
                    ax.axhline(data["fci"], linestyle="--", alpha=0.5)   

                plotted = True 

                            
    if plotted: 
        ax.set_xlabel("SPSA Iteration")
        ax.set_ylabel("Energy (Ha)")
        ax.set_title("VQE Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        out = os.path.join(RESULTS_DIR, "convergence.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")    
        print(f"\nConvergence plot saved to {out}")
        plt.close(fig)    

    else:
        print("\nNo convergence histories found to plot. ")
                             



if __name__ == "__main__":
    runs = load_results()
    print_summary(runs)

    if "--plot" in sys.argv:
        plot_convergence(runs)
