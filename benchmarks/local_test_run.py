import sys
import os
import numpy as np
import time
from datetime import datetime


# so python can find C++ module and src package
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "build"))

try:
    from src.api.interface import HPCHybridStack
    from src.api.problems import ChemistryProblem, FinanceProblem, ANSATZ_TIERS
    from src.api.molecule_resolver import (MoleculeResolver, MoleculeTooBigError, ResolutionError)
    from src.api.results import save_results
    from src.api.log import init_log, log, close_log
    import hpc_core
    print("hpc_core and interface imported.")
except ImportError as e:
    print(f"failed to import hpc_core module: {e}")
    sys.exit(1)



USE_GPU = os.environ.get("USE_GPU", "yes").strip().lower() == "yes"
BACKEND = os.environ.get("BACKEND", "simulator")
_env_molecules = os.environ.get("MOLECULES", "").strip()
MOLECULES = _env_molecules.split() if _env_molecules else ["H2", "LiH", "BeH2", "H2O"]

resolver = MoleculeResolver(max_qubits=20, allow_network=True, cache_dir=".pubchem_cache",)



def make_problem(molecule_input: str, force_tier: str|None= None)-> ChemistryProblem | None: 
    try:
        result= resolver.resolve(molecule_input,freeze_core=True)
        problem = result.to_chemistry_problem(force_tier=force_tier)
        problem.prepare() 
        return problem  
    
    except MoleculeTooBigError as e:
        print(f"\n[SKIPPED] {molecule_input}: {e}")
        return None
    except ResolutionError as e:
        print(f"\n[FAILED] {molecule_input}: {e}")
        return None
    except Exception as e:
        print(f"\n[ERROR] {molecule_input}: {type(e).__name__}: {e}")
        return None




def run_chemistry_local(stack: HPCHybridStack, molecule_input: str, force_tier: str | None= None): 
    if stack.rank == 0: print(f"\n\n--- RUNNING CHEMISTRY TASK {molecule_input} ---")

    problem = make_problem(molecule_input, force_tier=force_tier)
    if problem is None:
        return None, None
    
    t0 = time.perf_counter()
    # Scale iterations by parameter count — more params need more SPSA steps
    max_iters = max(200, problem.num_params * 8)
    ckpt_dir = os.path.join("checkpoints", problem.name)
    theta, history = stack.vqe_optimize(problem,
                                        max_iterations=max_iters,
                                        tolerance=1.6e-3,
                                        checkpoint_dir=ckpt_dir,
                                        seed=42)
    
    t_total = time.perf_counter() - t0
    
    if stack.rank ==0 and history:
        final_e = history[-1]
        fci = problem.fci_energy

        # Use best physical energy (above FCI) if final energy went below
        best_physical = getattr(stack, '_best_physical_energy', None)
        if fci is not None and final_e < fci - 1e-6 and best_physical is not None:
            report_e = best_physical
            report_iter = getattr(stack, '_best_physical_iter', len(history))
            report_label = "Best physical energy"
        else:
            report_e = final_e
            report_iter = len(history)
            report_label = "Final energy"

        error = abs(report_e - fci) if fci is not None else None
        error_str = f"{error:+.4f} Ha " if error is not None else "N/A"

        print(f"\n[{problem.name}] Ansatz tier:{problem.ansatz_tier}")
        print(f"[{problem.name}] Corr score: {problem.diagnostics.get('correlation_score', 'N/A'):.3f} ")
        print(f"[{problem.name}] {report_label}: {report_e:+.6f} Ha ")
        if fci is not None:
            print(f"[{problem.name}] Reference FCI : {fci:+.6f} Ha ")
            print(f"[{problem.name}] Absolute error: {error_str}")
            print(f"[{problem.name}] Iterations run: {len(history)} (best at iter {report_iter})")
            print(f"[{problem.name}] Wall time: {t_total:.2f}s (includes QPU queue + RTT) ")
            print(f"[{problem.name}] Time/iter: {t_total/ len(history):.2f} s ")

        if hasattr(problem, 'resolution_metadata'):
            meta = problem.resolution_metadata
            print(f"[{problem.name}] Source: {meta['source']} ")
            print(f"[{problem.name}] Active electrons: {meta['active_electrons']} ")
            print(f"[{problem.name}] Estimated qubits: {meta['estimated_qubits']} ")

            for w in meta.get('warnings',[]):    
                print(f"[{problem.name}] Note: {w}")

    return history, problem, t_total   




def run_finance_local(stack:HPCHybridStack):
    if stack.rank == 0: print("\n\n\n--- RUNNING FINANCE (4-Assest Portfolio QUBO) TASK ---")
    np.random.seed(42)
    n_assets = 4
    #  synthetic positive definite covariance matrix
    A = np.random.rand(n_assets, n_assets)
    cov = A @ A.T / n_assets
    mu  = np.random.uniform(0.02, 0.15, n_assets)

    problem = FinanceProblem(cov, expected_returns=mu, risk_factor=1.0)
    t0 = time.perf_counter()
    _, history = stack.vqe_optimize(problem,
                                        max_iterations=30,
                                        tolerance=1e-3,
                                        checkpoint_dir="checkpoints/finance",
    )
    t_total = time.perf_counter() - t0

    result = None
    if stack.rank == 0 and history:
        print(f"[Finance] Final objective : {history[-1]:+.6f} ")
        print(f"[Finance] Iterations run : {len(history)}")
        print(f"[Finance] Wall time : {t_total:.2f} s")
        result = {
            "final_objective": history[-1],
            "iterations": len(history),
            "wall_time": t_total,
            "n_assets": n_assets,
            "history": history,
        }
    return result



def run_scaling_local(stack: HPCHybridStack):
    if stack.rank == 0: print(f"\n RUNNING SCALING (with P={stack.size} ranks) ")

    problem = make_problem("LiH")
    if problem is None:
        if stack.rank == 0: print("[Scaling] LiH  resolution failed, skipping.")
        return None
    t0 = time.perf_counter()

    _, history = stack.vqe_optimize(problem, max_iterations=10,
                                    checkpoint_dir="checkpoints/scaling")
    t_total= time.perf_counter() - t0

    result = None
    if stack.rank == 0 and history:
        result_line = (f"P={stack.size} T_total={t_total:.4f} final_E={history[-1]:+.6f} tier={getattr(problem, 'ansatz_tier', 'N/A')} qubits={problem.num_qubits} iters={len(history)}")
        print(f"[Scaling] {result_line}")

        os.makedirs("results/scaling",exist_ok=True)
        with open(f"results/scaling/scaling_P{stack.size}.txt", "w") as f:
            f.write(result_line + "\n")

        result = {
            "ranks": stack.size,
            "wall_time": t_total,
            "final_energy": history[-1],
            "tier": getattr(problem, 'ansatz_tier', 'N/A'),
            "qubits": problem.num_qubits,
            "iterations": len(history),
        }
    return result



    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        MOLECULES = sys.argv[1:]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    init_log(f"results/run_{ts}.log")

    print(f"[Config] GPU={'requested' if USE_GPU else 'CPU mode'}")
    print(f"[Config] Molecules: {MOLECULES}")
    print(f"[Config] Resolver: max_qubits=20, cache=.pubchem_cache/")


    with HPCHybridStack(use_gpu=USE_GPU, backend=BACKEND) as stack:
        results = {}
        for mol in MOLECULES:
            history, problem, t_total = run_chemistry_local(stack, mol)

            if stack.rank == 0 and history:
                if problem is not None:
                    fci = getattr(problem, "fci_energy", None)
                    final_e = history[-1]
                    best_phys = getattr(stack, '_best_physical_energy', None)
                    # Report best physical energy if final went below FCI
                    if fci is not None and final_e < fci - 1e-6 and best_phys is not None:
                        report_e = best_phys
                    else:
                        report_e = final_e
                    results[mol] = {
                        "energy": report_e,
                        "tier":getattr(problem, "ansatz_tier", "hwe"),
                        "score":problem.diagnostics.get("correlation_score", 0.0),
                        "fci": fci,
                        "iters": len(history),
                        "wall_time": t_total,
                        "history": history,
                    }

        # Finance problem kept in codebase but excluded from chemistry-focused benchmarks
        # finance_result = run_finance_local(stack)
        scaling_result = run_scaling_local(stack)

        if stack.rank == 0:
            print("\n\n\n---ALL LOCAL BENCHMARKS COMPLETE ----")
            print(f"Ranks used : {stack.size}")
            print(f"GPU : {stack.use_gpu}")
            print(f"Backend : {BACKEND} ")
            print(f"{'Molecule':<10} {'Energy (Ha)':<16} {'Error (Ha)':<14} {'Ansatz':<20} {'Corr':<6} {'Iters':<8} {'Time(s)':<10} {'T/iter(s)':<10}")

            for mol, data in results.items():
                fci = data.get("fci")
                energy  = data["energy"]
                err_str = (f"{abs(energy-fci):+.4f} Ha" if fci is not None else "N/A")
                tier_label = ANSATZ_TIERS.get(data['tier'],{}).get("label", data['tier'])
                iters  = data.get("iters", 0)
                t_wall = data.get("wall_time", 0.0)
                t_per_iter = t_wall / iters if iters > 0 else 0.0

                print(f"{mol:<10} {energy:<16.6f} {err_str:<14} {tier_label:<20} {data['score']:<6.3f} {iters:<8} {t_wall:<10.2f} {t_per_iter:<10.4f}")

            save_results({
                "mpi_ranks": stack.size,
                "gpu": stack.use_gpu,
                "molecules": results,
                "scaling": scaling_result,
            }, backend=BACKEND)

    close_log()



