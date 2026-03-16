import sys
import os
import numpy as np
import time


# so python can find C++ module
sys.path.insert(0, os.path.abspath("./build"))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  

try:
    from src.api.interface import HPCHybridStack
    from src.api.problems import ChemistryProblem, FinanceProblem, ANSATZ_TIERS
    from src.api.molecule_resolver import (MoleculeResolver, MoleculeTooBigError, ResolutionError)
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
    if stack.rank == 0: print(f"\n--- RUNNING CHEMISTRY TASK {molecule_input} ---")

    problem = make_problem(molecule_input, force_tier=force_tier)
    if problem is None:
        return None, None
    
    t0 = time.perf_counter()
    theta, history = stack.vqe_optimize(problem, 
                                        max_iterations=50,        
                                        tolerance=1.6e-3, 
                                        checkpoint_dir="checkpoints") 
    
    t_total = time.perf_counter() - t0
    
    if stack.rank ==0 and history:
        final_e = history[-1]
        error= problem.energy_error(final_e)
        error_str = f"{error:+.4f} Ha " if error is not None else "N/A"   

        print(f"\n[{problem.name}] Ansatz tier:{problem.ansatz_tier}")
        print(f"[{problem.name}] Corr score: {problem.diagnostics.get('correlation_score', 'N/A'):.3f} ")  
        print(f"[{problem.name}] Final energy : {final_e:+.6f} Ha ")
        if problem.fci_energy is not None:
            print(f"[{problem.name}] Reference FCI : {problem.fci_energy:+.6f} Ha ")      #known value of LiH
            print(f"[{problem.name}] Absolute error: {error_str} Ha ")
            print(f"[{problem.name}] Iterations run: {len(history)}")
            print(f"[{problem.name}] Wall time: {t_total:.2f}s (includes QPU queue + RTT) ")   
            print(f"[{problem.name}] Time/iter: {t_total/ len(history):.2f} s ")   

        if hasattr(problem, 'resolution_metadata'):
            meta = problem.resolution_metadata
            print(f"[{problem.name}] Source: {meta['source']} ")
            print(f"[{problem.name}] Active electrons: {meta['active_electrons']} ")
            print(f"[{problem.name}] Estimated qubits: {meta['estimated_qubits']} ")

            for w in meta.get('warnings',[]):    
                print(f"[{problem.name}] Note: {w}")

    return history, problem   




def run_finance_local(stack:HPCHybridStack): 
    if stack.rank == 0: print("\n--- RUNNING FINANCE (4-Assest Portfolio QUBO) TASK ---")
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
                                        checkpoint_dir="checkpoints",
    )
    t_total = time.perf_counter() - t0

    if stack.rank == 0 and history:
        print(f"[Finance] Final objective : {history[-1]:+.6f} ")
        print(f"[Finance] Iterations run : {len(history)}")
        print(f"[Finance] Wall time : {t_total:.2f} s")    


  
def run_scaling_local(stack: HPCHybridStack):   
    if stack.rank == 0: print(f"\n RUNNING SCALING (with P={stack.size} ranks) ")

    # problem = ChemistryProblem.from_name("LiH")
    problem = make_problem("LiH")
    if problem is None:
        if stack.rank == 0: print("[Scaling] LiH  resolution failed, skipping.")  
        return
    t0 = time.perf_counter()
    
    _, history = stack.vqe_optimize(problem, max_iterations=10)
    t_total= time.perf_counter() - t0    

    if stack.rank == 0:
        print(f"[Scaling] P={stack.size} |  T_total={t_total:.3f}s |  final_E={history[-1]:+.6f} | tier={getattr(problem, 'ansatz_tier', 'N/A')} | qubits={problem.num_qubits}")



    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        MOLECULES = sys.argv[1:]   

    print(f"[Config] GPU={'requested' if USE_GPU else 'CPU mode'}")
    print(f"[Config] Molecules: {MOLECULES}") 
    print(f"[Config] Resolver: max_qubits=20, cache=.pubchem_cache/")


    with HPCHybridStack(use_gpu=USE_GPU, backend=BACKEND) as stack:   
        results = {}
        for mol in MOLECULES:
            history, problem = run_chemistry_local(stack, mol) 

            if stack.rank == 0 and history:
                if problem is not None:
                    results[mol] = {
                        "energy": history[-1],
                        "tier":getattr(problem, "ansatz_tier", "hwe"),
                        "score":problem.diagnostics.get("correlation_score", 0.0),
                        "fci":getattr(problem, "fci_energy", None),   
                    }

        # run_chemistry_local(stack)   
        run_finance_local(stack)   
        run_scaling_local(stack)   

        if stack.rank == 0:
            print("---ALL LOCAL BENCHMARKS COMPLETE ----")
            print(f"Ranks used : {stack.size}")
            print(f"GPU : {stack.use_gpu}")
            print(f"Backend : {BACKEND} ")
            print(f"{'Molecule':<10} {'Energy (Ha)':<16} {'Error (Ha)':<14} {'Ansatz':<20} {'Corr'} ")    

            for mol, data in results.items():
                fci = data.get("fci")                                    #= problem.fci_energy
                energy  = data["energy"]
                err_str = (f"{abs(energy-fci):+.4f} Ha" if fci is not None else "N/A")   
                tier_label = ANSATZ_TIERS.get(data['tier'],{}).get("label", data['tier'])

                print(f"{mol:<10} {energy:<16.6f} {err_str:<14} {tier_label:<20} {data['score']:.3f}")



