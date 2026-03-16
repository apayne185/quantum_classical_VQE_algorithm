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
    from src.api.problems import ChemistryProblem, FinanceProblem
    import hpc_core
    print("hpc_core and interface imported.")
except ImportError as e:
    print(f"failed to import hpc_core module: {e}")
    sys.exit(1)


USE_GPU = os.environ.get("USE_GPU", "yes").strip().lower() == "yes"
BACKEND = os.environ.get("BACKEND", "simulator")
MOLECULES = ["H2", "LiH", "BeH2", "H2O"]
# MOLECULES = ["H2", "LiH", "BeH2", "H2O", "NH3"]


def run_chemistry_local(stack: HPCHybridStack, name:str): 
    if stack.rank == 0: print(f"\n--- RUNNING CHEMISTRY TASK {name} ---")

    problem = ChemistryProblem.from_name(name)
    t0 = time.perf_counter()
    theta, history = stack.vqe_optimize(problem, 
                                        max_iterations=50,        
                                        tolerance=1.6e-3, 
                                        checkpoint_dir="checkpoints") 
    
    t_total = time.perf_counter() - t0
    
    if stack.rank ==0 and history:
        final_e = history[-1]
        error   = problem.energy_error(final_e)
        error_str = f"{error:+.4f} Ha " if error is not None else "N/A"   

        print(f"[{name}] Final energy : {final_e:+.6f} Ha ")
        if problem.fci_energy is not None:
            print(f"[{name}] Reference FCI : {problem.fci_energy:+.6f} Ha ")      #known value of LiH
            print(f"[{name}] Absolute error: {error_str} Ha ")
            print(f"[{name}] Iterations run: {len(history)}")
            print(f"[{name}] Wall time: {t_total:.2f}s (includes QPU queue + RTT) ")   
            print(f"[{name}] Time/iter: {t_total / len(history):.2f} s ")   

    return history  




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
    if stack.rank == 0: print(f"\n RUNNING SCALAING (with P={stack.size} ranks)")

    problem = ChemistryProblem.from_name("LiH")
    t0 = time.perf_counter()
    
    _, history = stack.vqe_optimize(problem, max_iterations=10)
    t_total= time.perf_counter() - t0    

    if stack.rank == 0:
        print(f"[Scaling] P={stack.size} |  T_total={t_total:.3f}s |  final_E={history[-1]:+.6f}  ")



    

if __name__ == "__main__":
    print(f"[Config] GPU={'requested' if USE_GPU else 'CPU mode'}")
    print(f"[Config] Molecules: {MOLECULES}") 

    with HPCHybridStack(use_gpu=USE_GPU, backend=BACKEND) as stack:   

        results = {}
        for mol in MOLECULES:
            history = run_chemistry_local(stack, mol)
            if stack.rank == 0 and history:
                results[mol] = history[-1] 

        # run_chemistry_local(stack)   
        run_finance_local(stack)   
        run_scaling_local(stack)   

        if stack.rank == 0:
            print("---ALL LOCAL BENCHMARKS COMPLETE ----")
            print(f"Ranks used : {stack.size}")
            print(f"GPU : {stack.use_gpu}")
            print(f"Backend : {BACKEND} ")
            for mol, energy in results.items():
                problem = ChemistryProblem.from_name(mol)
                fci= problem.fci_energy
                err_str = (f" error={abs(energy-fci):+.4f} Ha" if fci else "")
                print(f"{mol:6s}: {energy:+.6f} Ha{err_str} \n\n")



