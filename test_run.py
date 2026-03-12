from qiskit import qasm3
import sys
import os
import numpy as np
from src.api.interface import HPCHybridStack
from src.api.problems import ChemistryProblem, FinanceProblem

# so python can find C++ module
sys.path.insert(0, os.path.abspath("./build"))
# sys.path.append('./build/Release')    
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



def run_chemistry_test(stack: HPCHybridStack): 
    if stack.rank == 0: print("\n--- RUNNING CHEMISTRY TASK ---")

    chem_task = ChemistryProblem("Li 0 0 0; H 0 0 1.59") #liH
    theta, history = stack.vqe_optimize(chem_task, 
                                        max_iterations=50, 
                                        tolerance=1.6e-3, 
                                        checkpoint_dir="checkpoints")
    if stack.rank ==0 and history:
        print(f"[LiH] Final energy : {history[-1]:+.6f} Ha ")
        print(f"[LiH] Reference FCI : -7.882500 Ha")      #known value of LiH
        print(f"[LiH] Absolute error: {abs(history[-1] - (-7.8825)):+.6f} Ha ")
        print(f"[LiH] Iterations run: {len(history)}")


def run_finance_test(stack:HPCHybridStack): 
    if stack.rank == 0: print("\n--- RUNNING FINANCE (4-Assest Portfolio QUBO) TASK ---")
    np.random.seed(42)
    n_assets = 4
    #  Synthetic positive definite covariance matrix
    A = np.random.rand(n_assets, n_assets)
    cov = A @ A.T / n_assets
    mu  = np.random.uniform(0.02, 0.15, n_assets)

    problem = FinanceProblem(cov, expected_returns=mu, risk_factor=1.0)
    theta, history = stack.vqe_optimize(problem,
                                        max_iterations=30,
                                        tolerance=1e-3,
                                        checkpoint_dir="checkpoints",
    )

    if stack.rank == 0 and history:
        print(f"[Finance] Final objective : {history[-1]:+.6f} ")
        print(f"[Finance] Iterations run : {len(history)}")


  
def run_scaling_test(stack: HPCHybridStack):   
    if stack.rank == 0:
        print(f"\n[Scaling] Running with P={stack.size} ranks …")

    problem = ChemistryProblem("Li 0 0 0; H 0 0 1.59")  

    import time 
    t0 = time.perf_counter()
    
    _, history = stack.vqe_optimize(problem, max_iterations=10)
    t_total= time.perf_counter() - t0    

    if stack.rank == 0:
        print(f"[Scaling] P={stack.size} |  T_total={t_total:.3f}s |  final_E={history[-1]:+.6f}  ")



        


# def run_universal_test():
#     with HPCHybridStack(use_gpu=True) as stack:
#         if stack.rank == 0: print("\n--- RUNNING CHEMISTRY TASK ---")
#         # chem_task = ChemistryProblem("H 0 0 0; H 0 0 0.74")  #hydrogen
#         chem_task = ChemistryProblem("Li 0 0 0; H 0 0 1.59") #liH
#         theta_chem, hist_chem = stack.vqe_optimize(chem_task, max_iterations=20)
        
#         if stack.rank == 0: print("\n--- RUNNING FINANCE TASK ---")
#         fake_cov = np.random.rand(4, 4)
#         fin_task = FinanceProblem(fake_cov)
#         theta_fin, hist_fin = stack.vqe_optimize(fin_task, max_iterations=20)


#         if stack.rank == 0:
#             print("UNIVERSAL STACK SUMMARY")
#             print(f"Chemistry Final Energy: {hist_chem[-1]:.6f} Ha")
#             print(f"Finance Final Metric: {hist_fin[-1]:.6f}")
#             print(f"HPC execution complete across {stack.size} nodes.")


if __name__ == "__main__":
    backend = os.environ.get("BACKEND", "simulator")

    with HPCHybridStack(use_gpu=True, backend=backend) as stack:
        run_chemistry_test(stack)   
        run_finance_test(stack)   
        run_scaling_test(stack)   

        if stack.rank == 0:
            print("---ALL TESTS COMPLETE ----")
            print(f"Ranks used : {stack.size}")
            print(f"Backend : {backend} ")
