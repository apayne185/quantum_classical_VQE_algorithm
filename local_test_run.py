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




def run_chemistry_local(stack: HPCHybridStack): 
    if stack.rank == 0: print("\n--- RUNNING CHEMISTRY TASK (LiH Ground State) ---")

    problem = ChemistryProblem("Li 0 0 0; H 0 0 1.59") #liH
    t0 = time.perf_counter()
    theta, history = stack.vqe_optimize(problem, 
                                        max_iterations=50,        
                                        tolerance=1.6e-3, 
                                        checkpoint_dir="checkpoints") 
    
    t_total = time.perf_counter() - t0
    
    if stack.rank ==0 and history:
        print(f"[LiH] Final energy : {history[-1]:+.6f} Ha ")
        print(f"[LiH] Reference FCI : -7.882500 Ha")      #known value of LiH
        print(f"[LiH] Absolute error: {abs(history[-1] - (-7.8825)):+.6f} Ha ")
        print(f"[LiH] Iterations run: {len(history)}")
        print(f"[LiH] Wall time: {t_total:.2f}s (includes QPU queue + RTT) ")   
        print(f"[LiH] Time/iter: {t_total / len(history):.2f} s ")     




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

    problem = ChemistryProblem("Li 0 0 0; H 0 0 1.59")  
    t0 = time.perf_counter()
    
    _, history = stack.vqe_optimize(problem, max_iterations=10)
    t_total= time.perf_counter() - t0    

    if stack.rank == 0:
        print(f"[Scaling] P={stack.size} |  T_total={t_total:.3f}s |  final_E={history[-1]:+.6f}  ")



    

if __name__ == "__main__":
    with HPCHybridStack(use_gpu=USE_GPU, backend=BACKEND) as stack:
        run_chemistry_local(stack)   
        run_finance_local(stack)   
        run_scaling_local(stack)   

        if stack.rank == 0:
            print("---ALL LOCAL BENCHMARKS COMPLETE ----")
            print(f"Ranks used : {stack.size}")
            print(f"GPU : {stack.use_gpu}")
            print(f"Backend : {BACKEND} ")



