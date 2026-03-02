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
    import hpc_core
    print("hpc_core and interface imported.")
except ImportError as e:
    print(f"failed to import hpc_core module: {e}")
    sys.exit(1)



def run_universal_test():
    with HPCHybridStack(use_gpu=True) as stack:
        if stack.rank == 0: print("\n--- RUNNING CHEMISTRY TASK ---")
        # chem_task = ChemistryProblem("H 0 0 0; H 0 0 0.74")  #hydrogen
        chem_task = ChemistryProblem("Li 0 0 0; H 0 0 1.59") #liH
        theta_chem, hist_chem = stack.vqe_optimize(chem_task, max_iterations=20)
        
        if stack.rank == 0: print("\n--- RUNNING FINANCE TASK ---")
        fake_cov = np.random.rand(4, 4)
        fin_task = FinanceProblem(fake_cov)
        theta_fin, hist_fin = stack.vqe_optimize(fin_task, max_iterations=20)


        if stack.rank == 0:
            print("UNIVERSAL STACK SUMMARY")
            print(f"Chemistry Final Energy: {hist_chem[-1]:.6f} Ha")
            print(f"Finance Final Metric: {hist_fin[-1]:.6f}")
            print(f"HPC execution complete across {stack.size} nodes.")


if __name__ == "__main__":
    run_universal_test()