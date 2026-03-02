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
        
        # if stack.rank == 0: print("\n--- RUNNING CHEMISTRY TASK ---")
        # chem_task = ChemistryProblem("H 0 0 0; H 0 0 0.74")  #hydrogen
        # # chem_task = ChemistryProblem("Li 0 0 0; H 0 0 1.59") #liH
        # res_chem = stack.run(chem_task)
        
        # if stack.rank == 0: print("\n--- RUNNING FINANCE TASK ---")
        # fin_task = FinanceProblem([[1, 0.5], [0.5, 1]])
        # res_fin = stack.run(fin_task)

        problem = ChemistryProblem("H 0 0 0; H 0 0 0.74")
        final_theta, history = stack.vqe_optimize(problem)

        if stack.rank == 0:
            print("UNIVERSAL STACK SUMMARY")
            # print(f"Problem Energy: {final_energy} Hartree")
            # print(f"Finance Riskbuild: {res_fin.energy:.6f} units")
            # print(f"HPC Path Used: {problem.used_path}")
            print(f"HPC execution complete across {stack.size} nodes.")


if __name__ == "__main__":
    run_universal_test()