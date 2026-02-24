from qiskit import qasm3
import sys
import os
import numpy as np
from src.api.interface import HPCHybridStack
from src.api.problems import ChemistryProblem, FinanceProblem

# so python can find C++ module
sys.path.append('./build')
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




# # Simulates qiskit like object   -- placeholder while i setup 
# class FakeCircuit:
#     def __init__(self, qubits):
#         self.num_qubits = qubits

#     # def depth(self):
#     #     return 12             # dispatcher will use to estimate workload

#     def qasm(self): 
#         return "OPENQASM 3.0;  include \"stdgates.inc\"; qubit[20] q; h q[0];"


def run_universal_test():
    with HPCHybridStack(use_gpu=True) as stack:
        
        if stack.rank == 0: print("\n--- RUNNING CHEMISTRY TASK ---")
        chem_task = ChemistryProblem("H 0 0 0; H 0 0 0.74")  #hydrogen
        # chem_task = ChemistryProblem("Li 0 0 0; H 0 0 1.59") #liH
        res_chem = stack.run(chem_task)
        
        if stack.rank == 0: print("\n--- RUNNING FINANCE TASK ---")
        fin_task = FinanceProblem([[1, 0.5], [0.5, 1]])
        res_fin = stack.run(fin_task)

        if stack.rank == 0:
            print("UNIVERSAL STACK SUMMARY")
            print(f"Chemistry Energy: {res_chem.energy:.6f} Hartree")
            print(f"Finance Risk:     {res_fin.energy:.6f} units")
            print(f"HPC Path Used:    {res_chem.used_path}")





# def run_benchmarks():
#     stack = HPCHybridStack(use_gpu=True)   
    
#     #tests parameter batching dispatcher logic   
#     num_params = 1000 
#     params = np.random.random(num_params).tolist()
#     param_batch = [params]
#     qc = FakeCircuit(qubits=20) 

#     if stack.rank ==0: 
#         print("\n---- DISTRIBUTED STACK STRESS TEST - REPORT -----")
#         print(f"Cluster Size: {stack.size} nodes")
#         print(f"Qubits used: {qc.num_qubits}")
#         print(f"Precision: Mixed (FP32 Kernel -> FP64 Reduction)")


#     result = stack.run_vqe_batch(qc, param_batch, backend="hpc_cluster")

#     if stack.rank == 0: 
#         print("\n---RESULTS VERIFICATION----")
        
#         if "Distributed" in result.used_path:
#             print(f"PASS: Dispatcher routed to CUDA/MPI path.")
#         else:
#             print(f"FAIL: Dispatcher routed to: {result.used_path}")

#         expected_val = 1000.0      # local energy would be 1000*0.5 = 500, since we are using 2 ranks 500*2 = 1000
#         if result.energy != 0:
#             print(f"PASS: CUDA Kernel returned data.")
#             if abs(result.energy - expected_val) < 1e-5:
#                 print(f"PASS: Energy Result Correct ({result.energy:.2f})")
#             else:
#                 print(f"FAIL: Expected {expected_val}. Instead, returned: {result.energy:.2f}")
#         else:
#             print(f"FAIL: Energy return is 0.0 (Check the work)")


#         print(f"Result Variance: {result.variance}")
#         print(f"Wall Clock Time (T_total): {result.execution_time:.6f} s ")
#         # print(f"QCircuit: {stack.circuit_qasm}\n")
#         print(f"Status Message: {result.success_msg}")




if __name__ == "__main__":
    run_universal_test()