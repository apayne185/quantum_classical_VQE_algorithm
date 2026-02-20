from qiskit import qasm3
import sys
import os
import numpy as np

# so python can find C++ module
sys.path.append('./build')
sys.path.append('./build/Release')    
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.api.interface import HPCHybridStack
    import hpc_core
    print("hpc_core and interface imported.")
except ImportError as e:
    print(f"failed to import: {e}")
    sys.exit(1)





# Simulates qiskit like object   -- placeholder while i setup 
class FakeCircuit:
    def __init__(self, qubits):
        self.num_qubits = qubits

    # def depth(self):
    #     return 12             # dispatcher will use to estimate workload

    def qasm(self): 
        return "OPENQASM 3.0;  include \"stdgates.inc\"; qubit[20] q; h q[0];"
    


# def prepare_workload(qc, params, backend='simulator'):
#     # creates job/workload
#     stack = hpc_core.HybridWorkload()
#     stack.num_qubits = qc.num_qubits
#     stack.parameters = params
#     stack.circuit_depth = qc.depth()
#     stack.requires_gpu = True if qc.num_qubits > 15 else False
#     stack.backend_target = backend
#     # stack.circuit_qasm = qasm3.dumps(qc)    to be used in real run, ltaer stages
#     stack.circuit_qasm = "OPENQASM 3.0; gate x q; x $0;"

#     return stack


def run_benchmarks():
    stack = HPCHybridStack(use_gpu=True)   

    #tests parameter batching dispatcher logic   
    num_params = 1000 
    param_batch = [np.random.random(num_params).tolist()]

    qc = FakeCircuit(qubits=20)
    if stack.rank ==0: 
        print("\n----PHASE 1 & 2 STRESS TEST - REPORT -----")
        print(f"Cluster Size: {stack.size} nodes")
        print(f"Qubits used: {qc.num_qubits}")
        print(f"Precision: Mixed (FP32 Kernel -> FP64 Reduction)")


    result = stack.run_vqe_batch(qc, param_batch, backend="hpc_cluster")

    if stack.rank == 0: 
        print("\n---RESULTS VERIFICATION----")
        
        if "Distributed" in result.used_path:
            print(f"PASS: Dispatcher routed to CUDA/MPI path.")
        else:
            print(f"FAIL: Dispatcher routed to: {result.used_path}")

        if result.energy != 0:
            print(f"PASS: CUDA Kernel returned data: {result.energy:.6f}")
        else:
            print(f"FAIL: Energy return is 0.0 (Check work)")



        print(f"Stack returned result: {result}")
        print(f"Result Variance: {result.variance}")
        print(f"Wall Clock Time (T_total): {result.execution_time} s ")
        # print(f"QCircuit: {stack.circuit_qasm}\n")
        print(f"Status Message: {result.success_msg}")




if __name__ == "__main__":
    run_benchmarks()