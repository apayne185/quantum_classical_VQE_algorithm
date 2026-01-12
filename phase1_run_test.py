import sys
import os
sys.path.append('./build/Release')    # for C++ modile
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    import hpc_core
    print("hpc_core imported.")
except ImportError as e:
    print(f"failed to import: {e}")
    sys.exit(1)

from qiskit import qasm3




# Simulates qiskit like object   -- placeholder while i setup 
class FakeCircuit:
    def __init__(self):
        self.num_qubits = 20

    def depth(self):
        return 12             # dispatcher will use to estimate workload


def prepare_workload(qc, params, backend='simulator'):
    # creates job/workload
    stack = hpc_core.HybridWorkload()
    stack.num_qubits = qc.num_qubits
    stack.parameters = params
    stack.circuit_depth = qc.depth()
    stack.requires_gpu = True if qc.num_qubits > 15 else False
    stack.backend_target = backend
    # stack.circuit_qasm = qasm3.dumps(qc)    to be used in real run, ltaer stages
    stack.circuit_qasm = "OPENQASM 3.0; gate x q; x $0;"

    return stack


qc = FakeCircuit()
params = [0.1, 0.2, 0.3]
stack = prepare_workload(qc, params)  

# triggers Dispatcher
result = hpc_core.execute(stack)
print(f"Stack returned result: {result}")
print(f"Result VQE energy: {result.energy}")
print(f"Result Variance: {result.variance}")
print(f"Backend Used: {result.used_path}")
print(f"Time Taken: {result.execution_time} ms")