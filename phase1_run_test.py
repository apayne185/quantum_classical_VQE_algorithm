import sys
import os
# sys.path.append('./build')    # for C++ modile

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    import hpc_core
    print("hpc_core imported.")
except ImportError as e:
    print(f"failed to import: {e}")
    sys.exit(1)



# Simulates qiskit like object   -- placeholder while i setup 
class FakeCircuit:
    def __init__(self):
        self.num_qubits = 20

# creates job/workload
stack = hpc_core.HybridWorkload()
stack.num_qubits = 20
stack.parameters = [0.1, 0.5, 0.9]
stack.circuit_qasm = "OPENQASM 3.0; include 'stdgates.inc'; ..."     # From qc.qasm()
stack.requires_gpu = True

# triggers Dispatcher
result = hpc_core.execute(stack)
print(f"Stack returned result: {result}")
print(f"Result VQE energy: {result.energy}")
print(f"Backend Used: {result.used_path}")
print(f"Time Taken: {result.execution_time} ms")