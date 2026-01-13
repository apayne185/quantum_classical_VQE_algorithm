from qiskit import qasm3
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




hpc_core.init_mpi()
rank = hpc_core.get_rank()       #if 0, master, else worker
num_nodes = hpc_core.get_size()

qc = FakeCircuit()
params = []
if rank == 0:
    params = [0.5, 0.5, 0.5, 0.5]     #only master node defines data
stack = prepare_workload(qc, params)  

# triggers Dispatcher
result = hpc_core.execute(stack)


if rank ==0: 
    print("\n----MASTER VQE REPORT-----")
    print(f"Stack returned result: {result}")
    print(f"Result VQE energy: {result.energy}")
    print(f"Result Variance: {result.variance}")
    print(f"Backend Used: {result.used_path}")
    print(f"Time Taken: {result.execution_time} ms")
    print(f"QCircuit: {stack.circuit_qasm}\n")
    print(f"Nodes used: {num_nodes}")

hpc_core.finalize_mpi()