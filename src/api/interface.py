# src/api/interface.py
import hpc_core  #compiled C++ module

class HPCHybridStack:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def run(self, circuit, parameters):
        # 'Contract' object
        workload = hpc_core.HybridWorkload()
        workload.num_qubits = circuit.num_qubits
        workload.parameters = list(parameters)
        workload.requires_gpu = self.use_gpu
        
        # Call C++ Bridge
        result = hpc_core.execute(workload)
        return result