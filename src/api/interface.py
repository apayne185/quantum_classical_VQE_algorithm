# The API, users interact with this class instead of C++, automates creation of object
import hpc_core  #compiled C++ module

class HPCHybridStack:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def run(self, circuit, parameters):
        # 'contract' object - stack_types.h
        workload = hpc_core.HybridWorkload()
        workload.num_qubits = circuit.num_qubits
        workload.parameters = list(parameters)
        workload.requires_gpu = self.use_gpu
        
        # call C++ Bridge
        result = hpc_core.execute(workload)

        # print(f"--- Execution Report ---")
        # print(f"Target Path: {result.used_path}")
        # print(f"Wall Time:   {result.execution_time}s")
        # print(f"VQE Energy:  {result.energy}")
        # print(f"Variance:   {result.variance}")

        return result.energy