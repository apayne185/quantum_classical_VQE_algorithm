# The API, users interact with this class instead of C++, automates creation of object
import hpc_core  #compiled C++ module
import numpy as np

class HPCHybridStack:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

        #setup MPI environemnt using C++ bridge
        self.provided_thread_level = hpc_core.init_mpi()
        self.rank = hpc_core.get_rank()
        self.size = hpc_core.get_size()


    def run_vqe_batch(self, circuit, parameter_batch, backend="qpu"):
        # 'contract' object - stack_types.h
        workload = hpc_core.HybridWorkload()
        workload.num_qubits = circuit.num_qubits
        # workload.parameters = list(parameters)
        workload.parameters = [item for sublist in parameter_batch for item in sublist]
        workload.requires_gpu = self.use_gpu
        workload.backend_target = backend


        # done to map QASM from circuit  - PLACEHOLDER
        if hasattr(circuit, 'qasm'):
            workload.circuit_qasm = circuit.qasm()
        else:
            workload.circuit_qasm = "OPENQASM 3.0; // Simulated Circuit"

        
        # call C++ Bridge
        result = hpc_core.execute(workload)

        if self.rank == 0:
            self._display_report(result)
        

        return result
    

    def _display_report(self, result): 
        print(f"--- Execution Report ---")
        print(f"Target Path: {result.used_path}")
        print(f"Wall Time: {result.execution_time}s")
        print(f"VQE Energy: {result.energy}")
        print(f"Variance: {result.variance}")
        print(f"Status: {result.success_msg}")



def __del__(self):
        """Ensures clean MPI exit when the object is destroyed."""
        try:
            hpc_core.finalize_mpi()
        except:
            pass