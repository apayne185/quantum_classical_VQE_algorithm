# The API, users interact with this class instead of C++, automates creation of object
import hpc_core  #compiled C++ module
import numpy as np
from src.api.problems import QuantumProblem

class HPCHybridStack:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        #setup MPI environemnt using C++ bridge
        self.provided_thread_level = hpc_core.init_mpi()
        self.rank = hpc_core.get_rank()
        self.size = hpc_core.get_size()

        if use_gpu:
            try:
                hpc_core.set_cuda_device(self.rank)
            except Exception as e:
                if self.rank == 0: 
                    print(f"Rank {self.rank}: GPU initialization failed, falling back to CPU.  Error: {e}")
                self.use_gpu = False


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
        return result


    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.finalize()


    def finalize(self):
        # hpc_core.execute_barrier()
        hpc_core.finalize_mpi()


    # def __del__(self):
    #         try:
    #             hpc_core.finalize_mpi()
    #         except:
    #             pass


    # for now, the middleware accepts input of problem types:  chemistry, finance, max_cut
    def run(self, problem: QuantumProblem): 
        problem.prepare()      #domain specific logic/math handled 
        terms = self.partition(problem.pauli_terms)

        workload = hpc_core.HybridWorkload()
        workload.parameters = [float(coeff) for _, coeff in terms]        
        workload.num_qubits = len(terms[0][0]) if terms else 0
        workload.requires_gpu = self.use_gpu
        workload.circuit_qasm = problem.circuit_qasm

        return hpc_core.execute(workload)


    def partition(self, full_list): 
        n= len(full_list)
        chunk = n// self.size
        start = self.rank * chunk
        end = (self.rank + 1) * chunk if self.rank != self.size - 1 else n
        return full_list[start:end]
