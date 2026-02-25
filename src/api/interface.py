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



    def vqe_optimize(self, problem: QuantumProblem, iterations=20):
        problem.prepare()
        num_params = len(problem.pauli_terms[0][0])

        # init theta on Manager node, workers init empty arrays
        theta = np.random.uniform(0, 2*np.pi, num_params) if self.rank == 0 else np.zeros(num_params)

        # SPSA hyperparameters (standard coeffs)
        a, c, A = 0.6, 0.1, 10
        alpha, gamma = 0.602, 0.101
        history = []

        for k in range(1, iterations +1): 
            # Updates step sizes based on iteration  k
            ak = a / (k + A)**alpha
            ck = c / k**gamma

            # Manager node updates the stochastic pertubation delta value
            if self.rank == 0:
                delta = np.random.choice([-1, 1], size=num_params)
                theta_plus = theta + ck * delta
                theta_minus = theta - ck * delta
            else:
                theta_plus = np.zeros(num_params)
                theta_minus = np.zeros(num_params)

            # Parallel expectation value estimation
            # Evaluate  E(theta+delta)
            res_plus = self.evaluate(problem, theta_plus)
            
            # Evaluate E(theta-delta)   
            res_minus = self.evaluate(problem, theta_minus)

            # Parameters update - Manager only
            if self.rank == 0:
                # Gradient aproximation
                gradient = (res_plus.energy-res_minus.energy) / (2*ck*delta)
                theta = theta - ak*gradient

                # Masking efficiency M stored in res.variance 
                avg_m = (res_plus.variance +res_minus.variance) / 2
                history.append(res_plus.energy)
                
                print(f"Iter {k:03} |  Energy: {res_plus.energy:.6f} |  Masking M: {avg_m:.8f}")
       

        # # done to map QASM from circuit  - PLACEHOLDER
        # if hasattr(circuit, 'qasm'):
        #     workload.circuit_qasm = circuit.qasm()
        # else:
        #     workload.circuit_qasm = "OPENQASM 3.0; // Simulated Circuit"

        
        # # call C++ Bridge
        # result = hpc_core.execute(workload)
        # return result
        final_energy = res_plus.energy

        return theta, history, final_energy




    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.finalize()

    def finalize(self):
        # hpc_core.execute_barrier()
        hpc_core.finalize_mpi()




    # for now, the middleware accepts input of problem types:  chemistry, finance, max_cut
    def evaluate(self, problem, params): 
        workload = hpc_core.HybridWorkload()
        # workload.parameters = [float(coeff.real) for _, coeff in terms]     
        workload.parameters = params.tolist()   
        # workload.num_qubits = len(terms[0][0]) if terms else 0
        workload.num_qubits = len(params)
        workload.requires_gpu = self.use_gpu
        workload.circuit_qasm = problem.circuit_qasm

        return hpc_core.execute(workload)


    def partition(self, full_list): 
        n= len(full_list)
        chunk = n// self.size
        start = self.rank * chunk
        end = (self.rank + 1) * chunk if self.rank != self.size - 1 else n
        return full_list[start:end]
