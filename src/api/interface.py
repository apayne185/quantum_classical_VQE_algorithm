# The API, users interact with this class instead of C++, automates creation of object
import hpc_core  #compiled C++ module
import numpy as np
from src.api.problems import QuantumProblem
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI


class HPCHybridStack:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        #setup MPI environemnt using C++ bridge
        self.provided_thread_level = hpc_core.init_mpi()
        if not MPI.Is_initialized():
            MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = hpc_core.get_rank()
        self.size = hpc_core.get_size()

        if use_gpu:
            try:
                hpc_core.set_cuda_device(self.rank)
            except Exception as e:
                if self.rank == 0: 
                    print(f"Rank {self.rank}: GPU initialization failed, falling back to CPU.  Error: {e}")
                self.use_gpu = False



    def vqe_optimize(self, problem: QuantumProblem, max_iterations=100,tolerance=1.6e-3):
        comm = MPI.COMM_WORLD
        problem.prepare()
        num_params = len(problem.pauli_terms[0][0])

        # init theta on Manager node, workers init empty arrays
        theta = np.random.uniform(0, 2*np.pi, num_params) if self.rank == 0 else np.zeros(num_params)

        prev_energy = float('inf')
        # SPSA hyperparameters (standard coeffs)
        a, c, A = 0.01, 0.01, 10       # 0.6, 0.1, 10
        alpha, gamma = 0.602, 0.101
        history = []

        for k in range(1, max_iterations +1): 
            stop_signal = np.array([0], dtype=np.int32)

            # Manager node updates the stochastic pertubation delta value
            if self.rank == 0:
                # Updates step sizes based on iteration  k
                ak = a / (k + A)**alpha
                ck = c / k**gamma
                delta = np.random.choice([-1, 1], size=num_params)
                combined_params = np.concatenate([theta + ck * delta, theta - ck * delta])
            else:
                combined_params = np.zeros(num_params * 2)
                # delta = np.zeros(num_params)
                ck = 0

            # Parallel expectation value estimation
            result = self.evaluate(problem, combined_params, num_qubits=4)

            # Parameters update - Manager only
            if self.rank == 0:
                # E+ is in energy, E- is in variance (the hack we made in C++)
                e_plus = result.energy
                e_minus = result.variance
                current_energy = result.energy
                history.append(current_energy)
                
                delta_e = abs(current_energy - prev_energy)       #energy difference 
                
                if delta_e < tolerance:
                    print(f"Convergence Reached: {delta_e:.6f} < {tolerance}")
                    stop_signal[0] = 1
                    # break

                # Gradient aproximation
                gradient = (e_plus - e_minus) / (2*ck*delta)
                theta = theta - ak*gradient
                prev_energy = current_energy

                # Masking efficiency M stored in res.variance 
                # avg_m = (res_plus.variance +res_minus.variance) / 2
                # history.append(res_plus.energy)
                
                print(f"Iter {k:03} | Energy: {current_energy:.6f} | Delta: {delta_e:.6f} | M: {result.masking_metric:.8f}")  

            comm.Bcast(stop_signal, root=0)
            
            if stop_signal[0] == 1:
                break     

        # final_energy = current_energy.energy

        return theta, history



    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.finalize()

    def finalize(self):
        # hpc_core.execute_barrier()
        hpc_core.finalize_mpi()




    # for now, the middleware accepts input of problem types:  chemistry, finance, max_cut
    def evaluate(self, problem, params, num_qubits): 
        workload = hpc_core.HybridWorkload()
        workload.parameters = params.tolist()
        workload.num_qubits = num_qubits
        workload.requires_gpu = self.use_gpu
        workload.circuit_qasm = problem.circuit_qasm

        return hpc_core.execute(workload)


    def partition(self, full_list): 
        n= len(full_list)
        chunk = n// self.size
        start = self.rank * chunk
        end = (self.rank + 1) * chunk if self.rank != self.size - 1 else n
        return full_list[start:end]
