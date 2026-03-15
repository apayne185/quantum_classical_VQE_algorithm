# The API, users interact with this class instead of C++, automates creation of object
import hpc_core  #compiled C++ module
from src.api.problems import QuantumProblem

import os
import sys
import numpy as np
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI


class HPCHybridStack:
    def __init__(self, use_gpu: bool | None = None, backend:str = 'simulator'):
        if use_gpu is None:
            env_val = os.environ.get("USE_GPU", "yes").strip().lower()
            use_gpu = (env_val == "yes") 

        self.use_gpu = use_gpu
        self.backend = backend

        #Init MPI environemnt using C++ bridge
        self.provided_thread_level = hpc_core.init_mpi()
        if not MPI.Is_initialized():
            MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = hpc_core.get_rank()
        self.size = hpc_core.get_size()

        # Assign 1 GPU per rank, round robin if < GPUs than ranks
        if self.use_gpu:
            try:
                hpc_core.set_cuda_device(self.rank)
            except Exception as e:
                if self.rank == 0: 
                    print(f"Rank {self.rank}: GPU initialization failed, falling back to CPU.  Error: {e}")
                self.use_gpu = False

        if self.rank == 0:
            print(f"[Stack]  Initialized {self.size} MPI rank(s), GPU={'enabled' if self.use_gpu else 'disabled'}, backend='{self.backend}'")


    # Runs SPSA VQE loop
    def vqe_optimize(self, problem: QuantumProblem, max_iterations:int=100, tolerance:int=1.6e-3, restart_from:str|None =None, checkpoint_dir: str = "checkpoints") -> tuple[np.ndarray, list[float]]:
        comm = self.comm
        problem.prepare()
        num_params = problem.num_params
        num_qubits  = problem.num_qubits 

        if num_params == 0:
            raise ValueError(
                "problem.num_params= 0 after prepare(), check that ansatz was built correctly. "
            )
        
        os.makedirs(checkpoint_dir, exist_ok=True)         # /checkpoints
        theta = np.zeros(num_params, dtype=np.float64)   

        if self.rank == 0:
            checkpoint_path = (restart_from if restart_from and os.path.exists(restart_from) else None)

            if checkpoint_path:
                print(f"[RESILIENCE] Loading θ from {checkpoint_path}...")
                theta = np.load(checkpoint_path).astype(np.float64)
                if theta.shape[0] != num_params:
                    raise ValueError(f"Checkpoint has {theta.shape[0]} params but problem has {num_params} params ")
            else:
                theta = np.random.uniform(0.0, 2*np.pi, num_params)
        else:
            theta = np.zeros(num_params)

        # Broadcast intials theta on Manager node, workers init empty arrays
        comm.Bcast(theta, root=0)

        


        # SPSA hyperparameters (standard coeffs)
        a, c, A = 0.628, 0.1, max_iterations * 0.1      # 0.6, 0.1, 10
        alpha, gamma = 0.602, 0.101
        history: list[float] = []
        prev_energy = float('inf')
        stop_signal = np.array([0], dtype=np.int32)


        for k in range(1, max_iterations +1): 
            stop_signal[0] = 0

            combined_params = np.zeros(num_params*2, dtype=np.float64)
            ck = np.float64(0.0)

            # Manager node updates the stochastic pertubation delta value
            if self.rank == 0:
                # Updates step sizes based on iteration  k
                ak = a / (k + A)**alpha
                ck = np.float64(c / k**gamma)
                delta = np.random.choice([-1.0, 1.0], size=num_params)
                combined_params[:num_params] = theta + ck * delta
                combined_params[num_params:] = theta - ck * delta
            # else:
            #     combined_params = np.zeros(num_params * 2)
            #     ck = 0
            comm.Bcast(combined_params, root=0)
            ck_arr = np.array([ck], dtype=np.float64)
            comm.Bcast(ck_arr, root=0)
            ck = ck_arr[0]    

            # Parallel expectation value estimation
            result = self.evaluate(problem, combined_params, num_qubits)

            # Parameters update - Manager only
            if self.rank == 0:
                # E+ is in energy, E- is in variance 
                e_plus = result.energy
                e_minus = result.e_minus
                current_energy = e_plus                 

                # Gradient aproximation
                gradient = (e_plus - e_minus) / (2*ck*delta)
                theta = theta - ak * gradient
                delta_e = abs(current_energy - prev_energy)       #energy difference 

                history.append(current_energy)
                prev_energy = current_energy
                
                print(f"Iter {k:04d} | Energy: {current_energy:.6f} | Delta: {delta_e:.2e} | M: {result.masking_metric:.8f} | Path: {result.used_path}") 

                if delta_e < tolerance:
                    print(f"Convergence Reached: {delta_e:.2e} < tol={tolerance} at iteration {k}")
                    stop_signal[0] = 1
                    # break

                if k % 5 == 0:
                    # np.save(f"/checkpoints/checkpoint_job_{k}.npy", theta)   
                    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{k:04d}.npy")
                    np.save(ckpt_path, theta)
                    print(f"[RESILIENCE] Iteration {k}: Global theta state checkpointed at path {ckpt_path}. ")   

            # self.comm.Barrier()
            comm.Bcast(theta, root=0)
            comm.Bcast(stop_signal, root=0)
            
            if stop_signal[0] == 1:
                break     

        return theta, history



    def __enter__(self): 
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.finalize()

    def finalize(self):
        # hpc_core.execute_barrier()
        hpc_core.finalize_mpi()     # clean MPI shutdown




    # for now, the middleware accepts input of problem types:  chemistry, finance, max_cut
    def evaluate(self, problem: QuantumProblem, combined_params: np.ndarray, num_qubits:int): 
        workload = hpc_core.HybridWorkload()
        workload.parameters = combined_params.tolist()
        workload.num_qubits = num_qubits
        workload.requires_gpu = self.use_gpu
        workload.circuit_qasm = problem.circuit_qasm
        workload.backend_target = self.backend    

        local_pauli_terms = self.partition(problem.pauli_terms)
        workload.pauli_terms = [hpc_core.PauliTerm(op, coeff) for op, coeff in local_pauli_terms]

        return hpc_core.execute(workload)


    def partition(self, full_list:list) -> list: 
        n= len(full_list)
        chunk = n // self.size
        start = self.rank * chunk
        end = (self.rank + 1) * chunk if self.rank != self.size - 1 else n
        return full_list[start:end]
