# The API, users interact with this class instead of C++, automates creation of object
import hpc_core  #compiled C++ module
from src.api.problems import QuantumProblem

import os
import sys
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector 

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
    def vqe_optimize(self, problem: QuantumProblem, max_iterations:int=100, tolerance:float=1.6e-3, restart_from:str | None = None, checkpoint_dir: str = "checkpoints", start_iter: int = 0, seed: int | None = None) -> tuple[np.ndarray, list[float]]:
        self._below_fci_warned = False
        if seed is not None:
            np.random.seed(seed)
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
                import re
                match = re.search(r'checkpoint_iter_(\d+)', checkpoint_path)
                if match and start_iter == 0:
                    start_iter= int(match.group(1))
                    print(f"[RESILIENCE] Resuming SPSA schedle from iteration {start_iter}  ")
            else:
                # Initialize near zero — keeps state close to HF reference |00...0>
                # and avoids unphysical particle-number sectors
                theta = np.random.uniform(-0.1, 0.1, num_params)
        else:
            theta = np.zeros(num_params)

        # Broadcast intials theta on Manager node, workers init empty arrays
        comm.Bcast(theta, root=0)


        # SPSA hyperparameters — mildly scaled by parameter count
        # c: perturbation size (good for angles in [0, 2π])
        # a: step size, scaled down gently for high-dimensional problems
        # A: stability constant, delays aggressive early steps
        c = 0.1
        a = 0.628 / np.sqrt(num_params / 8.0)
        A = max_iterations * 0.1
        alpha, gamma = 0.602, 0.101
        min_iters_before_convergence = max(10, min(num_params * 2, 50))
        history: list[float] = []
        prev_energy = float('inf')
        stop_signal = np.array([0], dtype=np.int32)


        for loop_k in range(1, max_iterations +1):
            k = loop_k + start_iter
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

            comm.Bcast(combined_params, root=0)
            ck_arr = np.array([ck], dtype=np.float64)
            comm.Bcast(ck_arr, root=0)
            ck = ck_arr[0]

            # Parallel expectation value estimation
            if self.backend == "simulator":
                # Exact statevector simulation, distributed across MPI ranks
                e_plus, e_minus, masking_metric, used_path = self._evaluate_distributed_statevector(problem, combined_params)
            else:
                # C++ dispatcher path for IBM cloud QPU
                result = self._evaluate(problem, combined_params, num_qubits)
                e_plus = result.energy
                e_minus = result.e_minus
                used_path = result.used_path
                masking_metric = result.masking_metric

            # Parameters update - Manager only
            if self.rank == 0:
                # E(theta + ck*delta) and E(theta - ck*delta)
                current_energy = (e_plus + e_minus) / 2.0

                # Track best physically valid energy (at or above FCI)
                fci = getattr(problem, 'fci_energy', None)
                if fci is not None:
                    if current_energy >= fci - 1e-6:
                        # Valid energy — track best
                        if not hasattr(self, '_best_physical_energy') or current_energy < self._best_physical_energy:
                            self._best_physical_energy = current_energy
                            self._best_physical_theta = theta.copy()
                            self._best_physical_iter = k
                    elif not self._below_fci_warned:
                        print(f"NOTE: Energy {current_energy:.4f} crossed below FCI {fci:.4f} at iter {k} — HWE ansatz leaving physical sector (expected limitation)")
                        self._below_fci_warned = True

                # Gradient approximation
                gradient = (e_plus - e_minus) / (2*ck*delta)

                if not np.all(np.isfinite(gradient)):
                    print(f"WARNING: Non-finite gradient at iter {k}, skipping update")
                else:
                    theta = theta - ak * gradient

                delta_e = abs(current_energy - prev_energy)

                history.append(current_energy)
                prev_energy = current_energy

                print(f"Iter {k:04d} | Energy: {current_energy:.6f} | Delta: {delta_e:.2e} | M: {masking_metric:.4f} | Path: {used_path}")

                # Sliding-window convergence: only check after minimum iterations
                if loop_k >= min_iters_before_convergence and len(history) >= 10:
                    recent = history[-10:]
                    spread = max(recent) - min(recent)
                    if spread < tolerance:
                        print(f"Converged: energy spread over last 10 iters = {spread:.2e} < tol={tolerance}, at iteration {k}")
                        stop_signal[0] = 1

                if k % 5 == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{k:04d}.npy")
                    np.save(ckpt_path, theta)
                    print(f"[RESILIENCE] Iteration {k}: Global theta state checkpointed at path {ckpt_path}. ")

            comm.Bcast(theta, root=0)
            comm.Bcast(stop_signal, root=0)

            if stop_signal[0] == 1:
                break

        # Report best physically valid result if energy went below FCI
        if self.rank == 0 and fci is not None and hasattr(self, '_best_physical_energy'):
            if history[-1] < fci - 1e-6:
                best_e = self._best_physical_energy
                best_iter = self._best_physical_iter
                best_err = abs(best_e - fci)
                print(f"[VQE] Best physical energy: {best_e:.6f} Ha (error: {best_err:.4f} Ha) at iter {best_iter}")
                print(f"[VQE] Final energy {history[-1]:.6f} is below FCI — reporting best physical result")

        return theta, history



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def finalize(self):
        hpc_core.finalize_mpi()     # clean MPI shutdown




    # for now, the middleware accepts input of problem types:  chemistry, finance, max_cut
    def _evaluate(self, problem: QuantumProblem, combined_params: np.ndarray, num_qubits:int):
        workload = hpc_core.HybridWorkload()
        workload.parameters = combined_params.tolist()
        workload.num_qubits = num_qubits
        workload.requires_gpu = self.use_gpu
        workload.circuit_qasm = problem.circuit_qasm
        workload.backend_target = self.backend

        local_pauli_terms = self.partition(problem.pauli_terms)
        workload.pauli_terms = [hpc_core.PauliTerm(op, coeff) for op, coeff in local_pauli_terms]

        return hpc_core.execute(workload)


    def evaluate(self, problem, combined_params, num_qubits):
        return self._evaluate(problem, combined_params, num_qubits)


    def partition(self, full_list:list) -> list:
        n= len(full_list)
        chunk = n // self.size
        start = self.rank * chunk
        end = (self.rank + 1) * chunk if self.rank != self.size - 1 else n
        return full_list[start:end]




    def _evaluate_distributed_statevector(self, problem, combined_params):
        import time as _time
        comm = self.comm
        num_params = len(combined_params) // 2
        theta_plus = combined_params[:num_params]
        theta_minus = combined_params[num_params:]

        ansatz = problem.ansatz_circuit
        sorted_params = sorted(ansatz.parameters, key=lambda x: x.name)

        # T_accel: statevector build + Pauli expectation (the "acceleration" work)
        t_accel_start = _time.perf_counter()

        bound_plus = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted_params, theta_plus)})
        bound_minus = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted_params, theta_minus)})

        sv_plus = Statevector(bound_plus)
        sv_minus = Statevector(bound_minus)

        # Partition Pauli terms across MPI ranks
        local_terms = self.partition(problem.pauli_terms)

        if len(local_terms) > 0:
            local_op = SparsePauliOp.from_list(local_terms)
            e_plus_local = float(sv_plus.expectation_value(local_op).real)
            e_minus_local = float(sv_minus.expectation_value(local_op).real)
        else:
            e_plus_local = 0.0
            e_minus_local = 0.0

        t_accel = _time.perf_counter() - t_accel_start

        # T_comm: MPI_Allreduce communication time
        t_comm_start = _time.perf_counter()
        e_plus_global = np.array([0.0], dtype=np.float64)
        e_minus_global = np.array([0.0], dtype=np.float64)
        comm.Allreduce(np.array([e_plus_local], dtype=np.float64), e_plus_global, op=MPI.SUM)
        comm.Allreduce(np.array([e_minus_local], dtype=np.float64), e_minus_global, op=MPI.SUM)
        t_comm = _time.perf_counter() - t_comm_start

        # M = T_accel / T_comm — measures how well compute masks communication
        masking_metric = (t_accel / t_comm) if t_comm > 1e-9 else 0.0

        path = f"Statevector MPI-distributed ({self.size} ranks)"
        return float(e_plus_global[0]), float(e_minus_global[0]), masking_metric, path


    def _evaluate_statevector(self, problem, theta):
        """Single-rank exact statevector evaluation (kept for standalone use)."""
        ansatz = problem.ansatz_circuit
        bound = ansatz.assign_parameters(
            {p: v for p, v in zip(sorted(ansatz.parameters, key=lambda x: x.name), theta)})

        sv = Statevector(bound)
        pauli_op = SparsePauliOp.from_list(problem.pauli_terms)
        energy = sv.expectation_value(pauli_op).real
        return float(energy)
