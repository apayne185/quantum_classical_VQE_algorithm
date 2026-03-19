import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath("./build"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    import hpc_core    
    from src.api.interface import HPCHybridStack
    from src.api.problems import ChemistryProblem, FinanceProblem
except ImportError as exc:  
    print(f"[FAIL] Import error: {exc}")
    print("Make sure 'make build' completed successfully.")
    sys.exit(1)

USE_GPU = os.environ.get("USE_GPU", "yes").strip().lower() == "yes"
BACKEND = os.environ.get("BACKEND", "simulator")
  

def section(title: str):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)




def test_mpi_layer(stack: HPCHybridStack):
    section("LAYER 1: MPI Bridge")
    if stack.rank == 0:
        print(f"Total MPI ranks: {stack.size}")
        print(f"This rank: {stack.rank}")
        print(f"GPU enabled: {stack.use_gpu}")
        print(f"Backend target: {stack.backend}")

    stack.comm.Barrier()
    if stack.rank == 0:
        print("MPI Barrier: PASSED")
        print("[LAYER 1] OK")




def test_problem_layer(stack: HPCHybridStack):
    section("LAYER 2: Problem Preparation")

    if stack.rank == 0:
        print("Building H2 molecule (smallest possible — fast to prepare) ...")

    problem = ChemistryProblem("H 0 0 0; H 0 0 0.74")
    problem.prepare()

    if stack.rank == 0:
        print(f"Pauli terms: {problem.get_pauli_count()}")
        print(f"Qubits: {problem.num_qubits}")
        print(f"Variational params: {problem.num_params}")
        print(f" QASM length: {len(problem.circuit_qasm)} chars")

        assert problem.num_qubits > 0,  "num_qubits not set"
        assert problem.num_params > 0,  "num_params not set"
        assert len(problem.circuit_qasm) > 0, "circuit_qasm is empty"
        assert len(problem.pauli_terms) > 0,  "pauli_terms is empty"

        print("\n  First 3 Pauli terms:")
        for op, coeff in problem.pauli_terms[:3]:
            print(f"{op}  :  {coeff:+.6f}")

        print("[LAYER 2] OK")

    return problem




def test_dispatcher_layer(stack: HPCHybridStack, problem: ChemistryProblem):
    section("LAYER 3: C++ Dispatcher (single dispatch)")
    num_params = problem.num_params
    theta= np.random.uniform(0, 2 * np.pi, num_params)
    ck = 0.1
    delta= np.random.choice([-1.0, 1.0], size=num_params)
    combined= np.concatenate([theta + ck * delta, theta - ck * delta])

    if stack.rank == 0:
        print(f"Dispatching combined params vector (len={len(combined)}) ...")

    t0= time.perf_counter()
    result = stack._evaluate(problem, combined, problem.num_qubits)
    t1 = time.perf_counter()

    if stack.rank == 0:
        print(f"E+: {result.energy:+.6f}")
        print(f"E-: {result.e_minus:+.6f}")
        print(f"Execution time: {result.execution_time:.4f} s")
        print(f"Masking metric M: {result.masking_metric:.4f}")
        print(f"Used path: {result.used_path}")
        print(f"Wall time (Python): {t1 - t0:.4f} s")
        print(f"Status: {result.success_msg}")

        assert result.success_msg == "OK", \
            f"Dispatcher returned non-OK status: {result.success_msg}"
        print("[LAYER 3] OK")



def test_vqe_loop(stack: HPCHybridStack):
    section("LAYER 4: VQE Loop (H2, 10 iterations)")

    problem = ChemistryProblem("H 0 0 0; H 0 0 0.74")

    t0 = time.perf_counter()
    theta, history = stack.vqe_optimize(
        problem,
        max_iterations=10,
        tolerance=1.6e-3,
        checkpoint_dir="checkpoints",
    )
    t1 = time.perf_counter()

    if stack.rank == 0:
        print(f"\n  Iterations run: {len(history)}")    
        print(f"Initial energy: {history[0]:+.6f} Ha")    
        print(f"Final energy: {history[-1]:+.6f} Ha")  
        print(f"Energy change: {history[-1] - history[0]:+.6f} Ha")
        print(f"Wall time: {t1 - t0:.2f} s")   
        print(f"Time per iter: {(t1 - t0) / len(history):.2f} s")

        assert np.isfinite(history[-1]),"Final energy is not finite"
        assert history[-1] < 100.0, "Final energy implausibly large"
        assert len(history) > 0,"No iterations recorded"

        print("[LAYER 4] OK")

    return history



# --- FINANCE PROBLEM (extensibility demonstration, not active in chemistry-focused benchmarks) ---
def test_finance_layer(stack: HPCHybridStack):
    section("LAYER 5: Finance QUBO (4 assets, 10 iterations)")
    np.random.seed(0)
    n= 4
    A = np.random.rand(n, n)
    cov = A @ A.T / n
    mu= np.array([0.05, 0.08, 0.12, 0.07])
    problem = FinanceProblem(cov, expected_returns=mu, risk_factor=1.0)

    t0 = time.perf_counter()
    _, history = stack.vqe_optimize(problem, max_iterations=10)
    t1 = time.perf_counter()

    if stack.rank == 0:
        print(f"Iterations run: {len(history)}")  
        print(f"Final objective: {history[-1]:+.6f}")  
        print(f"Wall time: {t1 - t0:.2f} s")
        assert np.isfinite(history[-1]), "Finance final value is not finite"
        print("[LAYER 5] OK")



def test_checkpoint_resilience(stack: HPCHybridStack):
    section("LAYER 6: Checkpoint Resilience")

    # Use a dedicated test directory to avoid conflicts with old checkpoints
    test_ckpt_dir = "checkpoints/_resilience_test"
    if stack.rank == 0:
        import shutil
        if os.path.exists(test_ckpt_dir):
            shutil.rmtree(test_ckpt_dir)

    stack.comm.Barrier()

    problem = ChemistryProblem("H 0 0 0; H 0 0 0.74")
    _, _ = stack.vqe_optimize(problem, max_iterations=5,
                               checkpoint_dir=test_ckpt_dir)

    if stack.rank == 0:
        ckpt = os.path.join(test_ckpt_dir, "checkpoint_iter_0005.npy")
        assert os.path.exists(ckpt), f"Expected checkpoint not found: {ckpt}"
        saved_theta = np.load(ckpt)
        print(f"Checkpoint file: {ckpt}")
        print(f"Saved theta shape: {saved_theta.shape}")
        print(f"Saved theta[:3]: {saved_theta[:3]}")


    if stack.rank == 0:
        print("\n  Restarting from checkpoint... ")

    problem2 = ChemistryProblem("H 0 0 0; H 0 0 0.74")
    ckpt_path = os.path.join(test_ckpt_dir, "checkpoint_iter_0005.npy")
    _, history2 = stack.vqe_optimize(
        problem2,
        max_iterations=5,
        restart_from=ckpt_path,
        checkpoint_dir=test_ckpt_dir,
    )

    if stack.rank == 0:
        assert len(history2) > 0, "No iterations after checkpoint restart"
        print(f"Post-restart iterations: {len(history2)}")   
        print("[LAYER 6] OK")



def print_summary(stack: HPCHybridStack, passed: list, failed: list):
    section("TRIAL RUN SUMMARY")
    if stack.rank != 0:  
        return 
    total = len(passed) + len(failed)   
    print(f"MPI ranks used: {stack.size}")   
    print(f"GPU enabled: {stack.use_gpu}")
    print(f"Backend: {stack.backend}")
    print(f"Tests passed: {len(passed)} / {total}")      
     
    for name in passed:
        print(f"[PASS] {name}")
    for name, err in failed:
        print(f"[FAIL] {name}  —  {err}")
    if not failed:
        print("\n  All layers verified, stack is ready.")
    else:  
        print("\n  Some layers failed ")





if __name__ == "__main__":
    passed = []
    failed = []

    with HPCHybridStack(use_gpu=USE_GPU, backend="simulator") as stack:

        tests = [
            ("MPI Bridge", lambda: test_mpi_layer(stack)),   
            ("Problem Preparation", lambda: test_problem_layer(stack)),
            ("C++ Dispatcher", lambda: test_dispatcher_layer(
                                          stack, ChemistryProblem("H 0 0 0; H 0 0 0.74"))),
            ("VQE Loop",lambda: test_vqe_loop(stack)),
            # ("Finance QUBO", lambda: test_finance_layer(stack)),  # Extensibility demo — uncomment to test
            ("Checkpoint Resilience", lambda: test_checkpoint_resilience(stack)),
        ]

        for name, fn in tests:
            try:
                if name == "C++ Dispatcher":
                    p = ChemistryProblem("H 0 0 0; H 0 0 0.74")
                    p.prepare()
                    test_dispatcher_layer(stack, p)
                else:    
                    fn()  
                passed.append(name)
            except Exception as exc:
                if stack.rank == 0:
                    print(f"[FAIL] {name}: {exc}")
                failed.append((name,str(exc)))   
        print_summary(stack,passed, failed)