from qiskit import qasm3
import sys
import os
import numpy as np
import time
from datetime import datetime


# so python can find C++ module
sys.path.insert(0, os.path.abspath("./build"))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.api.interface import HPCHybridStack
    from src.api.problems import ChemistryProblem, FinanceProblem
    from src.api.results import save_results
    from src.api.log import init_log, log, close_log
    import hpc_core
    print("hpc_core and interface imported.")
except ImportError as e:
    print(f"failed to import hpc_core module: {e}")
    sys.exit(1)


USE_GPU = os.environ.get("USE_GPU", "yes").strip().lower() == "yes"
BACKEND = os.environ.get("BACKEND", "ibm_cloud")



def check_credentials():
    token= os.environ.get("IBM_QUANTUM_TOKEN", "")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "")
    backend= os.environ.get("IBM_QUANTUM_BACKEND", "ibm_brisbane")
    region = os.environ.get("IBM_QUANTUM_REGION", "us-east")

    if not token or not instance:
        print("[ERROR] IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE must be set w/in .env.")
        print("Add to your .env file and run 'make run-ibm'")
        sys.exit(1)

    print(f"[IBM] Token: {'*' * 8}{token[-4:]} ")
    print(f"[IBM] Instance: {instance[:40]}... ")
    print(f"[IBM] Backend: {backend}")
    print(f"[IBM] Region: {region}  ")



def run_chemistry_ibm(stack: HPCHybridStack):
    if stack.rank == 0: print("\n--- RUNNING IBM QPU CHEMISTRY TASK (H2 Ground State) ---")

    problem = ChemistryProblem.from_name("H2")
    t0 = time.perf_counter()
    theta, history = stack.vqe_optimize(problem,
                                        max_iterations=10,
                                        tolerance=1.6e-3,
                                        checkpoint_dir="checkpoints/H2_ibm",
                                        seed=42)

    t_total = time.perf_counter() - t0

    result = None
    if stack.rank == 0 and history:
        fci = problem.fci_energy
        final_e = history[-1]
        error = abs(final_e - fci) if fci else None
        print(f"\n[H2-IBM] Final energy : {final_e:+.6f} Ha ")
        print(f"[H2-IBM] Reference FCI : {fci:+.6f} Ha")
        print(f"[H2-IBM] Absolute error: {error:+.6f} Ha " if error else "")
        print(f"[H2-IBM] Iterations run: {len(history)}")
        print(f"[H2-IBM] Wall time: {t_total:.2f}s (includes QPU queue + RTT)")
        print(f"[H2-IBM] Time/iter: {t_total/len(history):.2f}s")
        print(f"[H2-IBM] Path: IBM EstimatorV2 (Python)")

        result = {
            "molecule": "H2",
            "energy": final_e,
            "fci": fci,
            "error": error,
            "iterations": len(history),
            "wall_time": t_total,
            "time_per_iter": t_total / len(history),
            "history": history,
        }
    return result



def run_finance_ibm(stack: HPCHybridStack):
    if stack.rank == 0:
        print("\n--- RUNNING FINANCE (4-Asset Portfolio QUBO) on IBM QPU ---")
        print("[WARNING] This uses QPU time. Each iteration submits a real job.")
    np.random.seed(42)
    n_assets = 4
    A = np.random.rand(n_assets, n_assets)
    cov = A @ A.T / n_assets
    mu  = np.random.uniform(0.02, 0.15, n_assets)

    problem = FinanceProblem(cov, expected_returns=mu, risk_factor=1.0)
    t0 = time.perf_counter()
    _, history = stack.vqe_optimize(problem,
                                        max_iterations=15,
                                        tolerance=1e-3,
                                        checkpoint_dir="checkpoints/finance_ibm",
    )
    t_total = time.perf_counter() - t0

    result = None
    if stack.rank == 0 and history:
        print(f"[Finance-IBM] Final objective : {history[-1]:+.6f} ")
        print(f"[Finance-IBM] Iterations run : {len(history)}")
        print(f"[Finance-IBM] Wall time : {t_total:.2f}s")
        result = {
            "final_objective": history[-1],
            "iterations": len(history),
            "wall_time": t_total,
            "n_assets": n_assets,
            "history": history,
        }
    return result



def run_scaling_ibm(stack: HPCHybridStack):
    if stack.rank == 0:
        print(f"\n[Scaling-IBM] Running with P={stack.size} ranks ...")
        print("[WARNING] This uses QPU time.")

    problem = ChemistryProblem.from_name("LiH")
    t0 = time.perf_counter()

    _, history = stack.vqe_optimize(problem, max_iterations=5,
                                    checkpoint_dir="checkpoints/scaling_ibm")
    t_total = time.perf_counter() - t0

    result = None
    if stack.rank == 0 and history:
        print(f"[Scaling-IBM] P={stack.size} | T_total={t_total:.3f}s | final_E={history[-1]:+.6f}")
        result = {
            "ranks": stack.size,
            "wall_time": t_total,
            "final_energy": history[-1],
            "iterations": len(history),
        }
    return result




if __name__ == "__main__":
    check_credentials()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    init_log(f"results/ibm_run_{ts}.log")

    print(f"[Config] GPU= {'requested' if USE_GPU else 'CPU mode'} ")
    print(f"[Config] Backend= {BACKEND} ")
    print("[Config] WARNING: This run submits real jobs to IBM Quantum.")
    print(" Each iteration incurs QPU time. Monitor usage at  https://quantum.cloud.ibm.com \n\n")

    with HPCHybridStack(use_gpu=USE_GPU, backend=BACKEND) as stack:
        chem_result = run_chemistry_ibm(stack)
        finance_result = run_finance_ibm(stack)
        scaling_result = run_scaling_ibm(stack)

        if stack.rank == 0:
            print("\n---ALL IBM QPU BENCHMARKS COMPLETE ----")
            print(f"Ranks used : {stack.size}")
            print(f"GPU : {stack.use_gpu}")
            print(f"Backend : {BACKEND} ")

            save_results({
                "mpi_ranks": stack.size,
                "gpu": stack.use_gpu,
                "chemistry": chem_result,
                "finance": finance_result,
                "scaling": scaling_result,
            }, backend=BACKEND)

    close_log()
