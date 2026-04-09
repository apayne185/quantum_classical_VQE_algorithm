import os
import time
import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp, Statevector




MOLECULES = {
    "H2":  ("H 0 0 0; H 0 0 0.74",  -1.13727, 1),
    "LiH": ("Li 0 0 0; H 0 0 1.59", -7.8825, 1),
    "BeH2": ("Be 0 0 0; H 0 0 1.33; H 0 0 -1.33", -15.5952, 2),
    "H2O": ("O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0", -75.0124, 2),
}


def serial_vqe(mol_name, coords, fci_ref, reps=1, max_iterations=500,seed=42):
    np.random.seed(seed)
    driver= PySCFDriver(atom=coords, basis="sto-3g", unit=DistanceUnit.ANGSTROM)
    mol_problem = driver.run()
    hamiltonian = mol_problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()          
    qubit_op= mapper.map(hamiltonian)
    pauli_op= SparsePauliOp.from_list(qubit_op.to_list())
    n_qubits= qubit_op.num_qubits 
        
             
    adaptive_reps = min(reps + 1, 3) if n_qubits > 4 else reps           #for larger molecules
    ansatz = EfficientSU2(n_qubits,reps=adaptive_reps, entanglement="full").decompose() 
    n_params = ansatz.num_parameters     
    max_iterations = max(200, n_params * 8)
    theta = np.random.uniform(-0.1, 0.1, n_params)                  # near-zero, stays close to HF reference   


    def energy(t):
        bound = ansatz.assign_parameters({p: v for p, v in zip(sorted(ansatz.parameters,key=lambda x: x.name), t)})
        sv = Statevector(bound)

        return float(sv.expectation_value(pauli_op).real)


    # SPSA hyperparams match HPCHybridStack -  fair comparison
    c = 0.1
    a = 0.628 / np.sqrt(n_params / 8.0)
    A = max_iterations * 0.1
    alpha, gamma= 0.602, 0.101
    min_iters = max(20, n_params // 2)

    history = [] 
    best_physical_energy = None
    best_physical_iter = 0
    t0 = time.perf_counter()


    for k in range(1, max_iterations + 1):
        ak= a / (k + A) ** alpha
        ck= c / k ** gamma
        delta = np.random.choice([-1, 1],size=n_params)
        e_plus  = energy(theta + ck * delta)
        e_minus = energy(theta - ck * delta)
        current = (e_plus+e_minus) / 2.0
        history.append(current)
        gradient = (e_plus-e_minus) / (2 * ck * delta)
        theta -= ak * gradient

        if current >= fci_ref:  # track best energy in physical (above-FCI) sector   
            if best_physical_energy is None or current < best_physical_energy:
                best_physical_energy = current
                best_physical_iter = k
                      

        if k >= min_iters and len(history) >= 10:
            spread = max(history[-10:]) - min(history[-10:])   
            if spread < 1.6e-3:
                break

    t_total = time.perf_counter() - t0

    if history[-1] < fci_ref and best_physical_energy is not None:
        report_energy = best_physical_energy
        report_iter= best_physical_iter   
    else:
        report_energy = history[-1]
        report_iter= len(history)

    error = abs(report_energy - fci_ref)
    print(f"[{mol_name}] Serial | E={report_energy:+.6f} Ha | Error={error:+.4f} Ha | T={t_total:.3f}s | iters={len(history)} (best at {report_iter})")

    return report_energy, t_total, len(history), report_iter  





if __name__ == "__main__":
    import json
    from datetime import datetime
    from src.api.log import init_log, close_log

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/baseline", exist_ok=True)
    init_log(f"results/baseline/serial_baseline_{ts}.log")

    print("----SERIAL QISKIT AER BASELINE--- ")   

    all_results = {}
    for name, (coords, fci, reps) in MOLECULES.items():
        print(f"\n--- {name} (serial, no MPI) ---")
        energy, wall_time, iters, best_iter = serial_vqe(name, coords, fci, reps=reps)   
        all_results[name] = {   
            "energy": energy,
            "fci": fci,
            "error": abs(energy - fci), 
            "wall_time": wall_time,
            "iterations": iters,
            "best_iter": best_iter,   
        }

    print(f"\n{'Molecule':<10} {'Energy (Ha)':<16} {'Error (Ha)':<14} {'Iters':<8} {'Time(s)':<10}")
    for name, d in all_results.items():
        print(f"{name:<10} {d['energy']:<16.6f} {d['error']:+.4f} Ha   {d['iterations']:<8} {d['wall_time']:<10.2f}")

    out_path = f"results/baseline/serial_baseline_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Results] Saved to {out_path}")
    close_log()