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
}


def serial_vqe(mol_name, coords, fci_ref, reps=1, max_iterations=500):
    driver = PySCFDriver(atom=coords, basis="sto-3g", unit=DistanceUnit.ANGSTROM)
    mol_problem = driver.run()
    hamiltonian = mol_problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op= mapper.map(hamiltonian)
    pauli_op= SparsePauliOp.from_list(qubit_op.to_list())
    n_qubits= qubit_op.num_qubits

    ansatz = EfficientSU2(n_qubits, reps=reps, entanglement="full").decompose()
    n_params = ansatz.num_parameters
    # Initialize near zero — keeps state close to HF reference |00...0>
    theta = np.random.uniform(-0.1, 0.1, n_params)


    def energy(t):
        bound = ansatz.assign_parameters({p: v for p, v in zip(sorted(ansatz.parameters,key=lambda x: x.name), t)})
        sv = Statevector(bound)

        return float(sv.expectation_value(pauli_op).real)


    # SPSA w same hyperparams as stack
    n_params_val = n_params
    c = 0.1
    a = 0.628 / np.sqrt(n_params_val / 8.0)
    A = max_iterations * 0.1
    alpha, gamma = 0.602, 0.101
    min_iters = max(20, n_params // 2)
    history = []
    t0 = time.perf_counter()

    for k in range(1, max_iterations + 1):
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        delta = np.random.choice([-1, 1], size=n_params)
        e_plus  = energy(theta + ck * delta)
        e_minus = energy(theta - ck * delta)
        current = (e_plus + e_minus) / 2.0
        history.append(current)
        gradient = (e_plus - e_minus) / (2 * ck * delta)
        theta -= ak * gradient
        if k >= min_iters and len(history) >= 10:
            spread = max(history[-10:]) - min(history[-10:])
            if spread < 1.6e-3:
                break

    t_total = time.perf_counter() - t0
    error = abs(history[-1] - fci_ref)
    print(f"[{mol_name}] Serial | E={history[-1]:+.6f} Ha | Error={error:+.4f} Ha | T={t_total:.3f}s | iters={len(history)}")  

    return history[-1], t_total, len(history)  




if __name__ == "__main__":
    print("----SERIAL QISKIT AER BASELINE--- ") 

    for name, (coords, fci, reps) in MOLECULES.items():
        serial_vqe(name, coords, fci, reps=reps)