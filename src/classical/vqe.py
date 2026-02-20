
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import UGate
from qiskit.quantum_info import SparsePauliOp, Operator

import numpy as np
import matplotlib.pyplot as plt
from random import random



def hamiltonian_operator(a, b, c, d):
    #Creates a*I + b*Z + c*X + d*Y pauli sum = Hamiltonian operator.
    pauli_list = [("I", a),("Z", b),("X", c),("Y", d)]
    
    return SparsePauliOp.from_list(pauli_list)



def quantum_state_preparation(circuit, parameters):
    circuit.ry(parameters[1], 0) # parameter[1] = phi
    circuit.rz(parameters[0], 0) # parameter[0] = theta
    return circuit


def callback(nfev, x_next, f_next, stepsize, accepted):
    energy_history.append(f_next)


def vqe_circuit(parameters, measure):
    """
    Creates ansatz circuit for optimization.
    param parameters: list of parameters to construct ansatz state.
    param measure: measurement type ('X', 'Y', or 'Z').
    """
    
    q = QuantumRegister(1, name="q")
    c = ClassicalRegister(1, name="c")
    circuit = QuantumCircuit(q, c)

    # quantum state preparation
    circuit = quantum_state_preparation(circuit, parameters)

    # measurement basis transformation
    if measure == 'Z':
        # Z-basis= default measurement basis
        circuit.measure(q[0], c[0])
        
    elif measure == 'X':
        # Hgate maps X-basis to Z-basis
        circuit.h(q[0])
        circuit.measure(q[0], c[0])

    elif measure == 'Y':    
        # S-dagger rotates Y to X  (sdg)
        # Hgate rotates X to X 
        circuit.sdg(q[0]) 
        circuit.h(q[0])
        circuit.measure(q[0], c[0])
    else:
        raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')

    return circuit


def quantum_module(parameters, measure):
    # I operator always has expect. value of 1
    if measure == 'I':
        return 1
    
    if measure in ['X', 'Y', 'Z']:
        circuit = vqe_circuit(parameters, measure)
    else:
        raise ValueError('Not valid input: input should be "I", "X", "Y", or "Z"')
    
    
    # Transpile the circuit for specific backend and run
    t_circuit = transpile(circuit, backend, optimization_level=0)     #set to 0 as this is a light computational load
    job = backend.run(t_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    expectation_value = 0
    for measure_result, count in counts.items():
        sign = 1 if measure_result == '0' else -1
        expectation_value += sign * (count / shots)
        
    return expectation_value



def pauli_operator_to_dict(pauli_operator):
    """
    From a SparsePauliOp return a dict:
    {'I': 0.7, 'X': 0.6, 'Z': 0.1, 'Y': 0.5}.
    :param pauli_operator: qiskit.quantum_info.SparsePauliOp
    :return: a dict in the desired form.
    """

    labels = pauli_operator.paulis.to_labels()
    coeffs = pauli_operator.coeffs
    
    paulis_dict = {}
    for label, coeff in zip(labels, coeffs):
        paulis_dict[label] = coeff.real

    return paulis_dict

def vqe(parameters):
    """
    Calculates the total expectation value (energy) for the Hamiltonian.
    :param parameters: 1D array of parameters from the optimizer.
    :return: Total energy (float).
    """
    total_energy = 0
    
    # Iterate through the dict\
    for label, coefficient in pauli_dict.items():
        # Get expectaton value for specific Pauli string
        # <O> = coeff * <psi|O|psi>
        
        expectation_value = quantum_module(parameters, label)
        total_energy += coefficient * expectation_value
        
    return total_energy





if __name__ == "__main__": 
    shots = 8192
    backend = AerSimulator()

    #creating a,b,c,d from random real numbers bewtween [0,10]
    scale = 10
    a, b, c, d = (scale*random(), scale*random(), 
                scale*random(), scale*random())
    H = hamiltonian_operator(a, b, c, d)

    exact_result = NumPyMinimumEigensolver()
    result = exact_result.compute_minimum_eigenvalue(operator=H)
    reference_energy = result.eigenvalue.real
    print('(Reference) The exact ground state energy is: {}'.format(reference_energy))


    H_gate = Operator(UGate(np.pi/2, 0, np.pi)).data   #hadamard gate 
    print("H_gate:")
    print((H_gate * np.sqrt(2)).round(5))

    Y_gate = Operator(UGate(np.pi/2, 0, np.pi/2)).data    
    print("Y_basis_gate:")
    print((Y_gate * np.sqrt(2)).round(5))


    pauli_dict = pauli_operator_to_dict(H)
    parameters_array = np.array([np.pi, np.pi])
    energy_history = []

    spsa = SPSA(maxiter=15, callback=callback)

    vqe_result = spsa.minimize(fun=vqe,     #handles statistical noise better than gradient based optimizers 
                            x0=parameters_array)


    print(f'The exact ground state energy is: {reference_energy:.6f}')
    print(f'The VQE estimated energy is:{vqe_result.fun:.6f}')
    print(f'Accuracy Error: {abs(reference_energy - vqe_result.fun):.6f}')


    plt.plot(energy_history, label='SPSA Optimization')
    plt.axhline(y=reference_energy, color='r', linestyle='--', label='Exact Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.show()


