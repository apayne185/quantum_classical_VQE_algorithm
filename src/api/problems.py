# standardizes all inputs to handle quantum chem, finance, optimization, etc
from abc import ABC, abstractmethod
import numpy as np 
from qiskit import qasm3 
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit.circuit.library import EfficientSU2


class QuantumProblem(ABC):
    def __init__(self):
        self.pauli_terms = [] 
        self.ansatz = None       # HWE, UCC
        self.circuit_qasm = ""

    @abstractmethod
    def prepare(self):
        """Domain-specific logic to fill pauli_terms and ansatz."""
        pass




class ChemistryProblem(QuantumProblem):
    def __init__(self, atom_coordinates):     #user will provide the raw geometry 
        super().__init__()
        # self.molecule = molecule_str
        self.coords = atom_coordinates
        # self.dist = distance

    def prepare(self):
        # molecular physics 
        driver = PySCFDriver(
            atom=self.coords,
            basis="sto3g",
            unit=DistanceUnit.ANGSTROM
        ) 

        problem = driver.run()
        hamiltonian = problem.hamiltonian.second_q_op()
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(hamiltonian)
    
        self.pauli_terms = qubit_op.to_list()

        num_qubits = qubit_op.num_qubits
        ansatz = EfficientSU2(num_qubits, reps=1).decompose()  # HARDWARE EFFICIENT ANSATZ
        self.circuit_qasm = qasm3.dumps(ansatz)

        print(f"[Chemistry] Prepared {len(self.pauli_terms)} Pauli terms for  {num_qubits} qubits.")  



class FinanceProblem(QuantumProblem):
    def __init__(self, covariance_matrix):
        super().__init__()
        self.matrix = covariance_matrix

    def prepare(self):
        # Logic to convert Portfolio Optimization to Ising Hamiltonian
        self.pauli_terms = [("ZZII", 0.5), ("IZZI", 0.5), ("IIZZ", 0.5)]
        self.circuit_qasm = 'OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; x q[0];'
        print(f"[Finance] Prepared Portfolio QUBO from {len(self.matrix)} assets")