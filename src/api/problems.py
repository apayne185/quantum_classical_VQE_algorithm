# standardizes all inputs to handle quantum chem, finance, optimization, etc
from abc import ABC, abstractmethod
import numpy as np 



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
    def __init__(self, molecule_str, distance):
        super().__init__()
        self.molecule = molecule_str
        self.dist = distance

    def prepare(self):
        # Here we will eventualy call qiskit_nature
        # rn we mock H2/LiH output
        self.pauli_terms = [("IIII", -0.8126), ("ZIII", -0.2252), ("IZII", 0.1723)]
        self.circuit_qasm = 'OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; h q[0];'
        print(f"[Chemistry] Prepared {self.molecule} at {self.dist}A")  



class FinanceProblem(QuantumProblem):
    def __init__(self, covariance_matrix):
        super().__init__()
        self.matrix = covariance_matrix

    def prepare(self):
        # Logic to convert Portfolio Optimization to Ising Hamiltonian
        self.pauli_terms = [("ZZII", 0.5), ("IZZI", 0.5), ("IIZZ", 0.5)]
        self.circuit_qasm = 'OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; x q[0];'
        print(f"[Finance] Prepared Portfolio QUBO from {len(self.matrix)} assets")