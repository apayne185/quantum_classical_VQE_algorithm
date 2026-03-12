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
        self.pauli_terms:list[tuple[str, float]]  = [] 
        # self.ansatz = None       # HWE, UCC
        self.circuit_qasm: str = ""
        self.num_qubits: int = 0
        self.num_params: int = 0

    @abstractmethod
    def prepare(self):
        """Domain-specific logic to fill pauli_terms, num params, num_qubits"""
        pass

    def get_pauli_count(self) -> int:
        return len(self.pauli_terms)
    
    def get_num_params(self) -> int:
        return self.num_params
    
    def get_num_qubits(self) -> int:
        return self.num_qubits
    





class ChemistryProblem(QuantumProblem):
    def __init__(self, atom_coordinates):     #user will provide the raw geometry 
        super().__init__()
        self.coords = atom_coordinates

    def prepare(self):
        # molecular physics 
        driver = PySCFDriver(
            atom=self.coords,
            basis="sto-3g",
            unit=DistanceUnit.ANGSTROM,
        ) 

        problem = driver.run()
        hamiltonian = problem.hamiltonian.second_q_op()
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(hamiltonian)


        raw = qubit_op.to_list()                                      #[("IIZI", (- 0.81+0j)), ..]     
        # self.pauli_terms = qubit_op.to_list()
        self.pauli_terms =[(op, float(coeff.real)) for op, coeff in raw]

        self.num_qubits = qubit_op.num_qubits
        ansatz = EfficientSU2(self.num_qubits, reps=1).decompose()              # HARDWARE EFFICIENT ANSATZ  (HWE)
        self.circuit_qasm = qasm3.dumps(ansatz)
        self.num_params = ansatz.num_parameters

        print(f"[Chemistry] Prepared {len(self.pauli_terms)} Pauli terms for {self.num_qubits} qubits,  {self.num_params} variational params.")  


    # def get_pauli_strings(self):
    #     # pass   #will return list of operatorsz, weights
    #     return [("IIZI", -0.81), ("IZIZ", 0.17)]
    
    # def get_ansatz(self): 
    #     pass   #returns qiskit circuit for the domain 

    # def get_pauli_count(self):
    #     return len(self.pauli_terms)



class FinanceProblem(QuantumProblem):
    def __init__(self, covariance_matrix: np.ndarray, expected_returns: np.ndarray|None = None, risk_factor: float=1.0):
        super().__init__()
        self.matrix = np.array(covariance_matrix, dtype=float)
        n = self.matrix.shape[0]    
        self.returns =(np.zeros(n) if expected_returns is None else np.array(expected_returns, dtype=float))
        self.risk_factor = risk_factor
        self.num_qubits = n    


    def prepare(self):
        # Logic to map Mean-Varince Portfolio Optimization to Ising Hamiltonian using QUBO formulation
        n = self.num_qubits
        sigma = self.matrix
        mu = self.returns
        lam = self.risk_factor   
        pauli_terms: list[tuple[str, float]] = []


        for i in range(n):              # Diagonal terms - Zi coefficients 
            zi_coeff = 0.5* mu[i] - 0.5 * lam *sigma[i,i]
            op = "I" * i +"Z" + "I" * (n-i-1)
            if abs(zi_coeff) > 1e-12:
                pauli_terms.append((op, zi_coeff))

        for i in range(n):               # Off diagonal terms - Zi Zj coefficients 
            for j in range(i+1, n): 
                zizj_coeff = 0.25*lam * sigma[i,j]
                if abs(zizj_coeff) > 1e-12:
                    op = (
                        "I"* i + "Z"
                        +"I" * (j-i-1) + "Z"
                        +"I" * (n-j-1)
                    )
                    pauli_terms.append((op, zizj_coeff))

        self.pauli_terms = pauli_terms
        ansatz = EfficientSU2(n, reps=1).decompose()
        self.circuit_qasm = qasm3.dumps(ansatz)
        self.num_params = ansatz.num_paramaters

        # self.pauli_terms = [("ZIII", 1.0), ("IZII", 2.0), ("IIZI", 3.0), ("IIIZ", 4.0)]
        # self.circuit_qasm = 'OPENQASM 3.0; include "stdgates.inc"; qubit[4] q; h q; s q[0];'
        # print(f"[Finance] Prepared Portfolio QUBO from {len(self.matrix)} assets")
        print(f"[Finance] Prepared {len(self.pauli_terms)} Pauli terms for {n}-asset portfolio ({self.num_qubits} qubits,  {self.num_params} params)  ")


        
    # def get_pauli_strings(self):
    #     # pass   #will return list of operatorsz, weights
    #     return [("ZZII", 0.5), ("IZZI", 0.5)]     #qubo/ising mapping logic - need to make this more applicable agnostic 
    
    # def get_ansatz(self): 
    #     pass   #returns qiskit circuit for the domain 

    # def get_pauli_count(self):
    #     return len(self.pauli_terms)
