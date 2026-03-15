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
    


# MOLECULE REGISTRY - reference FCIs (hartrees), recommended ansatz reps - increasingly more complex
# Source - STO-3G basis 
MOLECULE_REGISTRY= {     
    "H2": {
        "geometry": "H 0 0 0; H 0 0 0.74",     
        "fci_energy": -1.13727,
        "reps": 1,  
        "description": "Hydrogen molecule (H2), 2 electrons, 4 qubits ",
    },
    "LiH": {
        "geometry": "Li 0 0 0; H 0 0 1.59",
        "fci_energy": -7.8825,
        "reps": 1,
        "description": "Lithium hydride (LiH), 4 electrons, 12 qubits ",
    },
    "BeH2": {
        "geometry": "Be 0 0 0; H 0 0 1.33; H 0 0 -1.33",
        "fci_energy": -15.5952,
        "reps":2,
        "description": "Beryllium hydride (BeH2), 6 electrons, 14 qubits",
    },    
    "H2O": {
        "geometry": "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
        "fci_energy": -75.0129,
        "reps": 2,
        "description": "Water (H2O), 8 electrons, 14 qubits",
    },
    "NH3":{
        "geometry": (
            "N 0 0 0.116; "
            "H 0 0.931 -0.269; "
            "H 0.807 -0.466 -0.269; "
            "H -0.807 -0.466 -0.269"
        ),     

        "fci_energy":-55.4546,
        "reps": 3,
        "description": "Ammonia- 10 electrons, 16 qubits (NISQ upper limit.) ",
    },   
}   
 




class ChemistryProblem(QuantumProblem):
    def __init__(self, atom_coordinates:str, reps: int=1, name:str = "custom" ):     #user will provide the raw geometry 
        super().__init__()
        self.coords = atom_coordinates
        self.reps= reps
        self.name = name
        self.fci_energy = None 


    @classmethod
    def from_name(cls, molecule_name:str):
        key = molecule_name.strip()
        if key not in MOLECULE_REGISTRY:
            supported = list(MOLECULE_REGISTRY.keys())
            raise ValueError({"Unknown molecule '{key}', not within known MOLECULE_REGISTRY "})

        entry = MOLECULE_REGISTRY[key]
        problem = cls(atom_coordinates = entry["geometry"], reps=entry["reps"], name=key)
        problem.fci_energy = entry["fci_entry"]
        return problem 
    




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


        raw = qubit_op.to_list()                                 #[("IIZI", (- 0.81+0j)), ..]     
        self.pauli_terms =[(op, float(coeff.real)) for op, coeff in raw]

        self.num_qubits = qubit_op.num_qubits
        ansatz = EfficientSU2(self.num_qubits, reps=self.reps).decompose()              # HARDWARE EFFICIENT ANSATZ  (HWE)
        self.circuit_qasm = qasm3.dumps(ansatz)
        self.num_params = ansatz.num_parameters
        fci_str = (f"FCI reference: {self.fci_energy:.4f} Ha" if self.fci_energy is not None else "") 
        print(f"[Chemistry:{self.name}] {len(self.pauli_terms)} Pauli terms, {self.num_qubits} qubits, {self.num_params} params,  reps={self.reps}.{fci_str} ")  
        # print(f"[Chemistry] Prepared {len(self.pauli_terms)} Pauli terms for {self.num_qubits} qubits,  {self.num_params} variational params.")  


        def energy_error(self, computed_energy:float) -> float |None:
            if self.fci_energy is None:
                return None
            return abs(computed_energy-self.fci_energy)





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
            op = "I" * i + "Z" + "I" * (n-i-1)
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
        self.num_params = ansatz.num_parameters

        print(f"[Finance] Prepared {len(self.pauli_terms)} Pauli terms for {n}-asset portfolio ({self.num_qubits} qubits,  {self.num_params} params)  ")


