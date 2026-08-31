from abc import ABC, abstractmethod
import numpy as np
from qiskit import qasm3
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit.circuit.library import EfficientSU2, TwoLocal



ANSATZ_TIERS = {
    "hwe":{"label": "HWE (shallow)", "color": "green"},
    "hwe_adaptive": {"label": "HWE (adaptive)", "color": "amber"},
    "uccsd": {"label": "UCCSD (particle-conserving)", "color": "blue"},
}

 
def estimate_correlation_strength(pauli_terms: list) -> dict:    
    if not pauli_terms:
        return {"correlation_score": 0.0, "recommended_tier": "hwe","reasoning": "No Pauli terms — defaulting to HWE."}
 
    coeffs = np.array([abs(c) for _, c in pauli_terms])
 
    diagonal_coeffs = []   
    off_diagonal_coeffs= []  
    two_body_coeffs= []   
 
    for op, coeff in pauli_terms:
        has_xy = any(c in ('X', 'Y') for c in op)
        zz_like= op.count('Z') >= 2 and not has_xy   
        if has_xy:
            off_diagonal_coeffs.append(abs(coeff))
        else:
            diagonal_coeffs.append(abs(coeff))  

        if zz_like or (has_xy and sum(1 for c in op if c != 'I') >=2):
            two_body_coeffs.append(abs(coeff))
 
    max_diag= max(diagonal_coeffs) if diagonal_coeffs else 1e-12    
    max_off_diag = max(off_diagonal_coeffs) if off_diagonal_coeffs else 0.0
    mean_coeff= np.mean(coeffs) if len(coeffs) > 0 else 1.0
 
    off_diag_ratio= max_off_diag / max_diag
    spectral_spread = (coeffs.max()- coeffs.min()) / (mean_coeff + 1e-12)
    zz_fraction=len(two_body_coeffs) / max(len(pauli_terms), 1)

    score = (
        0.50 * min(off_diag_ratio / 2.0,1.0) +        # off-diag coupling
        0.30 * min(spectral_spread / 10.0, 1.0) +           # energy spread
        0.20 * min(zz_fraction / 0.5, 1.0)               # 2-body density
    )
    score = float(np.clip(score, 0.0, 1.0))
 
    if score < 0.25:
        tier = "hwe"
        reason = (f"score={score:.3f} < 0.25. Off-diagonal coupling is weak (ratio={off_diag_ratio:.3f}). Single-reference character, HWE ansatz is appropriate.")
    elif score < 0.55:
        tier = "hwe_adaptive"
        reason = (f"score={score:.3f} in [0.25, 0.55). Moderate correlation (off_diag_ratio={off_diag_ratio:.3f}, zz_frac={zz_fraction:.3f}). Adaptive HWE with full entanglement recommended.")
    else:
        tier = "uccsd"
        reason = (f"score={score:.3f} >= 0.55. Strong correlation detected (off_diag_ratio={off_diag_ratio:.3f}, zz_frac={zz_fraction:.3f}). UCCSD particle-conserving ansatz recommended.")
 
    return {
        "off_diag_ratio":off_diag_ratio,
        "spectral_spread":spectral_spread,
        "zz_fraction": zz_fraction,
        "correlation_score":score,
        "recommended_tier": tier,
        "reasoning":reason,
    }
 
 
def build_ansatz(num_qubits: int, tier: str, reps: int, mol_problem=None, mapper=None):
    # Builds ansatz circuit for the given tier 
    # for 'uccsd' tier, mol_problem and mapper are required to construct UCCSD ansatz with HF initial state
    if tier == "hwe":
        ansatz = EfficientSU2(
            num_qubits,
            reps=reps,
            entanglement="linear",
        ).decompose()

    elif tier == "hwe_adaptive":
        adaptive_reps = min(reps + 1, 3)
        # full entanglement generates n*(n-1)/2 CNOTs per layer -- 435 at n=30.
        # Combined with .decompose() and QASM3 serialization this dominates
        # wall-clock at large n (>100 min Python-side on CO2 30q). Fall back
        # to linear entanglement above the threshold: same expressibility class
        # (HWE is not universal either way), 15x fewer gates.
        entanglement = "full" if num_qubits <= 20 else "linear"
        ansatz = EfficientSU2(
            num_qubits,
            reps=adaptive_reps,
            entanglement=entanglement,
        ).decompose()

    elif tier == "uccsd" and mol_problem is not None and mapper is not None:
        from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

        num_particles = mol_problem.num_particles
        num_spatial_orbitals = mol_problem.num_spatial_orbitals

        if num_qubits <= 12:        # UCCSD depth grows fast, therefore impractical beyond 12
            try:
                hf_init = HartreeFock(
                    num_spatial_orbitals=num_spatial_orbitals,
                    num_particles=num_particles,
                    qubit_mapper=mapper,
                )

                ansatz = UCCSD(
                    num_spatial_orbitals=num_spatial_orbitals,
                    num_particles=num_particles,
                    qubit_mapper=mapper,
                    initial_state=hf_init,
                    reps=min(reps, 1),
                )
                # Do not decompose - breaks UCC parameter ties and particle-number conservation
                print(f"[Ansatz] UCCSD built: {ansatz.num_parameters} UCC amplitudes, {num_particles} particles, {num_spatial_orbitals} spatial orbitals")
            except Exception as e:
                print(f"[Ansatz] UCCSD construction failed: {e} - falling back to HWE-adaptive")
                adaptive_reps = min(reps + 1, 3)
                ansatz = EfficientSU2(
                    num_qubits,
                    reps=adaptive_reps,
                    entanglement="full",
                ).decompose()
        else:
            print(f"[Ansatz] UCCSD impractical for {num_qubits} qubits - using HWE-adaptive instead")
            adaptive_reps = min(reps + 1, 3)
            ansatz = EfficientSU2(
                num_qubits,
                reps=adaptive_reps,
                entanglement="full",
            ).decompose()

    else:              # fallback for non chemistry problems/missing mol_problem
        ucc_reps = min(reps, 2)
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            entanglement="full",
            reps=ucc_reps,
            skip_final_rotation_layer=False,
        ).decompose()

    return ansatz, ansatz.num_parameters




# STO-3G basis, FCI energies from PySCF
MOLECULE_REGISTRY = {     
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
    "N2": {
        "geometry": "N 0 0 0; N 0 0 1.09",
        "fci_energy": -108.9544,
        "reps": 2,
        "description": "Nitrogen (N2), 14 electrons, 20 qubits (GPU crossover test)",
    },
    "CO2": {
        "geometry": "C 0 0 0; O 0 0 1.16; O 0 0 -1.16",
        "fci_energy": -187.6,
        "reps": 1,
        "description": "Carbon dioxide (CO2), 22 electrons, 30 qubits (ceiling test, not a convergence run: 16k Pauli terms x default max_iters is multi-day; use MAX_ITERS<=10)",
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
 



class QuantumProblem(ABC):
    def __init__(self):
        self.pauli_terms:list[tuple[str, float]]  = [] 
        self.circuit_qasm: str = ""
        self.num_qubits: int = 0
        self.num_params: int = 0
        self._prepared:bool = False

    @abstractmethod
    def prepare(self):
        """Domain specific logic to fill pauli_terms, num params, num_qubits"""
        pass

    def get_pauli_count(self) -> int:
        return len(self.pauli_terms)
    
    def get_num_params(self) -> int:
        return self.num_params
    
    def get_num_qubits(self) -> int:
        return self.num_qubits



class ChemistryProblem(QuantumProblem):
    def __init__(self, atom_coordinates:str, reps: int=1, name:str = "custom", force_tier: str | None=None):
        super().__init__()
        self.coords = atom_coordinates
        self.reps= reps
        self.name = name
        self.fci_energy = None 
        self.force_tier = force_tier
        self.diagnostics: dict = {}
        self.ansatz_tier: str= "hwe"


    @classmethod
    def from_name(cls, molecule_name:str, force_tier: str | None=None)  -> "ChemistryProblem":
        key = molecule_name.strip()

        if key not in MOLECULE_REGISTRY:
            supported = list(MOLECULE_REGISTRY.keys())
            raise ValueError(f"Unknown molecule '{key}', not within known MOLECULE_REGISTRY. Supported: {supported}")

        entry = MOLECULE_REGISTRY[key]
        problem = cls(atom_coordinates = entry["geometry"], reps=entry["reps"], name=key,force_tier=force_tier,)
        problem.fci_energy = entry["fci_energy"]
        return problem 
    


    def prepare(self):
        if self._prepared:
            return

        # Per-phase timestamps -- CO2 sat silent for >100 min in ansatz build
        # with no way to tell if the process was frozen or making progress.
        # These give a heartbeat so users can tell live progress from a hang.
        import time as _t
        _t0 = _t.perf_counter()
        def _phase(msg):
            print(f"[Chemistry:{self.name}] [+{_t.perf_counter() - _t0:6.1f}s] {msg}",
                  flush=True)

        _phase("driver.run() -- PySCF electronic structure")
        driver = PySCFDriver(atom=self.coords, basis="sto-3g", unit=DistanceUnit.ANGSTROM)
        mol_problem = driver.run()

        _phase("second_q_op() + JW mapping")
        hamiltonian = mol_problem.hamiltonian.second_q_op()
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(hamiltonian)

        raw = qubit_op.to_list()        # [("IIZI", (-0.81+0j)), ...]
        self.pauli_terms = [(op, float(coeff.real)) for op, coeff in raw]

        self.num_qubits = qubit_op.num_qubits
        _phase(f"Hamiltonian: {len(self.pauli_terms)} Pauli terms, {self.num_qubits} qubits")

        self.diagnostics = estimate_correlation_strength(self.pauli_terms)


        _phase("FCI ground-truth via PySCF")
        # Compute FCI ground truth  (will override registry value if available)
        try:
            from pyscf import gto, fci as pyscf_fci
            pyscf_mol = gto.M(atom=self.coords.replace(";", "\n"), basis="sto-3g", unit="Angstrom")
            pyscf_mol.build()
            mf = pyscf_mol.RHF().run(verbose=0)
            fci_solver = pyscf_fci.FCI(mf)
            fci_e, _ = fci_solver.kernel()
            self.fci_energy = float(fci_e)
        except Exception as e:
            print(f"[Chemistry:{self.name}] FCI computation failed: {e}, using registry value")

        if self.force_tier is not None:
            self.ansatz_tier = self.force_tier
            tier_source = "forced"
        else:
            self.ansatz_tier = self.diagnostics["recommended_tier"]
            tier_source = "auto"

        _phase(f"build_ansatz(tier={self.ansatz_tier}, reps={self.reps}) -- this is the slow one at high n")
        ansatz, n_params = build_ansatz(
            self.num_qubits, self.ansatz_tier, self.reps,
            mol_problem=mol_problem, mapper=mapper)
        self.num_params = n_params
        self.ansatz_circuit = ansatz


        _phase(f"QASM3 serialization ({n_params} params, {ansatz.size()} gates)")
        try:        # QASM export can fail for nondecomposed UCCSD
            self.circuit_qasm = qasm3.dumps(ansatz)
        except Exception:
            self.circuit_qasm = qasm3.dumps(ansatz.decompose())
        _phase("prepare() complete")
        fci_str = (f"FCI reference: {self.fci_energy:.4f} Ha" if self.fci_energy is not None else "")
        tier_label = ANSATZ_TIERS.get(self.ansatz_tier, {}).get("label", self.ansatz_tier)


        print(f"[Chemistry:{self.name}] {len(self.pauli_terms)} Pauli terms, {self.num_qubits} qubits, {self.num_params} params, reps={self.reps}. {fci_str} ")
        print(f"[Ansatz] {tier_label} ({tier_source}) | corr_score={self.diagnostics['correlation_score']:.3f} | off_diag_ratio={self.diagnostics['off_diag_ratio']:.3f}")
        print(f"[Reason] {self.diagnostics['reasoning']}")

        self._prepared = True




    def energy_error(self, computed_energy:float) -> float |None:
            if self.fci_energy is None:
                return None
            return abs(computed_energy-self.fci_energy)  
    

    def get_ansatz_summary(self) -> dict:
        return {
            "molecule":self.name,
            "tier":self.ansatz_tier,
            "tier_label":ANSATZ_TIERS.get(self.ansatz_tier, {}).get("label", self.ansatz_tier),
            "num_qubits": self.num_qubits,
            "num_params": self.num_params,
            "corr_score": self.diagnostics.get("correlation_score", None),
            "off_diag_ratio": self.diagnostics.get("off_diag_ratio", None),
            "fci_energy":self.fci_energy,
        }





class FinanceProblem(QuantumProblem):
    # Portfolio optimization with QUBO - Ising mapping.
    # Demonstrates stack beyond chemistry. Not included in currest benchmarking
    def __init__(self, covariance_matrix: np.ndarray, expected_returns: np.ndarray|None = None, risk_factor: float=1.0):
        super().__init__()
        self.matrix = np.array(covariance_matrix, dtype=float)
        n = self.matrix.shape[0]    
        self.returns =(np.zeros(n) if expected_returns is None else np.array(expected_returns, dtype=float))
        self.risk_factor = risk_factor
        self.num_qubits = n    


    def prepare(self):
        if self._prepared:                 
            return 
        
        # QUBO -> Ising: map mean-variance portfolio to Pauli Hamiltonian
        n = self.num_qubits
        sigma = self.matrix
        mu = self.returns
        lam = self.risk_factor   
        pauli_terms: list[tuple[str, float]] = []


        for i in range(n):  # diagonal Zi terms
            zi_coeff = 0.5* mu[i] - 0.5 * lam *sigma[i,i]
            op = "I" * i + "Z" + "I" * (n-i-1)
            if abs(zi_coeff) > 1e-12:
                pauli_terms.append((op, zi_coeff))

        for i in range(n):  # off-diagonal ZiZj terms
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
        self.ansatz_circuit = ansatz
        self.num_params = ansatz.num_parameters

        print(f"[Finance] Prepared {len(self.pauli_terms)} Pauli terms for {n}-asset portfolio ({self.num_qubits} qubits,  {self.num_params} params)  ")  
        
        self._prepared = True 

