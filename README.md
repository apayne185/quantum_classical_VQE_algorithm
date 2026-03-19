# Hybrid Quantum-Classical VQE Stack for HPC

**Bachelors of Computer Science and Artificial Intelligence (BCSAI) Thesis — IE University**
**Anna Payne**

A hybrid quantum classical software/middlware stack that implements the Variational Quantum Eigensolver (VQE) with MPI parallelism, CUDA GPU acceleration, and IBM Quantum cloud integration. Designed primarily for molecular ground state energy computation.

---

## Quick Start

Run the commands below

```bash
git clone <repo-url> && cd quantum_classical_VQE_algorithm
cp .env.example .env          # Add IBM Quantum token (optional, only for QPU runs)
make build                      # Build Docker image (CUDA 12.2 + OpenMPI + Python 3.11)   
make trial                      # 5-layer diagnostic test (simulator, 2 MPI ranks)
make run NP=2 MOLECULES="H2 LiH"   # Full benchmark 
```

## Architecture

### Three Layer Stack

```
Python API (src/api/)
   ↓ QuantumProblem.prepare() to OpenQASM + Pauli Hamiltonian
C++ Dispatcher (src/dispatcher/)
   ↓ MPI broadcast to local compute to MPI_Allreduce
CUDA Kernels (src/classical/cuda/)
   ↓ Mixed-precision FP32 to FP64 Pauli expectations
```

**Data flow in stack:** `QuantumProblem.prepare()` builds an ansatz circuit and Pauli Hamiltonian. `HPCHybridStack.vqe_optimize()` runs the SPSA loop, where each iteration evaluates E(θ±) through statevector simulation (MPI distributed), the IBM QPU (EstimatorV2), or the C++ dispatcher ()`MPI_Allreduce` sums partial energies and handles  gradient update).

### Key Classes   

| Class | File | Role |
|-------|------|------|
| `HPCHybridStack` | `src/api/interface.py` | Main entry point: MPI init, SPSA loop, checkpoints |
| `ChemistryProblem` | `src/api/problems.py` | Molecular Hamiltonian via PySCF + Jordan-Wigner |
| `MoleculeResolver` | `src/api/molecule_resolver.py` | Registry, geometry, SMILES, PubChem cascade |
| `HybridWorkload` | `include/stack_types.h` | C++ dispatcher interface contract |

The stack will be developed in the future to be extensible to other problem domains (`FinanceProblem` is currently in progress for portfolio optimization using QUBO, where Ising mapping is included as a demonstration).   


## Makefile Targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make trial` | 5 layer diagnostic (simulator, 2 ranks) |
| `make run NP=4` | Full benchmark with MPI (simulator) |
| `make run-ibm NP=2` | Runs on IBM Quantum QPU |
| `make scaling` | Strong scaling sweep (P=1,2,4,8)  |
| `make baseline` | Serial Qiskit VQE for accuracy comparison |
| `make test` | Runs all tests |
| `make shell` | Interactive shell inside container |
| `make clean` | Removes image and logs | 


## IBM Quantum Setup

1. Get an instance API token at [quantum.ibm.com](https://quantum.ibm.com)
2. Copy `.env.example` to `.env` and fill in your credentials (all can be found on IBM site):
   ```
   IBM_QUANTUM_TOKEN=your_token_here
   IBM_QUANTUM_INSTANCE=your-crn-instance
   IBM_QUANTUM_BACKEND=ibm-location
   IBM_QUANTUM_REGION=us-east or eu-de
   ```  
3. Run: `make run-ibm NP=2`

This stack uses `EstimatorV2` with `mode=backend` (which is compatible with qiskit-ibm-runtime v0.45.1 open plan, but does not support Sessions), 4096 shots, and TREX measurement error mitigation (resilience_level=1).   

## Supported (Tested) Molecules

| Molecule | Qubits | FCI Energy (Ha) | Recommended Ansatz | Notes |
|----------|--------|------------------|--------------------|-------|
| H₂ | 4 | -1.1373 | HWE (shallow) | Fastest, good for QPU testing |
| LiH | 12 | -7.8825 | UCCSD / HWE-adaptive | Moderate correlation |
| BeH₂ | 14 | -15.5952 | HWE-adaptive | 6 electrons |
| H₂O | 14 | -75.0129 | HWE-adaptive | 8 electrons |

Custom molecules are supported within the `MoleculeResolver` (SMILES, PubChem lookup, or raw geometry).   

## Results

Results will be saved after runs as timestamped JSON files in `results/` including its git commit hash, per-molecule energies, convergence histories, and timing data. Analyze the data with:

```bash
python benchmarks/run_analysis.py          # summary table   
python benchmarks/run_analysis.py --plot   # convergence plots (requires matplotlib)  
```

## Dependencies

**Core:** qiskit ≥1.0, qiskit-nature, qiskit-ibm-runtime, pyscf, mpi4py, numpy, scipy
**Build:** CMake 3.18+, pybind11, nlohmann_json, libcurl, OpenMPI, CUDA 12.2
**Optional:** cupy (GPU), rdkit (SMILES), matplotlib (plots)

All dependencies are included in the Docker image. See the file `requirements.txt` for Python packages needed.

## Project Structure

```
├── src/
│   ├── api/              # Python API (interface, problems, resolver, results, log)   
│   ├── dispatcher/       # C++ MPI coordinator + IBM QPU client 
│   └── classical/cuda/   # CUDA kernels for Pauli expectations
├── include/              # C++ headers (stack_types.h)
├── tests/                # Test suite (layer tests, molecule tests)
├── benchmarks/           # Entry points for experiments
│   ├── local_test_run.py # Simulator benchmark
│   ├── ibm_test_run.py   # IBM Quantum benchmark
│   ├── serial_baseline.py# Serial Qiskit reference
│   └── run_analysis.py   # Results analysis + plotting
├── results/              # Timestamped JSON results + scaling data
├── Dockerfile            # CUDA 12.2 + OpenMPI container
├── Makefile              # Build orchestration
└── CMakeLists.txt        # C++ build config   
```
