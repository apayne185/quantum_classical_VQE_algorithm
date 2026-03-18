# Hybrid Quantum-Classical VQE Stack for HPC

**Bachelor's Thesis — CSAI, IE University**
**Anna Payne**

A hybrid quantum-classical software stack that implements the Variational Quantum Eigensolver (VQE) with MPI parallelism, CUDA GPU acceleration, and IBM Quantum cloud integration. Designed for molecular ground-state energy computation and portfolio optimization.

---

## Quick Start

```bash
git clone <repo-url> && cd quantum_classical_VQE_algorithm
cp .env.example .env          # Add your IBM Quantum token (optional, for QPU runs)
make build                    # Build Docker image (CUDA 12.2 + OpenMPI + Python 3.11)
make trial                    # 6-layer diagnostic test (simulator, 2 MPI ranks)
make run NP=2 MOLECULES="H2 LiH"   # Full benchmark
```

## Architecture

### Three-Layer Stack

```
Python API (src/api/)
   ↓ QuantumProblem.prepare() → OpenQASM + Pauli Hamiltonian
C++ Dispatcher (src/dispatcher/)
   ↓ MPI broadcast → local compute → MPI_Allreduce
CUDA Kernels (src/classical/cuda/)
   ↓ Mixed-precision FP32→FP64 Pauli expectations
```

**Data flow:** `QuantumProblem.prepare()` builds an ansatz circuit and Pauli Hamiltonian → `HPCHybridStack.vqe_optimize()` runs the SPSA loop → each iteration evaluates E(θ±) via statevector simulation (MPI-distributed), IBM QPU (EstimatorV2), or C++ dispatcher → `MPI_Allreduce` sums partial energies → gradient update.

### Key Classes

| Class | File | Role |
|-------|------|------|
| `HPCHybridStack` | `src/api/interface.py` | Main entry point: MPI init, SPSA loop, checkpoints |
| `ChemistryProblem` | `src/api/problems.py` | Molecular Hamiltonian via PySCF + Jordan-Wigner |
| `FinanceProblem` | `src/api/problems.py` | Portfolio optimization (QUBO → Ising) |
| `MoleculeResolver` | `src/api/molecule_resolver.py` | Registry → geometry → SMILES → PubChem cascade |
| `HybridWorkload` | `include/stack_types.h` | C++ dispatcher interface contract |

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make trial` | 6-layer diagnostic (simulator, 2 ranks) |
| `make run NP=4` | Full benchmark with MPI (simulator) |
| `make run-ibm NP=2` | Run on IBM Quantum QPU |
| `make scaling` | Strong scaling sweep (P=1,2,4,8) |
| `make baseline` | Serial Qiskit VQE for accuracy comparison |
| `make test` | Run all tests |
| `make shell` | Interactive shell inside container |
| `make clean` | Remove image and logs |

## IBM Quantum Setup

1. Get a token at [quantum.ibm.com](https://quantum.ibm.com)
2. Copy `.env.example` to `.env` and fill in your credentials:
   ```
   IBM_QUANTUM_TOKEN=your_token_here
   IBM_QUANTUM_INSTANCE=ibm-q/open/main
   IBM_QUANTUM_BACKEND=ibm_brisbane
   IBM_QUANTUM_REGION=us-east
   ```
3. Run: `make run-ibm NP=2`

The stack uses `EstimatorV2` with Session mode (keeps QPU allocated), 4096 shots, and TREX measurement error mitigation (resilience_level=1).

## Supported Molecules

| Molecule | Qubits | FCI Energy (Ha) | Recommended Ansatz | Notes |
|----------|--------|------------------|--------------------|-------|
| H₂ | 4 | -1.1373 | HWE (shallow) | Fastest, good for QPU testing |
| LiH | 12 | -7.8825 | UCCSD / HWE-adaptive | Moderate correlation |
| BeH₂ | 14 | -15.5952 | HWE-adaptive | 6 electrons |
| H₂O | 14 | -75.0129 | HWE-adaptive | 8 electrons |
| NH₃ | 16 | -55.4546 | HWE-adaptive | NISQ upper limit |

Custom molecules are supported via the `MoleculeResolver` (SMILES, PubChem lookup, or raw geometry).

## Results

Results are saved as JSON in `results/` with timestamps, git commit hash, per-molecule energies, convergence histories, and timing data. Analyze with:

```bash
python run_analysis.py          # Summary table
python run_analysis.py --plot   # + convergence plots (requires matplotlib)
```

## Dependencies

**Core:** qiskit ≥1.0, qiskit-nature, qiskit-ibm-runtime, pyscf, mpi4py, numpy, scipy
**Build:** CMake 3.18+, pybind11, nlohmann_json, libcurl, OpenMPI, CUDA 12.2
**Optional:** cupy (GPU), rdkit (SMILES), matplotlib (plots)

All dependencies are included in the Docker image. See `requirements.txt` for Python packages.

## Project Structure

```
├── src/
│   ├── api/              # Python API (interface, problems, resolver, results, log)
│   ├── dispatcher/       # C++ MPI coordinator + IBM QPU client
│   └── classical/cuda/   # CUDA kernels for Pauli expectations
├── include/              # C++ headers (stack_types.h)
├── tests/                # Test suite
├── local_test_run.py     # Simulator entry point
├── ibm_test_run.py       # IBM Quantum entry point
├── serial_baseline.py    # Qiskit reference baseline
├── run_analysis.py       # Results analysis + plotting
├── Dockerfile            # CUDA 12.2 + OpenMPI container
├── Makefile              # Build orchestration
└── CMakeLists.txt        # C++ build config
```
