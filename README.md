# Hybrid Quantum-Classical VQE Stack for HPC

**Bachelor's of Computer Science and Artificial Intelligence (BCSAI) Thesis — IE University**
**Anna Payne**

A hybrid quantum-classical middleware stack that implements the Variational Quantum Eigensolver (VQE) with MPI parallelism, CUDA GPU acceleration, and IBM Quantum cloud integration. This stack is designed for molecular ground-state energy computation with distributed Pauli-term evaluation across HPC resources.

---

## Quick Start

```bash
git clone <repo-url> && cd quantum_classical_VQE_algorithm
cp .env.example .env              # Add IBM Quantum credentials (optional, for QPU runs only)
make build                        # Build Docker image (CUDA 12.2 + OpenMPI + Python 3.11)
make trial                        # 7-layer diagnostic test (simulator, 2 MPI ranks)
make example NP=2                 # Run template (H2 ground state, 2 MPI ranks)
make run NP=2                     # Full chemistry benchmark (H2, LiH, BeH2, H2O)
```

See [`template.py`](template.py) for a step-by-step walkthrough. See [`MOLECULES.md`](MOLECULES.md) for all available built-in molecules.

## Architecture

### Three-Layer Stack

```
Python API (src/api/)
   ↓ QuantumProblem.prepare() → Pauli Hamiltonian + parameterized ansatz
C++ Dispatcher (src/dispatcher/)
   ↓ MPI broadcast → local compute → MPI_Allreduce
CUDA Kernels (src/classical/cuda/)
   ↓ Mixed-precision (FP32 trig → FP64 accumulation) Pauli expectations
```

### Evaluation Paths

The stack supports three evaluation backends, selected automatically:

| Path | When Used | Description |
|------|-----------|-------------|
| **Statevector (MPI)** | `BACKEND=simulator` | Exact statevector simulation distributed across MPI ranks. Each rank builds the full statevector and evaluates its partition of Pauli terms. Supports GPU acceleration via cuStateVec when available, with automatic CPU fallback. |
| **IBM QPU** | `BACKEND=ibm_cloud` | Submits circuits to IBM Quantum via EstimatorV2. Concurrent classical statevector computation overlaps with QPU round-trip time. |
| **C++ Dispatcher** | Fallback / Layer 3 test | MPI-coordinated dispatch through the C++ bridge with CUDA kernel or CPU mean-field approximation. |

### Data Flow

`ChemistryProblem.from_registry("LiH")` → PySCF driver → Jordan-Wigner mapping → Pauli Hamiltonian (631 terms) + HWE-adaptive ansatz (96 params) → `HPCHybridStack.vqe_optimize()` → SPSA loop distributes θ± across P ranks → each rank evaluates its Pauli partition → `MPI_Allreduce` sums energies → gradient update → repeat until convergence.

### Key Classes

| Class | File | Role |
|-------|------|------|
| `HPCHybridStack` | `src/api/interface.py` | Main entry point: MPI init, SPSA optimizer, checkpoint management, GPU/QPU routing |
| `ChemistryProblem` | `src/api/problems.py` | Molecular Hamiltonian via PySCF + Jordan-Wigner, auto-selects ansatz tier |
| `FinanceProblem` | `src/api/problems.py` | Portfolio optimization via QUBO → Ising (extensibility demo) |
| `MoleculeResolver` | `src/api/molecule_resolver.py` | Registry → raw geometry → SMILES → PubChem cascade |
| `HybridWorkload` | `include/stack_types.h` | C++ dispatcher interface contract |

### SPSA Optimizer Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Perturbation $c$ | 0.1 | Fixed; appropriate for angles in $[0, 2\pi]$ |
| Step size $a$ | $0.628 / \sqrt{p/8}$ | Scales with parameter count $p$ |
| Stability constant $A$ | $0.1 \times$ max_iterations | Delays aggressive early steps |
| Decay rates $\alpha, \gamma$ | 0.602, 0.101 | Standard SPSA (Spall 1998) |
| Convergence | Sliding window of 10 | Spread < 1.6 mHa (chemical accuracy) |
| Initialization | $\mathcal{U}(-0.1, 0.1)$ | Near-zero to stay in physical sector |
| Random seed | 42 | Fixed for reproducibility |
| Max iterations | $\max(200,\; 8 \times \text{num\_params})$ | Scales with problem size |


## Makefile Targets

| Target | Description |
|--------|-------------|
| `make build` | Build Docker image |
| `make trial` | 7-layer diagnostic + stress tests (simulator, 2 ranks) |
| `make run NP=4` | Full chemistry benchmark with MPI (simulator) |
| `make run-ibm NP=2` | Run on IBM Quantum QPU (requires `.env` credentials) |
| `make scaling` | Strong scaling sweep (P=1,2,4,8) |
| `make weak-scaling` | Weak scaling sweep (problem size grows with P) |
| `make baseline` | Serial Qiskit VQE reference (no MPI, no GPU) |
| `make test` | Run all tests (molecule resolver + layer diagnostics) |
| `make shell` | Interactive shell inside container |
| `make clean` | Remove image and build artifacts |


## Testing & Stress Tests

`make trial` runs 7 validation layers:

| Layer | What It Tests |
|-------|---------------|
| 1. MPI Bridge | Rank initialization, `MPI_Barrier` synchronization |
| 2. Problem Preparation | H₂ Hamiltonian construction, Pauli decomposition, ansatz building |
| 3. C++ Dispatcher | Single dispatch through pybind11 bridge, MPI broadcast + reduce |
| 4. VQE Loop | 10-iteration SPSA optimization, convergence detection |
| 5. Checkpoint Resilience | Save θ at iter 5, restart from checkpoint, verify continuity |
| 6. Latency Spiking | Random 0.5–2.0s delays injected per rank, verify MPI stays synchronized |
| 7. Drop-Out Recovery | Delete checkpoint at iter 10, recover from iter 5, verify no data loss |


## IBM Quantum Setup

1. Get an API token at [quantum.ibm.com](https://quantum.ibm.com)
2. Copy `.env.example` to `.env` and fill in credentials:
   ```
   IBM_QUANTUM_TOKEN=your_token_here
   IBM_QUANTUM_INSTANCE=your-crn-instance
   IBM_QUANTUM_BACKEND=ibm_marrakesh
   IBM_QUANTUM_REGION=us-east
   ```
3. Run: `make run-ibm NP=2`

Uses EstimatorV2 with `mode=backend` (compatible with open/free plan — no Sessions), 4096 shots, and T-REx measurement error mitigation (resilience level 1).


## Supported Molecules

| Molecule | Qubits | Pauli Terms | FCI Energy (Ha) | Ansatz | Notes |
|----------|--------|-------------|-----------------|--------|-------|
| H₂ | 4 | 15 | -1.1373 | HWE-adaptive | Fastest; good for QPU testing |
| LiH | 12 | 631 | -7.8825 | HWE-adaptive | Frozen 2 core electrons |
| BeH₂ | 14 | 666 | -15.5951 | HWE-adaptive | Frozen 2 core electrons |
| H₂O | 14 | 1,086 | -75.0124 | HWE-adaptive | Frozen 2 core electrons |

Custom molecules via `MoleculeResolver`: registry names, raw geometry strings, SMILES notation, or PubChem lookup.


## Results

All runs automatically save structured output to organized subdirectories:

```
results/
├── simulator/      # make run — JSON + full iteration logs
├── ibm/            # make run-ibm — JSON + QPU job logs
├── baseline/       # make baseline — JSON + logs
├── scaling/        # make scaling / make weak-scaling — summary + logs
└── trial/          # make trial — diagnostic test logs
```

Each file is timestamped (e.g., `simulator_20260319_212106.json`) and includes the git commit hash, per-molecule energies, convergence histories, and timing data. JSON results are never overwritten.

Analyze results with:
```bash
python benchmarks/run_analysis.py          # summary table
python benchmarks/run_analysis.py --plot   # convergence plots (requires matplotlib)
```


## Dependencies

**Core:** qiskit ≥1.0, qiskit-nature, qiskit-ibm-runtime ≥0.45, pyscf, mpi4py, numpy, scipy
**Build:** CMake 3.18+, pybind11, nlohmann_json, libcurl, OpenMPI, CUDA 12.2
**Optional:** cupy (GPU), rdkit (SMILES), matplotlib (plots)

All dependencies are included in the Docker image — no local installation required. See `requirements.txt` for Python packages.


## Project Structure

```
├── src/
│   ├── api/                 # Python API layer
│   │   ├── interface.py     # HPCHybridStack — MPI init, SPSA, checkpoints, GPU/QPU routing
│   │   ├── problems.py      # QuantumProblem, ChemistryProblem, FinanceProblem, ansatz selection
│   │   ├── molecule_resolver.py  # Registry → geometry → SMILES → PubChem cascade
│   │   ├── results.py       # Structured JSON persistence
│   │   └── log.py           # Dual-output logger (console + file via stdout tee)
│   ├── dispatcher/          # C++ MPI coordinator
│   │   ├── dispatcher.cpp   # MPI broadcast, GPU/CPU routing, Allreduce
│   │   ├── bridge.cpp       # pybind11 Python ↔ C++ bridge
│   │   └── qpu_client.cpp   # IBM Quantum REST client (IAM auth, job polling)
│   └── classical/cuda/
│       └── kernel.cu         # CUDA Pauli expectation kernel (FP32 trig → FP64 reduction)
├── include/
│   └── stack_types.h         # HybridWorkload / StackResult / PauliTerm structs
├── tests/
│   ├── test_layers_run.py    # 7-layer diagnostic + stress tests
│   └── test_molecules_run.py # Molecule resolver validation
├── benchmarks/
│   ├── local_test_run.py     # Simulator benchmark (H2, LiH, BeH2, H2O)
│   ├── ibm_test_run.py       # IBM Quantum QPU benchmark
│   ├── serial_baseline.py    # Serial Qiskit VQE reference
│   └── run_analysis.py       # Results analysis + plotting
├── results/                  # Auto-organized: simulator/, ibm/, baseline/, scaling/, trial/
├── checkpoints/              # Rolling SPSA checkpoints (per-molecule subdirs)
├── Dockerfile                # CUDA 12.2 + OpenMPI + Python 3.11 container
├── Makefile                  # Build and run orchestration
├── CMakeLists.txt            # C++ build configuration
├── .env.example              # IBM Quantum credential template
└── requirements.txt          # Python dependencies
```


## Known Limitations

- **HWE particle-number violation**: The Hardware-Efficient Ansatz does not conserve electron number. After sufficient SPSA iterations, the optimizer can find states below FCI. Mitigated via near-zero initialization and best-physical-energy tracking.
- **Single-host scaling**: Current benchmarks run on one machine with shared CPU cores. MPI scaling degrades at P≥4 due to resource contention. Multi-node InfiniBand deployment would improve this.
- **GPU acceleration pending**: The cuStateVec integration path is architecturally complete (automatic detection + fallback) but awaits GPU cluster access for benchmarking.