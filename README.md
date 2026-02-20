# Hybrid Quantum–Classical Software Stack for HPC Algorithm Acceleration
#### Thesis Research for Bachelors of CSAI at IE University
### By Anna Payne

## Phase 1: Hybrid Communitcation Contract
This software stack implements a Hybrid Quantum- Classical Dispatcher. 

### Architecture Features
1. Data Serialization: Quantum circuits get passed as OpenQASM 3.0 strings, ensuring the BE is hardware agnostic.
2. Dispatcher Heuristics: Workloads are dynamically routed, Circuits with >15 qubits are dispatched to CUDA accelerated kernels to maximize throughput.
3. Performance Metrics: The stack returns a struct that contains the VQE Energy, the Variance (for future noise analysis) and Wall Time (for HPC benchmarking).
4. Stability: A Pybind11 bridge with integrated error-handling prevents any memory faults from interrupting/crashing the classical optimization loop




## To Run the Software
### Activate the Conda Env
### To Compile the C++ Code
mkdir build 

cd build 

cmake ..

cmake --build . --config Release


### To Run the Python Script - with 4 nodes/processes

mpiexec -n 4 python test_run.py