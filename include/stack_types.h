#pragma once
#include <vector>
#include <string>

struct HybridWorkload {
    int num_qubits;
    std::vector<double> parameters;      // theta value
    int circuit_depth;
    bool requires_gpu;
    std::string backend_target;       // "simulator", "hpc_cluster", "qpu"
    std::string circuit_qasm;
};

// both C++ disptacher and python bridge needed to agree on workload


struct StackResult {
    double energy;       // vqe eigenvalue
    double execution_time;             //for benchmarking
    std::string success_msg;          // errors or status updates 
    std::string used_path;             // cpu gpu or simulator
}; 

StackResult route_workload(HybridWorkload& wl);
