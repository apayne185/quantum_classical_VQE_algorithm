#pragma once
#include <vector>
#include <string>

struct HybridWorkload {
    int num_qubits;
    std::vector<double> parameters;      // theta value
    int circuit_depth;
    bool requires_gpu;
    std::string backend_target;       // "simulator", "hpc_cluster", "qpu"
};

// both C++ disptacher and python bridge needed to agree on workload


// double run_local_sim(HybridWorkload& wl);
// double run_hpc_cluster(HybridWorkload& wl);

double route_workload(HybridWorkload& wl);