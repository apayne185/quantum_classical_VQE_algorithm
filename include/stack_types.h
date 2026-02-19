/* Defines "Contract", allows Python UI/C++ engine to function together. Contains:
 HybridWorkload - input packet for BE, 
 StackResult - output packet for future performance analysis */

#ifndef STACK_TYPES_H
#define STACK_TYPES_H

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
    int job_id;                   // for async tracking
};

// both C++ disptacher and python bridge needed to agree on workload


struct StackResult {
    double energy;       // vqe eigenvalue
    double execution_time;             //for benchmarking
    double variance;                   //for noise analysis
    std::string success_msg;          // errors or status updates 
    std::string used_path;             // cpu gpu or simulator
}; 

StackResult route_workload(HybridWorkload& wl);

//connects CUDA function to C++
// extern "C" double run_cuda_vqe(const double* h_params, int n);
extern "C" double run_cuda_vqe_fp32(const float* h_params, int n);

#endif
