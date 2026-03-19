/* Defines "Contract", allows Python UI/C++ engine to function together. Contains:
 HybridWorkload - input packet for BE, 
 StackResult - output packet for future performance analysis */

#ifndef STACK_TYPES_H
#define STACK_TYPES_H

#pragma once
#include <vector>
#include <string>

// Pauli term - single operation str and coefficient 
struct PauliTerm {
    std::string op;       // ex IIZI
    double coeff;           // -0.81
};

// input packet from Python to C++ dispatcher
struct HybridWorkload {
    int num_qubits = 0;
    std::vector<double> parameters;      // variational θ vector
    int circuit_depth =0;
    bool requires_gpu = true;
    std::string backend_target;       // "simulator" or "ibm_cloud"
    std::string circuit_qasm;
    std::string job_id = "";                    // for async tracking
    std::vector<PauliTerm> pauli_terms;     // hamiltonian decomposition
    int num_shots = 1024;                   // QPU measuremnt shots
};


// output packet retuned to Python
struct StackResult {
    double energy = 0.0;                      // VQE eigenvalue estimation 
    double e_minus= 0.0;                    // SPSA E(θ − ck * Δ)
    double execution_time = 0.0;             // Wall clock time for benchmarking
    double variance = 0.0;                   //for shot noise var of expectation value
    double masking_metric = 0.0;             // M = T_accel / T_comm - determines performanc metric
    std::string success_msg;          // errors or status updates 
    std::string used_path;             // MPI+CUDA, MPI+CPU  Fallback, ..
}; 



StackResult route_workload(HybridWorkload& wl);

// CUDA kernel entry points
extern "C" double run_cuda_vqe_fp32(const float* h_params, int n);  // Legacy fallback
extern "C" double run_cuda_pauli_expectation(
    const double* h_coeffs, const char* h_ops, const float* h_params,
    int n_terms, int n_qubits, int n_params
);

#endif
