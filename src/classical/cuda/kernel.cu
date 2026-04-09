// CUDA kernel for Pauli-term expectation value computation
// Uses mean-field approximation: ⟨P⟩ = Π_i f(op_i, θ_i)
// where f(Z,θ)=cos(θ), f(X,θ)=sin(θ), f(Y,θ)=sin(θ), f(I,θ)=1
//
// Mixed precision -  FP32 trigonometry for throughput, FP64 accumulation for accuracy
// One thread per Pauli term → tree reduction → total energy

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>


// Flattened Pauli term for GPU -  operator string encoded as char array
// Each term -  coeff (double) + op chars (padded to max_qubits)
// Layout in d_terms buffer -  [coeff_0][op_0_0..op_0_q] [coeff_1][op_1_0..op_1_q] ..


// KERNEL - one thread per Pauli term
// Each thread computes coeff * product of single-qubit expectations
// FP32 trig (cosf/sinf) → widen to FP64 for accumulation
__global__ void pauli_expectation_kernel(
    const double* __restrict__ coeffs,     // [n_terms] Pauli coefficients
    const char*   __restrict__ ops,        // [n_terms * max_qubits] operator chars ('I','X','Y','Z')
    const float*  __restrict__ params,     // [n_params] variational parameters θ
    double*       result,                  // [1] output energy accumulator
    int n_terms,
    int n_qubits,
    int n_params
) {
    __shared__ double sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    double term_val = 0.0;

    if (idx < (unsigned int)n_terms) {
        double expectation = 1.0;
        double stride = (n_qubits > 0) ? (double)n_params / (double)n_qubits : 1.0;

        // Compute product of single-qubit expectations (mean-field)
        for (int q = 0; q < n_qubits; q++) {
            char op = ops[idx * n_qubits + q];
            if (op == 'I') continue;

            int param_idx = (int)(q * stride);
            if (param_idx >= n_params) param_idx = n_params - 1;

            // FP32 trig for throughput, widen to FP64
            float theta = params[param_idx];
            switch (op) {
                case 'Z': expectation *= (double)cosf(theta); break;
                case 'X': expectation *= (double)sinf(theta); break;
                case 'Y': expectation *= (double)sinf(theta); break;
                default: break;
            }
        }

        term_val = coeffs[idx] * expectation;
    }

    // Tree reduction in shared memory (FP64)
    sdata[tid] = term_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 1st thread of each block writes to global accumulator
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}



// HOST ENTRY POINT - computes E = Σ_i coeff_i * ⟨P_i⟩ on GPU
// Receives Pauli data from C++ dispatcher
extern "C" double run_cuda_pauli_expectation(
    const double* h_coeffs,        // [n_terms] coefficients
    const char*   h_ops,           // [n_terms * n_qubits] operator chars
    const float*  h_params,        // [n_params] θ values (FP32)
    int n_terms,
    int n_qubits,
    int n_params
) {
    if (n_terms <= 0 || n_params <= 0) return 0.0;

    double *d_coeffs = nullptr, *d_result = nullptr;
    char   *d_ops = nullptr;
    float  *d_params = nullptr;
    double h_result = 0.0;
    cudaError_t err;


    // Allocate device memory
    err = cudaMalloc(&d_coeffs, n_terms * sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "[CUDA] Malloc coeffs: %s\n", cudaGetErrorString(err)); return 0.0; }

    err = cudaMalloc(&d_ops, n_terms * n_qubits * sizeof(char));
    if (err != cudaSuccess) { fprintf(stderr, "[CUDA] Malloc ops: %s\n", cudaGetErrorString(err)); cudaFree(d_coeffs); return 0.0; }

    err = cudaMalloc(&d_params, n_params * sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "[CUDA] Malloc params: %s\n", cudaGetErrorString(err)); cudaFree(d_coeffs); cudaFree(d_ops); return 0.0; }

    err = cudaMalloc(&d_result, sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "[CUDA] Malloc result: %s\n", cudaGetErrorString(err)); cudaFree(d_coeffs); cudaFree(d_ops); cudaFree(d_params); return 0.0; }


    // Copy data to device
    cudaMemset(d_result, 0, sizeof(double));
    cudaMemcpy(d_coeffs, h_coeffs, n_terms * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ops, h_ops, n_terms * n_qubits * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, h_params, n_params * sizeof(float), cudaMemcpyHostToDevice);


    // Launch - one thread per Pauli term
    int blockSize = 256;
    int gridSize = (n_terms + blockSize - 1) / blockSize;
    pauli_expectation_kernel<<<gridSize, blockSize>>>(
        d_coeffs, d_ops, d_params, d_result,
        n_terms, n_qubits, n_params
    );

    cudaDeviceSynchronize();
    cudaError_t errSync = cudaGetLastError();
    if (errSync != cudaSuccess) {
        fprintf(stderr, "[CUDA] Kernel error: %s\n", cudaGetErrorString(errSync));
    }

    // Copy result back (FP64)
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_coeffs);
    cudaFree(d_ops);
    cudaFree(d_params);
    cudaFree(d_result);

    return h_result;
}


// LEGACY - kept for backward compatibility with old dispatcher path
extern "C" double run_cuda_vqe_fp32(const float* /*h_params*/, int /*n*/) {
    return 0.0;
}
