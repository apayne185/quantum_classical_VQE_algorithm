// CPU-only stub — compiled when CUDA is not available.
// These functions match the signatures in kernel.cu so the linker is satisfied,
// but the dispatcher never calls them (it checks use_cuda first).
// If somehow reached, they throw a clear error instead of silently returning 0.

#include <stdexcept>

extern "C" double run_cuda_vqe_fp32(const float* /*h_params*/, int /*n*/) {
    throw std::runtime_error(
        "[VQE] This build was compiled without CUDA. "
        "GPU acceleration is not available. Set use_gpu=False or install the CUDA toolkit.");
}

extern "C" double run_cuda_pauli_expectation(
    const double* /*h_coeffs*/, const char* /*h_ops*/,
    const float* /*h_params*/, int /*n_terms*/, int /*n_qubits*/, int /*n_params*/)
{
    throw std::runtime_error(
        "[VQE] This build was compiled without CUDA. "
        "GPU acceleration is not available. Set use_gpu=False or install the CUDA toolkit.");
}

double compute_expectation_cuda(
    const double* /*theta*/, const void* /*pauli_terms*/)
{
    throw std::runtime_error(
        "[VQE] This build was compiled without CUDA. "
        "GPU acceleration is not available. Set use_gpu=False or install the CUDA toolkit.");
}