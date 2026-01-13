// currently placeholders, but Q state vector logic will be here in later phases
#include <cuda_runtime.h>
#include <vector>


__global__ void compute_vqe_energy_kernel(double* params, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // params[idx] = params[idx] * params[idx]; 
        // double contribution = params[idx] * params[idx] - 0.5;
        // double contribution = params[idx] + 10; 
        double contribution = 1.0;
        atomicAdd(result, contribution);
    }
}

extern "C" double run_cuda_vqe(const double* h_params, int n) {
    if (n <= 0) return -999.0;    // DEBUG LINE
    double *d_params, *d_result;
    double h_result= 0.0;

    cudaError_t err1 = cudaMalloc(&d_params, n * sizeof(double));     // DEBUG LINE
    cudaError_t err2 = cudaMalloc(&d_result, sizeof(double));
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {     //DEBUG LINE
        printf("CUDA Malloc Failed!\n");
        return -888.0;
    }

    // allocate GPU memory
    cudaMalloc(&d_params, n * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    // init result on GPU to 0, copy params over
    cudaMemset(d_result, 0, sizeof(double));
    cudaMemcpy(d_params, h_params, n * sizeof(double), cudaMemcpyHostToDevice);
    
    //launch kernel     
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    compute_vqe_energy_kernel<<<gridSize, blockSize>>>(d_params, d_result, n);
    // compute_vqe_energy_kernel<<<(n+255)/256, 256>>>(d_data, n);
    
    // Catch kernel launch errors
    cudaError_t errSync = cudaGetLastError();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }

    // final calculation sends back to CPU
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_params);
    cudaFree(d_result);

    return h_result; 
}