// currently placeholders, but Q state vector logic will be here in later phases
#include <cuda_runtime.h>
#include <vector>
#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void compute_vqe_energy_kernel(const float* params, double* result, int n) {
    __shared__ double sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (idx < n) {
        // simulate expectation value contribution - mixed precision 
        val = (double)params[idx] * 0.5;
    }
    sdata[tid] = val;
    __syncthreads();

    // inplace tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // onyl first thread of each block writes to global mem
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

extern "C" double run_cuda_vqe_fp32(const float* h_params, int n) {
    if (n <= 0) return -999.0;    // DEBUG LINE
    float *d_params = nullptr;
    double *d_result = nullptr;
    double h_result= 0.0;

    // cudaError_t err1 = cudaMalloc(&d_params, n * sizeof(float));     // DEBUG LINE
    // cudaError_t err2 = cudaMalloc(&d_result, sizeof(double));
    if (cudaMalloc(&d_params, n * sizeof(float)) != cudaSuccess || cudaMalloc(&d_result, sizeof(double)) != cudaSuccess)  {     //DEBUG LINE
        printf("CUDA Malloc Failed!\n");
        return -888.0;
    }


    // init result on GPU to 0, copy params over
    cudaMemset(d_result, 0, sizeof(double));
    cudaMemcpy(d_params, h_params, n * sizeof(float), cudaMemcpyHostToDevice);
    
    //launch kernel     
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    compute_vqe_energy_kernel<<<gridSize, blockSize>>>(d_params, d_result, n);
    // compute_vqe_energy_kernel<<<(n+255)/256, 256>>>(d_data, n);
    
    // Catch kernel launch errors
    cudaDeviceSynchronize();
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