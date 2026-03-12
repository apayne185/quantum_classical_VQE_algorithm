// currently placeholders, but Q state vector logic will be here in later phases
#include <cuda_runtime.h>
#include <vector>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>



__global__ void compute_vqe_energy_kernel(const float* __restrict__ params, double* result, int n) {
    __shared__ double sdata[256];

    unsigned int tid = threadIdx.x;
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;


    double val = 0.0;
    if (idx < (unsigned int)n) {
        // simulate expectation value contribution - mixed precision 
        // val = (double)params[idx];
        float theta = params[idx];
        val = (double)(-0.5f*cosf(theta));  
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
    if (n <= 0) {
        fprintf(stderr, "[CUDA] run_cuda_vqe_fp32: n=%d is invalid.\n", n);
        return 0.0;
    } // DEBUG LINE
    float *d_params = nullptr;
    double *d_result = nullptr;
    double h_result= 0.0;
    cudaError_t err;

    // if (cudaMalloc(&d_params, n * sizeof(float)) != cudaSuccess || cudaMalloc(&d_result, sizeof(double)) != cudaSuccess)  {     //DEBUG LINE
    //     printf("CUDA Malloc Failed!\n");
    //     return -888.0;
    // }

    err = cudaMalloc(&d_params, n * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error (Params Malloc): %s\n", cudaGetErrorString(err));
        return 0.0;
    }

    err = cudaMalloc(&d_result, sizeof(double));
    if (err != cudaSuccess) {
        printf("CUDA Error (Result Malloc): %s\n", cudaGetErrorString(err));
        cudaFree(d_params); // Clean up previous allocation
        return 0.0;
    }


    // init result on GPU to 0, copy params over
    cudaMemset(d_result, 0, sizeof(double));
    err = cudaMemcpy(d_params, h_params, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] H2D memcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_params);
        cudaFree(d_result);
        return 0.0;  
    }    

    //launch kernel     
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    // double total_sum = 0.0;
    // int iterations = 100000;
    compute_vqe_energy_kernel<<<gridSize, blockSize>>>(d_params, d_result, n);

    cudaDeviceSynchronize();
    cudaError_t errSync = cudaGetLastError();
    if (errSync != cudaSuccess) {
        fprintf(stderr, "[CUDA] Kernel error: %s\n", cudaGetErrorString(errSync));
        cudaFree(d_params);
        cudaFree(d_result);
        return 0.0;
    }


    err = cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] D2H memcpy failed: %s\n",cudaGetErrorString(err));
    }
    cudaFree(d_params);
    cudaFree(d_result);

    return h_result; 
}