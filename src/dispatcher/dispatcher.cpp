#include "stack_types.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>  
#include <numeric>
#include <future>   
#include <thread>   



double call_qpu_cloud(std::string qasm) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return -1.85;     // Mock QPU expectation value
}


StackResult route_workload(HybridWorkload& wl) {
    StackResult res; 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double qpu_val = 0.0;
    std::future<double> qpu_future;

    // metadata syncronization - non blocking
    int param_size = (rank == 0) ? wl.parameters.size() : 0;
    int qasm_size = (rank == 0) ? wl.circuit_qasm.size() : 0;
    
    // broadcast sizes first so workers can allocate the memory
    MPI_Bcast(&param_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qasm_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //resizes strings on worker nodes
    if (rank != 0) {
        wl.parameters.resize(param_size);}
    // explicit serialization buffer to bypass const_cast issues
    std::vector<char> qasm_buffer(qasm_size); 
    if (rank == 0) {
        std::copy(wl.circuit_qasm.begin(), wl.circuit_qasm.end(), qasm_buffer.begin());}


    double start_time = MPI_Wtime();
    MPI_Request requests[2];          // tracks async broadcasts

    // Shares numerical parameters - theta
    // starts async broadcast 
    MPI_Ibcast(wl.parameters.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Ibcast(qasm_buffer.data(), qasm_size, MPI_CHAR, 0, MPI_COMM_WORLD, &requests[1]);

    if (rank ==0){
        // rank 0 starts QPU RTT 
        std::cout << "[MASTER] Rank 0 dispatching QASM to Cloud QPU.." << std::endl;
        qpu_future = std::async(std::launch::async, call_qpu_cloud, wl.circuit_qasm);
    }

    // ensures data arrives before starting CUDA kernls
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);


    // ACCELERATION LAYER: mixed precision implemented 
    // Mixed Precision Strategy - cast result to FP64 for final energy sum
    double t_accel_start = MPI_Wtime();
    // double local_energy = 0.0;
    double e_plus_local = 0.0;
    double e_minus_local = 0.0;
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    bool is_batch = (wl.parameters.size() == (size_t)(wl.num_qubits * 2));

    if (wl.requires_gpu && deviceCount > 0) {
        if (is_batch) {
            // Path A: NVIDIA GPU available (CUDA execution)
            // convert to FP32 for heavy GPU state-vector math
            std::vector<float> p1(wl.parameters.begin(), wl.parameters.begin() + wl.num_qubits);
            e_plus_local = run_cuda_vqe_fp32(p1.data(), wl.num_qubits);

            std::vector<float> p2(wl.parameters.begin() + wl.num_qubits, wl.parameters.end());
            e_minus_local = run_cuda_vqe_fp32(p2.data(), wl.num_qubits);
            res.used_path = "MPI + CUDA Distributed";

        } else {
            // --- SINGLE MODE ---
            std::vector<float> params_fp32(wl.parameters.begin(), wl.parameters.end());
            e_plus_local = run_cuda_vqe_fp32(params_fp32.data(), wl.num_qubits);
            res.used_path = "MPI + CUDA Single Mode";
        }

    } else {
        // Path B: FALLBACK (No GPU found)
        // Simulate CUDA kernel logic on CPU
         for (int i = 0; i < param_size; ++i) {
            if (is_batch && i >= wl.num_qubits) e_minus_local += wl.parameters[i];
            else e_plus_local += wl.parameters[i];
        }
        res.used_path = (wl.num_qubits < 15) ? "Local Simulator" : "MPI + CPU Fallback";
        }

        double t_accel_end = MPI_Wtime();

        if (rank == 0) {
            qpu_val = qpu_future.get(); 
            std::cout << "[MANAGER] QPU result received. Merging with GPU results..." << std::endl;
        }

        double e_plus_global = 0.0;
        double e_minus_global = 0.0;
        // global reduction - non-blocking 
        MPI_Allreduce(&e_plus_local, &e_plus_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (is_batch) {
            MPI_Allreduce(&e_minus_local, &e_minus_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        std::cout << "[Rank " << rank << "] Local: " << e_plus_local 
              << " | Global: " << e_plus_global << std::endl;

        res.energy = e_plus_global + qpu_val;
        res.success_msg = "Success";
        res.execution_time = MPI_Wtime() - start_time;

        double t_comm = res.execution_time - (t_accel_end - t_accel_start);
        res.variance = is_batch ? (e_minus_global + qpu_val) : 0.0;
        res.masking_metric = (t_comm > 0) ? (t_accel_end - t_accel_start) / t_comm : 0.0;  

        return res; 
    }
    
