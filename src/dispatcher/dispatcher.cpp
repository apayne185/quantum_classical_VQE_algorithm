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

    double start_time = MPI_Wtime();
    MPI_Request requests[2];          // tracks async broadcasts

    // metadata syncronization - non blocking
    int param_size = (rank == 0) ? wl.parameters.size() : 0;
    int qasm_size = (rank == 0) ? wl.circuit_qasm.size() : 0;
    // broadcast sizes first so workers can allocate the memory
    MPI_Bcast(&param_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qasm_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //resizes strings on worker nodes
    if (rank != 0) {
        wl.parameters.resize(param_size);
    }

    // explicit serialization buffer to bypass const_cast issues
    std::vector<char> qasm_buffer(qasm_size); 
    if (rank == 0) {
        std::copy(wl.circuit_qasm.begin(), wl.circuit_qasm.end(), qasm_buffer.begin());
    }

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
    double local_energy = 0.0;
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);


    if (wl.requires_gpu && deviceCount > 0) {
        // Path A: NVIDIA GPU available (CUDA execution)
        // convert to FP32 for heavy GPU state-vector math
        std::vector<float> params_fp32(wl.parameters.begin(), wl.parameters.end());
        local_energy = run_cuda_vqe_fp32(params_fp32.data(), param_size);
        res.used_path = "MPI + CUDA Distributed";
    } else {
        // Path B: FALLBACK (No GPU found)
        // Simulate CUDA kernel logic on CPU
         for (double p : wl.parameters) {
            local_energy += p; 
        }
        res.used_path = (wl.num_qubits < 15) ? "Local Simulator" : "MPI + CPU Fallback";
        }

        double t_accel_end = MPI_Wtime();
        if (rank == 0) {
            qpu_val = qpu_future.get(); 
            std::cout << "[MANAGER] QPU result received. Merging with GPU results..." << std::endl;
        }

        double global_classical_energy = 0.0;
        // global reduction - non-blocking 
        MPI_Allreduce(&local_energy, &global_classical_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        std::cout << "[Rank " << rank << "] Local: " << local_energy 
              << " | Global: " << global_classical_energy << std::endl;

        res.energy = global_classical_energy + qpu_val;
        res.success_msg = "Success";
        res.execution_time = MPI_Wtime() - start_time;

        double t_comm = res.execution_time - (t_accel_end - t_accel_start);
        res.variance = (t_accel_end - t_accel_start) / t_comm;   // 0.001;      //placeholder for future noise


        return res; 
    }
    
