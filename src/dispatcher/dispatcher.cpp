#include "stack_types.h"
#include <iostream>
#include <vector>
#include <mpi.h>

StackResult route_workload(HybridWorkload& wl) {
    StackResult res; 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        // wl.circuit_qasm.resize(qasm_size);
    }
    
    // std::vector<char> qasm_buffer(wl.circuit_qasm.begin(), wl.circuit_qasm.end());

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
        // add more later
    }

    // ensures data arrives before starting CUDA kernls
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    // DEBUG
    if (param_size > 0) {
        std::cout << "[Node " << rank << "] First Param: " << wl.parameters[0] << std::endl;
    } else {
        std::cout << "[Node " << rank << "] ERROR: param_size is 0" << std::endl;
    }


    // Dispatch logic 
    if (wl.num_qubits < 15) {         
        res.energy = -1.137;    //placeholder vqe energy
        // res.execution_time = 0.002;   //secs 
        res.variance = 0.001;      //placeholder for noise
        res.used_path = "Local Simulator"; 

    } else {   //logc for HPC

        // ACCELERATION LAYER: mixed precision implemented 
        // convert to FP32 for heavy GPU state-vector math
        std::vector<float> params_fp32(wl.parameters.begin(), wl.parameters.end());

        // Mixed Precision Strategy - cast result to FP64 for final energy sum
        double local_energy = run_cuda_vqe_fp32(wl.parameters.data(), param_size);   // calls CUDA
        std::cout << "[Node " << rank << "] Local GPU Energy: " << local_energy << std::endl;     // DEBUG
        double global_energy = 0.0;

        //sums results from all nodes back to node 0 - Non blocking reduction
        MPI_Request red_req;

        // global reduction - non-blocking 
        MPI_Iallreduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &red_req);

        // wait for reduction to finish 
        MPI_Wait(&red_req, MPI_STATUS_IGNORE);

        res.energy = global_energy;
        res.variance = 0.001;      //placeholder for future noise
        // res.success_msg = "Success";
        res.used_path = "MPI + CUDA Distribued"; 
    }
    
    res.success_msg = "Success";
    res.execution_time = MPI_Wtime() - start_time;

    return res; 
}