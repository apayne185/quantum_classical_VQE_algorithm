/*Routes tasks, if too small tasks stay on CPU (avoids cost of moving to GPU), but otherwise
routes to the HPC path - calculation logic*/

#include "stack_types.h"
#include <iostream>
#include <mpi.h>

StackResult route_workload(HybridWorkload& wl) {
    StackResult res; 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "[DEBUG Master] Qubits: " << wl.num_qubits 
                  << " Params count: " << wl.parameters.size() << std::endl;
    }
    
    //share param, qasm string length w all nodes
    int param_size; 
    if (rank == 0){
        param_size = wl.parameters.size();
    }
    MPI_Bcast(&param_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int qasm_size = wl.circuit_qasm.size();   
    MPI_Bcast(&qasm_size, 1, MPI_INT, 0, MPI_COMM_WORLD);


    //resizes strings on worker nodes
    if (rank != 0) {
        wl.parameters.resize(param_size);
        wl.circuit_qasm.resize(qasm_size);
    }

    // Shares numerical parameters - theta
    MPI_Bcast(wl.parameters.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(const_cast<char*>(wl.circuit_qasm.data()), qasm_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank ==0){
        std::cout << "[MASTER DISPATCHER] Distributing "<< wl.num_qubits << " qubits across " << size << " nodes." <<std::endl;
    }

    if (param_size > 0) {
        std::cout << "[Node " << rank << "] First Param: " << wl.parameters[0] << std::endl;
    } else {
        std::cout << "[Node " << rank << "] ERROR: param_size is 0!" << std::endl;
    }


    // Dispatch logic 
    if (wl.num_qubits < 15) {         
        res.energy = -1.137;    //placeholder vqe energy
        // res.execution_time = 0.002;   //secs 
        res.variance = 0.001;      //placeholder for noise
        res.success_msg = "Success";
        res.used_path = "Local Simulator"; 

    } else {   //logc for HPC
        double local_energy = run_cuda_vqe(wl.parameters.data(), param_size);   // calls CUDA
        std::cout << "[Node " << rank << "] Local GPU Energy: " << local_energy << std::endl;
        double global_energy = 0.0;

        //sums results from all nodes back to node 0
        MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        res.energy = global_energy;
        // res.execution_time = 0.450;   // HPC latency included - placeholder
        res.variance = 0.001;      //placeholder for future noise
        res.success_msg = "Success";
        res.used_path = "MPI + CUDA Distribued"; 
    }
    
    double end_time = MPI_Wtime();
    res.execution_time = end_time - start_time;

    return res; 
}