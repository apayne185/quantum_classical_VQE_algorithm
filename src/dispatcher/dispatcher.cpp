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
    
    int param_size = wl.parameters.size();
    MPI_Bcast(&param_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        wl.parameters.resize(param_size);
    }

    // Shares numerical parameters - theta
    MPI_Bcast(wl.parameters.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank ==0){
        std::cout << "[MASTER DISPATCHER] Distributing "<< wl.num_qubits << " qubits across " << size << " nodes." <<std::endl;
    }

    // Dispatch logic 
    if (wl.num_qubits < 15) {         
        res.energy = -1.137;    //placeholder vqe energy
        res.execution_time = 0.002;   //secs 
        res.variance = 0.001;      //placeholder for noise
        res.success_msg = "Success";
        res.used_path = "Local Simulator"; 

    } else {   //logc for HPC
        double local_energy = -1.137 / size;     //palceholder for now, splits across nodes
        double global_energy = 0.0;

        //sums results from all nodes back to node 0
        MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        res.energy = global_energy;
        res.execution_time = 0.450;   // HPC latency included - placeholder
        res.variance = 0.001;      //placeholder for noise
        res.success_msg = "Success";
        res.used_path = "MPI + CUDA Distribued"; 
    }

    return res; 
}