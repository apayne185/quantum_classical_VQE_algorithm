#include "stack_types.h"
#include <iostream>

StackResult route_workload(HybridWorkload& wl) {
    StackResult res; 
    if (wl.num_qubits < 15) {         // logic for simulator 
    // if (wl.num_qubits < 15 && !wl.requires_gpu) {
        // std::cout << "[Dispatcher] Routing to local simulator..." << std::endl;
        // return 0.0; 
        res.energy = -1.137;    //placeholder vqe energy
        res.execution_time = 0.002;   //secs 
        res.success_msg = "Success";
        res.used_path = "Local Simulator"; 
    } else {   //logc for HPC
        // std::cout << "[Dispatcher] Routing to HPC Stub (No CUDA detected)..." << std::endl;
        // return 1.0; 
        res.energy = -1.137;    //placeholder vqe energy
        res.execution_time = 0.450;   // HPC latency included - placeholder
        res.success_msg = "Success";
        res.used_path = "HPC Cluster - GPU"; 
    }

    return res; 
}