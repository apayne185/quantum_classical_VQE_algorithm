#include "stack_types.h"
#include <iostream>

double route_workload(HybridWorkload& wl) {
    if (wl.num_qubits < 15 && !wl.requires_gpu) {
        std::cout << "[Dispatcher] Routing to local simulator..." << std::endl;
        return 0.0; 
    } else {
        std::cout << "[Dispatcher] Routing to HPC Stub (No CUDA detected)..." << std::endl;
        return 1.0; 
    }
}