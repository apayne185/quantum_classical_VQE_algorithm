/*Uses pybind11 to expose C++ structures to Python - uses MPI foe API metadata*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // for std::vector conversion
#include "stack_types.h"
#include <mpi.h>
#include <cuda_runtime.h>     // for CUDA API calls

namespace py = pybind11;

int init_mpi(){
    int provided;
    // MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    return provided;
}  

void finalize_mpi(){    //for clean exits
    MPI_Finalize();
}

// determines which node/rank we are in - needed to determine master v workers
int get_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int get_size() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

void execute_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}


void set_cuda_device(int rank) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int device = rank % deviceCount;
        cudaSetDevice(device);
    }
}



// links Python call to C++ Dispatcher
StackResult execute(const HybridWorkload& wl) {
    try {
        if (wl.circuit_qasm.empty()) {
            throw std::runtime_error("Empty QASM string receieved");
        }
        return route_workload(const_cast<HybridWorkload&>(wl));

    } catch(const std::exception& e){ 
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}




// maps C++ member variables to Python attributes 
PYBIND11_MODULE(hpc_core, m) {
    m.def("init_mpi", &init_mpi, "Initialize MPI environment");
    m.def("finalize_mpi", &finalize_mpi, "Finalize MPI environment");
    m.def("get_rank", &get_rank);
    m.def("get_size", &get_size);
    m.def("set_cuda_device", &set_cuda_device, "Assigns GPU to MPI rank");

    py::class_<HybridWorkload>(m, "HybridWorkload")
        .def(py::init<>())                              // allows python to do wl=hpc_core.HybridWorklod()
        .def_readwrite("num_qubits", &HybridWorkload::num_qubits)
        .def_readwrite("parameters", &HybridWorkload::parameters)
        .def_readwrite("circuit_depth", &HybridWorkload::circuit_depth)
        .def_readwrite("requires_gpu", &HybridWorkload::requires_gpu)
        .def_readwrite("backend_target", &HybridWorkload::backend_target)
        .def_readwrite("circuit_qasm", &HybridWorkload::circuit_qasm);
        

    py::class_<StackResult>(m, "StackResult")
        .def_readwrite("energy", &StackResult::energy)
        .def_readwrite("execution_time", &StackResult::execution_time)
        .def_readwrite("variance", &StackResult::variance)
        .def_readwrite("success_msg", &StackResult::success_msg)
        .def_readwrite("used_path", &StackResult::used_path);  
    

    m.def("execute", &execute, "Main entry point for hybrid stack");
    m.def("route_workload", &route_workload, "Core dispatcher function");
    // m.def("get_rank", &get_rank, "Get MPI rank of current process");
    // m.def("get_size", &get_size, "Get total number of MPI processes"); 
}