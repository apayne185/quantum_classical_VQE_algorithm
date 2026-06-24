/*Uses pybind11 to expose C++ structures to Python - uses MPI for API metadata*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "stack_types.h"
#include <mpi.h>
#include <stdexcept>
#include <string>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

int init_mpi(){
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE){
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            fprintf(stderr, "[Bridge] WARNING: MPI_THREAD_MULTIPLE requested but only level %d provided. Async QPU dispatch is unsafe", provided);
        }
    }
    return provided;
}  

void finalize_mpi(){          //for clean exit
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

// Determines current rank - needed to determine master v workers
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


bool cuda_build() {
#ifdef HAVE_CUDA
    return true;
#else
    return false;
#endif
}

void set_cuda_device(int rank) {
#ifdef HAVE_CUDA
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        throw std::runtime_error(
            "No CUDA devices found on this node. "
            "Check that your NVIDIA driver is installed and nvidia-smi works.");
    }
    int device = rank % deviceCount;
    cudaSetDevice(device);
#else
    (void)rank;
    throw std::runtime_error(
        "[VQE] This build was compiled without CUDA support. "
        "GPU acceleration is unavailable. Rebuild with CUDA toolkit installed, "
        "or run with use_gpu=False.");
#endif
}



// links Python call to C++ Dispatcher
StackResult execute(HybridWorkload& wl) {
    int rank = get_rank();

    if (wl.circuit_qasm.empty()) {
        throw std::runtime_error("Empty QASM string received on Rank " + std::to_string(rank));
    }

    try{
        return route_workload(wl);
    } catch(const std::exception& e){
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}




// maps C++ member variables to Python attributes 
PYBIND11_MODULE(hpc_core, m) {
    m.def("init_mpi", &init_mpi, "Initialize MPI environment w MPI_THREAD_MULTIPLE");
    m.def("finalize_mpi", &finalize_mpi, "Finalize MPI environment");
    m.def("get_rank", &get_rank, "Return MPI rank of calling process");
    m.def("get_size", &get_size, "Return number of MPI processes");
    m.def("set_cuda_device", &set_cuda_device, "Assigns CUDA device to MPI rank (round robin). Raises if built without CUDA.");
    m.def("execute_barrier", &execute_barrier, "MPI barrier — synchronizes all ranks");
    m.def("cuda_build", &cuda_build, "Returns True if this module was compiled with CUDA support");
    m.def("execute", &execute, py::arg("workload"), "Dispatch HybridWorkload through stack");
    m.def("route_workload", [](HybridWorkload& wl) { return route_workload(wl); }, py::arg("workload"), "Low level dispatcher");


    py::class_<PauliTerm>(m, "PauliTerm") 
        .def(py::init<>())
        .def(py::init<std::string, double>(), py::arg("op"), py::arg("coeff"))
        .def_readwrite("op",    &PauliTerm::op)
        .def_readwrite("coeff", &PauliTerm::coeff)
        .def("__repr__", [](const PauliTerm& pt) 
            {return "<PauliTerm op=" + pt.op +" coeff=" + std::to_string(pt.coeff) + ">"; }); 



    py::class_<HybridWorkload>(m, "HybridWorkload")
        .def(py::init<>())                              // allows python to do wl=hpc_core.HybridWorklod()
        .def_readwrite("num_qubits", &HybridWorkload::num_qubits)
        .def_readwrite("parameters", &HybridWorkload::parameters)
        .def_readwrite("circuit_depth", &HybridWorkload::circuit_depth)
        .def_readwrite("requires_gpu", &HybridWorkload::requires_gpu)
        .def_readwrite("backend_target", &HybridWorkload::backend_target)
        .def_readwrite("circuit_qasm", &HybridWorkload::circuit_qasm)
        .def_readwrite("job_id", &HybridWorkload::job_id)
        .def_readwrite("pauli_terms", &HybridWorkload::pauli_terms)
        .def_readwrite("num_shots", &HybridWorkload::num_shots);
        

    py::class_<StackResult>(m, "StackResult")
        .def_readwrite("energy", &StackResult::energy)
        .def_readwrite("e_minus",&StackResult::e_minus)
        .def_readwrite("execution_time", &StackResult::execution_time)
        .def_readwrite("variance", &StackResult::variance)
        .def_readwrite("masking_metric", &StackResult::masking_metric)
        .def_readwrite("success_msg", &StackResult::success_msg)
        .def_readwrite("used_path", &StackResult::used_path)
        .def("__repr__", [](const StackResult& r) {
            return "<StackResult energy=" + std::to_string(r.energy) +"M= " + std::to_string(r.masking_metric) + "path=" + r.used_path + ">";
        });

    }

     
