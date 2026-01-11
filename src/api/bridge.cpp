#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // for std::vector conversion
#include "stack_types.h"

namespace py = pybind11;

// links Python call to C++ Dispatcher
StackResult execute(HybridWorkload wl) {
    return route_workload(wl);
}

PYBIND11_MODULE(hpc_core, m) {
    py::class_<HybridWorkload>(m, "HybridWorkload")
        .def(py::init<>())                              // allows python to do wl=hpc_core.H..W..()
        .def_readwrite("num_qubits", &HybridWorkload::num_qubits)
        .def_readwrite("parameters", &HybridWorkload::parameters)
        .def_readwrite("circuit_depth", &HybridWorkload::circuit_depth)
        .def_readwrite("requires_gpu", &HybridWorkload::requires_gpu)
        .def_readwrite("backend_target", &HybridWorkload::backend_target)
        .def_readwrite("circuit_qasm", &HybridWorkload::circuit_qasm);
        

    py::class_<StackResult>(m, "StackResult")
        .def_readwrite("energy", &StackResult::energy)
        .def_readwrite("execution_time", &StackResult::execution_time)
        .def_readwrite("success_msg", &StackResult::success_msg)
        .def_readwrite("used_path", &StackResult::used_path);  
    

    m.def("execute", &execute, "Main entry point for hybrid stack");
    m.def("route_workload", &route_workload, "Core dispatcher function");
}