#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // for std::vector conversion
#include "stack_types.h"

namespace py = pybind11;

// links Python call to C++ Dispatcher
double execute(HybridWorkload wl) {
    return route_workload(wl);
}

PYBIND11_MODULE(hpc_core, m) {
    py::class_<HybridWorkload>(m, "HybridWorkload")
        .def(py::init<>())
        .def_readwrite("num_qubits", &HybridWorkload::num_qubits)
        .def_readwrite("parameters", &HybridWorkload::parameters)
        .def_readwrite("requires_gpu", &HybridWorkload::requires_gpu);

    m.def("execute", &execute, "The main entry point for the hybrid stack");
}