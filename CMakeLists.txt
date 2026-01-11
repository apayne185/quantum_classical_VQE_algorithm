cmake_minimum_required(VERSION 3.18)
project(HybridStack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(include)

pybind11_add_module(hpc_core 
    src/api/bridge.cpp 
    src/dispatcher/dispatcher.cpp
    src/classical/cuda/kernel.cu
)

target_link_libraries(hpc_core PRIVATE Python::Module)
set_target_properties(hpc_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)   