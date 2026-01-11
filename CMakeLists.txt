cmake_minimum_required(VERSION 3.18)
project(HybridStack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(include)

pybind11_add_module(hpc_core 
    src/api/bridge.cpp 
    src/dispatcher/dispatcher.cpp
)

target_link_libraries(hpc_core PRIVATE Python::Module)