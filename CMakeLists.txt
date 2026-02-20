cmake_minimum_required(VERSION 3.18)
project(HybridStack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})


find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(pybind11 REQUIRED)
find_package(MPI REQUIRED)

# Define module
pybind11_add_module(hpc_core 
    src/api/bridge.cpp 
    src/dispatcher/dispatcher.cpp
    src/classical/cuda/kernel.cu
)

# Configure hpc-core settings 
target_include_directories(hpc_core PRIVATE 
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
# include_directories(${MPI_CXX_INCLUDE_PATH})
# target_include_directories(hpc_core PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Links everything
target_link_libraries(hpc_core PRIVATE 
    Python::Module 
    MPI::MPI_CXX 
    pybind11::module
)


set_target_properties(hpc_core PROPERTIES 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON         #helps w complex kernels 
    POSITION_INDEPENDENT_CODE ON
)   