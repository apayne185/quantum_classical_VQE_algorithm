cmake_minimum_required(VERSION 3.18)

# Detect CUDA before declaring it as a project language.
# This allows CPU-only builds on machines without NVIDIA GPUs.
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(HybridStack LANGUAGES CXX CUDA)
    set(HAVE_CUDA ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    message(STATUS "[VQE] CUDA found (${CMAKE_CUDA_COMPILER}) — building with GPU support")
else()
    project(HybridStack LANGUAGES CXX)
    set(HAVE_CUDA OFF)
    message(WARNING "[VQE] CUDA not found — building CPU-only. GPU acceleration will be unavailable at runtime. Install CUDA toolkit to enable GPU support.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)


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
find_package(CURL REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Kernel source: GPU (.cu) when CUDA available, CPU stub otherwise
if(HAVE_CUDA)
    set(KERNEL_SOURCE src/classical/cuda/kernel.cu)
else()
    set(KERNEL_SOURCE src/classical/cpu/kernel_stub.cpp)
endif()

pybind11_add_module(hpc_core
    src/dispatcher/bridge.cpp
    src/dispatcher/dispatcher.cpp
    src/dispatcher/qpu_client.cpp
    ${KERNEL_SOURCE}
)

# Propagate HAVE_CUDA to all C++ translation units
if(HAVE_CUDA)
    target_compile_definitions(hpc_core PRIVATE HAVE_CUDA)
endif()

target_include_directories(hpc_core PRIVATE
    include
    ${MPI_CXX_INCLUDE_DIRS}
)
if(HAVE_CUDA)
    target_include_directories(hpc_core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

target_link_libraries(hpc_core PRIVATE
    Python::Module
    MPI::MPI_CXX
    pybind11::module
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

set_target_properties(hpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(HAVE_CUDA)
    set_target_properties(hpc_core PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

target_compile_options(hpc_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
)