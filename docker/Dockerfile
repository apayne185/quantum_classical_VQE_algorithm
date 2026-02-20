FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 2. Install Python, MPI system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libopenmpi-dev \
    openmpi-bin \
    cmake \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# 3. Install Python dependencies 
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    mpi4py \
    cupy-cuda12x \
    qiskit>=1.0.0 \
    pybind11 \
    qiskit-aer \
    qiskit-algorithms \
    pylatexenc

COPY . .

# RUN rm -rf build && mkdir build && cd build && \
#     cmake .. && \
#     cmake --build . --config Release

# 7. Build the C++ Core
RUN rm -rf build && mkdir build && cd build && \
    cmake .. \
    -DPython_EXECUTABLE=/usr/bin/python3.11 \
    -Dpybind11_DIR=$(python3.11 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    cmake --build . --config Release


ENV PYTHONPATH="/app/build:${PYTHONPATH:+:${PYTHONPATH}}"

# 4. MPI Environment setups
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Use python3.11 as entry point
ENTRYPOINT ["mpirun", "-np", "2", "python3"]
# CMD ["src/classical/vqe.py"]
CMD ["test_run.py"]