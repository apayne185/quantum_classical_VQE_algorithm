FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopenmpi-dev \
    openmpi-bin \
    libcurl4-openssl-dev \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libgfortran5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1


RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    mpi4py \
    pybind11 \
    qiskit \
    qiskit-nature \
    qiskit-ibm-runtime \
    pyscf

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p build && cd build && \
    cmake .. \
      -DPython_EXECUTABLE=/usr/bin/python3.11 \
      -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

ENV PYTHONPATH="/workspace/build:/workspace"

ENV IBM_QUANTUM_TOKEN=""
ENV IBM_QUANTUM_INSTANCE=""
ENV IBM_QUANTUM_BACKEND="ibm_brisbane"
ENV IBM_QUANTUM_REGION="us-east"
ENV BACKEND="simulator"

CMD ["mpirun", "--allow-run-as-root", "-np", "2", "python3", "tests/test_layers_run.py"]