FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

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

# Core dependencies (install WITHOUT qiskit-aer so GPU version takes priority)
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    mpi4py \
    pybind11 \
    qiskit \
    qiskit-nature \
    qiskit-ibm-runtime \
    pyscf

# GPU acceleration: install cupy first, then qiskit-aer-gpu
# cupy-cuda12x works with CUDA 12.x (any minor version)
# qiskit-aer-gpu replaces qiskit-aer with GPU-enabled build
RUN pip3 install --no-cache-dir cupy-cuda12x
RUN pip3 install --no-cache-dir "qiskit-aer-gpu" || \
    echo "WARNING: qiskit-aer-gpu install failed — GPU statevector will use CPU fallback"

# Verify cupy installed; aer-gpu may fail on Qiskit 2.x — stack handles this gracefully
RUN python3 -c "import cupy; print(f'CuPy: {cupy.__version__}')"
RUN python3 -c "from qiskit_aer import AerSimulator; print('Aer GPU: OK')" 2>/dev/null || \
    echo "NOTE: qiskit-aer-gpu incompatible with installed Qiskit version — CPU statevector will be used"

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p build && cd build && \
    cmake .. \
      -DPython_EXECUTABLE=/usr/bin/python3.11 \
      -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

ENV PYTHONPATH="/workspace/build:/workspace"
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV IBM_QUANTUM_TOKEN=""
ENV IBM_QUANTUM_INSTANCE=""
ENV IBM_QUANTUM_BACKEND="ibm_brisbane"
ENV IBM_QUANTUM_REGION="us-east"
ENV BACKEND="simulator"

CMD ["mpirun", "--allow-run-as-root", "-np", "2", "python3", "tests/test_layers_run.py"]