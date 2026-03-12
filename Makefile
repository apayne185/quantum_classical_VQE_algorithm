IMAGE_NAME = vqe-mpi-gpu
INSIDE_CONTAINER = $(shell [ -f /.dockerenv ] && echo yes || echo no)
NP ?= 2       						#override with -  make run NP=4
.PHONY: build run clean

build:
ifeq ($(INSIDE_CONTAINER),yes)
	@echo "Detected: Dev Container. Compiling C++ Core ... "
	mkdir -p build && cd build && cmake .. -DPython_EXECUTABLE=/usr/bin/python3.11 && make -j$(nproc)
else
	@echo "Detected: Host. Building Docker Image ..."
	docker build -t $(IMAGE_NAME) .
endif

run:
ifeq ($(INSIDE_CONTAINER),yes)
	@echo "Running MPI Simulation run with $(NP) ranks inside container ..."
	mpirun --allow-run-as-root -np $(NP) python3 test_run.py
else
	@echo "Running Docker Image from host ..."
	docker run --rm $(IMAGE_NAME)
endif

run-ibm:
	@echo "[Make] Launching MPI run -> IBM Quantum backend …"
	@[ -n "$$IBM_QUANTUM_TOKEN" ] || (echo "ERROR: IBM_QUANTUM_TOKEN not set"; exit 1)
	@[ -n "$$IBM_QUANTUM_INSTANCE" ] || (echo "ERROR: IBM_QUANTUM_INSTANCE not set"; exit 1)
	mpirun --allow-run-as-root -np $(NP) \
	  env BACKEND=ibm_cloud python3 test_run.py

scaling:
	@echo "Starting Strong Scaling Analysis"
	@mkdir -p scaling_logs
	@for p in 1 2 4 8; do \
	  echo "  P=$$p …"; \
	  mpirun --allow-run-as-root -np $$p python3 test_run.py \
	    > scaling_logs/scaling_p$$p.log 2>&1; \
	done
	@echo "Scaling logs generated. Check scaling_p*.log for T_total and M-metric."


clean:
	rm -rf build/
	rm -rf scaling_logs/
	rm -f *.log
	rm -f *.npy
	@if [ "$(INSIDE_CONTAINER)" = "no" ]; then \
		docker rmi $(IMAGE_NAME) || true; \
	fi