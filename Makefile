IMAGE_NAME = vqe-mpi-gpu
INSIDE_CONTAINER = $(shell [ -f /.dockerenv ] && echo yes || echo no)

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
	@echo "Running MPI Simulation inside container ..."
	mpirun --allow-run-as-root -np 2 python3 test_run.py
else
	@echo "Running Docker Image from host ..."
	docker run --rm $(IMAGE_NAME)
endif

scaling:
	@echo "Starting Strong Scaling Analysis"
	@echo "Running P=2..."
	mpirun --allow-run-as-root -np 2 python3 test_run.py > scaling_p2.log
	@echo "Running P=4..."
	mpirun --allow-run-as-root -np 4 python3 test_run.py > scaling_p4.log
	@echo "Scaling logs generated. Check scaling_p*.log for T_total and M-metric."


clean:
	rm -rf build/
	rm -f *.log
	rm -f *.npy
	@if [ "$(INSIDE_CONTAINER)" = "no" ]; then \
		docker rmi $(IMAGE_NAME) || true; \
	fi