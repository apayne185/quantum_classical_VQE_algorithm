IMAGE_NAME = vqe-mpi-gpu
NP ?= 2       						#override with -  make run NP=4

ifneq (,$(wildcard .env))
  include .env
  export
endif
 
GPU_AVAILABLE := $(shell docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 && echo yes || echo no)
ifeq ($(GPU_AVAILABLE),yes)    
  GPU_FLAG = --gpus all  
  $(info [Make] GPU detected — CUDA acceleration enabled.)
else   
  GPU_FLAG =  
  $(info [Make] No GPU detected — falling back to CPU mode.)   
endif     


.PHONY: build trial run run-ibm scaling baseline clean shell

build:
	@echo "[Make] Building Docker image '$(IMAGE_NAME)' ..."
	docker build -t $(IMAGE_NAME) .
	@echo "[Make] Build complete."


# DIAGNOSTIC - tests the 6 layers on simulator
trial:
	@echo "[Make] Running diagnostic trial (simulator, $(NP) ranks) ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
 	  mpirun --allow-run-as-root -np $(NP) python3 tests/test_layers_run.py       
 


# FULL BENCHMARK - simualtor only
run:
	@echo "[Make] Running full benchmark (simulator, $(NP) ranks) ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  -v "$$(pwd)/results:/workspace/results" \
	  $(IMAGE_NAME) \
	  mpirun --allow-run-as-root -np $(NP) python3 benchmarks/local_test_run.py




# # FULL BENCHMArk - IBM quantum QPU 
run-ibm:
	@[ -n "$(IBM_QUANTUM_TOKEN)" ] || (echo "ERROR: IBM_QUANTUM_TOKEN not set in .env"; exit 1)
	@[ -n "$(IBM_QUANTUM_INSTANCE)" ] || (echo "ERROR: IBM_QUANTUM_INSTANCE not set in .env"; exit 1)
	@echo "[Make] Running $(NP) ranks -> IBM Quantum ($(IBM_QUANTUM_BACKEND)) ..."
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=ibm_cloud \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -e IBM_QUANTUM_TOKEN="$(IBM_QUANTUM_TOKEN)" \
	  -e IBM_QUANTUM_INSTANCE="$(IBM_QUANTUM_INSTANCE)" \
	  -e IBM_QUANTUM_BACKEND="$(IBM_QUANTUM_BACKEND)" \
	  -e IBM_QUANTUM_REGION="$(IBM_QUANTUM_REGION)" \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  -v "$$(pwd)/results:/workspace/results" \
	  $(IMAGE_NAME) \
	  mpirun --allow-run-as-root -np $(NP) python3 benchmarks/ibm_test_run.py


# STRONG SCALAING SWEEP - simulator      
scaling:
	@echo "[Make] Starting strong scaling analysis ..."
	@mkdir -p results/scaling
	@for p in 1 2 4 8; do \
	  echo "  Running P=$$p ..."; \
	  docker run --rm \
	    $(GPU_FLAG) \
	    -e BACKEND=simulator \
		-e USE_GPU=$(GPU_AVAILABLE) \
	    -v "$$(pwd)/results/scaling:/workspace/results/scaling" \
	    $(IMAGE_NAME) \
	    mpirun --allow-run-as-root -np $$p python3 benchmarks/local_test_run.py \
	    > results/scaling/scaling_p$$p.log 2>&1; \
	  echo "  P=$$p done."; \
	done
	@echo "[Make] Scaling logs saved to results/scaling/. Check T_total and M-metric."



# SERIAL BASELINE - single-core Qiskit VQE for comparison (no MPI)
baseline:
	@echo "[Make] Running serial Qiskit baseline (no MPI, no GPU) ..."
	docker run --rm \
	  -e USE_GPU=no \
	  -v "$$(pwd)/results:/workspace/results" \
	  $(IMAGE_NAME) \
	  python3 benchmarks/serial_baseline.py
	@echo "[Make] Serial baseline complete."


# RUN ALL TESTS- resolver + layer diagnostic
test:
	@echo "[Make] Running test suite ..."
# 	python3 tests/test_resolver.py
	python3 tests/test_molecules_run.py
	docker run --rm \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
	  mpirun --allow-run-as-root -np 2 python3 tests/test_layers_run.py
	@echo "[Make] All tests complete."



shell:
	docker run --rm -it \
	  $(GPU_FLAG) \
	  -e BACKEND=simulator \
	  -e USE_GPU=$(GPU_AVAILABLE) \
	  -v "$$(pwd)/checkpoints:/workspace/checkpoints" \
	  $(IMAGE_NAME) \
	  /bin/bash


clean:
	docker rmi $(IMAGE_NAME) || true
	rm -rf results/scaling/
	rm -f *.log *.npy
	find checkpoints/ -name "*.npy" -delete 2>/dev/null || true