#include "stack_types.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>  
#include <numeric>
#include <future>   
#include <thread>   
#include <string>
#include <chrono>


extern std::string submit_qpu_job(const std::string& qasm, const std::string& backend, int num_shots);
extern double poll_qpu_job(const std::string& job_id);


double mock_qpu(const std::string& qasm, int shots) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000+ std::rand() % 1000));
    (void)qasm; 
    return -1.8505;     // Mock QPU value - LiH expectation value
}


StackResult route_workload(HybridWorkload& wl) {
    StackResult res; 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    // double qpu_val = 0.0;
    // std::future<double> qpu_future;

    // metadata syncronization - non blocking
    int param_size = static_cast<int>(wl.parameters.size());
    int qasm_size =  static_cast<int>(wl.circuit_qasm.size());
    int n_pauli =  static_cast<int>(wl.pauli_terms.size()); 
    int num_qubits = wl.num_qubits;
    int num_shots = wl.num_shots;
    
    // broadcast sizes first so workers can allocate the memory
    MPI_Bcast(&param_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&qasm_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_pauli, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&num_qubits, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&num_shots, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    wl.num_qubits = num_qubits;
    wl.num_shots = num_shots;

    //resizes strings on worker nodes
    if (rank != 0) {
        wl.parameters.resize(param_size);
    }

    // explicit QASM serialization into char buffer for MPI-Ibcast
    std::vector<char> qasm_buffer(qasm_size +1, '\0'); 
    if (rank == 0) {
        std::copy(wl.circuit_qasm.begin(), wl.circuit_qasm.end(), qasm_buffer.begin());}


    MPI_Request requests[2];          // tracks async broadcasts

    // starts async broadcast 
    MPI_Ibcast(wl.parameters.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Ibcast(qasm_buffer.data(), qasm_size, MPI_CHAR, 0, MPI_COMM_WORLD, &requests[1]);

    std::future<double> qpu_future; 
    if (rank ==0){
        // rank 0 starts QPU RTT 
        std::cout << "[MASTER] Dispatching QASM to Cloud QPU.." << wl.backend_target << "..."<< std::endl;
        const std::string backend = wl.backend_target;
        const std::string qasm = wl.circuit_qasm;
        const int shots = wl.num_shots; 

        if (backend== "ibm_cloud"){
            qpu_future = std::async(std::launch::async, [qasm, backend, shots]() {      // ISSUE HERE?
                std::string job_id = submit_qpu_job(qasm, backend, shots);
                return poll_qpu_job(job_id);
            });
        } else {
            qpu_future = std::async(std::launch::async, mock_qpu, qasm, shots);
        }

    }

    // ensures data arrives before starting CUDA kernls
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    if (rank != 0) {
        wl.circuit_qasm = std::string(qasm_buffer.begin(), qasm_buffer.begin() + qasm_size);
    }

    const int num_params = param_size /2;
    std::vector<double> theta_plus (wl.parameters.begin(), wl.parameters.begin() + num_params);
    std::vector<double> theta_minus (wl.parameters.begin() + num_params, wl.parameters.end());



    // ACCELERATION LAYER
    // Mixed Precision Strategy - cast result to FP64 for final energy sum
    double t_accel_start = MPI_Wtime();
    double e_plus_local = 0.0;
    double e_minus_local = 0.0;
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    const bool use_cuda = wl.requires_gpu && (deviceCount > 0);

    bool is_batch = (wl.parameters.size() == (size_t)(wl.num_qubits * 2));

    auto compute_expectation = [&](const std::vector<double>& params) ->double {
        if (use_cuda){
            std::vector<float> fp32(params.begin(), params.end()); 
            return run_cuda_vqe_fp32(fp32.data(), static_cast<int>(fp32.size()));
        } else {
            double s = 0.0;
            for (double v: params) s+= v;
            return s;
        }
    }; 

    if (size == 1) {
        e_plus_local = compute_expectation(theta_plus);
        e_minus_local = compute_expectation(theta_minus);
        res.used_path = use_cuda ? "Single Rank CUDA" : "Single Rank CPU"; 
    } else {
        if (rank % 2 == 0) {
            e_plus_local = compute_expectation(theta_plus);
        } else {
            e_minus_local = compute_expectation(theta_minus);
        }
        res.used_path = use_cuda ? "MPI + CUDA Distributed" :"MPI + CPU Fallback";
    }

    const double t_accel_end = MPI_Wtime();
    double qpu_val = 0.0;
    double t_qpu_wait_ms = 0.0; 

    if (rank == 0 && qpu_future.valid()){
        const double t_wait_start = MPI_Wtime();
        qpu_val = qpu_future.get();
        t_qpu_wait_ms = (MPI_Wtime() - t_wait_start) * 1000.0;
        std::cout << "[Manager] QPU result = " << qpu_val << "  (residual wait = " <<t_qpu_wait_ms << "ms)" << std::endl; 
    }

    MPI_Bcast(&qpu_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

    double e_plus_global = 0.0;
    double e_minus_global = 0.0;

    // Global reduction - non-blocking 
    MPI_Allreduce(&e_plus_local, &e_plus_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&e_minus_local, &e_minus_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
    if (rank ==0){
        std::cout << "[Rank " << rank << "] E+ Global : " << e_plus_global << "  | E- global: " << e_minus_global << "QPU= " << qpu_val << std::endl;
    }

    res.energy = e_plus_global + qpu_val;
    res.e_minus= e_minus_global + qpu_val;
    res.success_msg = "OK";    //success
    res.execution_time = MPI_Wtime() - start_time;

    double t_accel = t_accel_end - t_accel_start;
    double t_comm = res.execution_time - t_accel;
    const double delta_e= e_plus_global - e_minus_global;
    res.variance= (delta_e * delta_e) / 4.0;
    // res.variance = is_batch ? (e_minus_global + qpu_val) : 0.0;
    // res.masking_metric = (t_comm > 0) ? (t_accel_end - t_accel_start) / t_comm : 0.0;  
    res.masking_metric   = (t_comm > 1e-9) ? (t_accel/ t_comm) : 0.0;

    return res; 
    }
    
