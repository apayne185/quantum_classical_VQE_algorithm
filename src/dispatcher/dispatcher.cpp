#include "stack_types.h"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>  
#include <cmath>
#include <numeric>
#include <future>   
#include <thread>   
#include <string>
#include <chrono>


extern std::string submit_qpu_job(const std::string& qasm, const std::string& backend, int num_shots);
extern double poll_qpu_job(const std::string& job_id);



// MOCK QPU: MPI synchronization overhead, simulator mode returns immediently ->  T_comm is transparent
static double mock_qpu(const std::string& /*qasm*/, int /*shots*/) {
    // std::this_thread::sleep_for(std::chrono::milliseconds(2000+ std::rand() % 1000));
    return 0.0;     
}




// PAULI EXPECTATION VALUE: mean field approximation valid for shallower HWE circuits
// ⟨Z_i⟩= cos(θ_i), ⟨X_i⟩ = sin(θ_i),  ⟨I⟩=1    

static double pauli_expectation(const std::string& op, const std::vector<double>& theta) {
    double val = 1.0;
    int n= static_cast<int>(op.size());   
    int n_par = static_cast<int>(theta.size());
    double stride = (n > 0) ? static_cast<double>(n_par) / n : 1.0;   
 
    for (int i = 0; i < n; ++i) {
        if (op[i] == 'I') continue; 

        int param_idx = std::min(static_cast<int>(i * stride),n_par - 1); 

        // map qubit idx i to parameter idx  (HWE with reps=1 has 2*n_qubits parameters, 1st layer use θ[i], 2nd uses θ[i+n]
        double t= theta[param_idx];
        switch (op[i]) {
            // case 'I': break;                                //identity matrix,  multiply by 1
            case 'Z': val *= std::cos(t);  break; 
            case 'X': val *= std::sin(t);  break;
            case 'Y': val *= std::sin(t);  break;  
            default: break;    
        }
    }
    return val;   
}   




// LOCAL HAMILTONIAN EXPECTATION VALUE: sums coeff * P over ranks partition of pauli terms
// uses mean-field approximation (independent qubits) - valid for product states
// entangled ansatzes (HWE/UCC), use Python statevector path instead.
static double compute_hamiltonian_expectation(const std::vector<PauliTerm>& pauli_terms, const std::vector<double>& theta) {
    double energy = 0.0;
    for (const auto& term : pauli_terms) {
        energy += term.coeff * pauli_expectation(term.op, theta);
    }
    return energy;
}



// CUDA PATH: uses kernel for FP32 trig then widens into FP64, computes -0.5*cos(θ_i) p parameter as fast approx of Z expectation weighted sum
// result scaled by sum of absolte Pauli coefficients 
static double compute_expectation_cuda(const std::vector<double>& theta,const std::vector<PauliTerm>& pauli_terms){
    std::vector<float> fp32(theta.begin(), theta.end());
    double kernel_val= run_cuda_vqe_fp32(fp32.data(), static_cast<int>(fp32.size()));
 
    // scale kernel output by sum of abs coeffs for correct energy units
    double coeff_sum = 0.0;
    for (const auto& t : pauli_terms) coeff_sum += std::abs(t.coeff);
    if (coeff_sum<1e-12) coeff_sum = 1.0;
 
    // kernel produces sum of -0.5*cos(θ_i) over n params - normalaize (divide by n for per parameter avg, scale by coeff_sum)
    double n = static_cast<double>(theta.size());
    return (n > 0) ? (kernel_val/ n)*coeff_sum : 0.0;
}



StackResult route_workload(HybridWorkload& wl) {
    StackResult res; 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > 1 && size % 2 != 0) {
        if (rank == 0) {
            fprintf(stderr, "[Dispatcher] ERROR: odd number of MPI ranks (%d) isn't supported. Even/odd rank split for e_plus/e_minus requires even rank count. Re-run with NP=2,4,6.. \n",  size);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const double start_time = MPI_Wtime();

    // BROADCAST:  metadata syncronization 
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



    //BROADCAST NONBLOCKING: θ, QASM   
    MPI_Request requests[2];                    // tracks async broadcasts

    // starts async broadcast 
    MPI_Ibcast(wl.parameters.data(), param_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Ibcast(qasm_buffer.data(), qasm_size, MPI_CHAR, 0, MPI_COMM_WORLD, &requests[1]);


    // ASYNCHRONOUS QPU JOB - rank 0 starts QPU RTT
    std::future<double> qpu_future; 
    if (rank ==0){
        std::cout << "[MASTER] Dispatching QASM to Cloud QPU.. " << wl.backend_target << "..."<< std::endl;
        const std::string backend = wl.backend_target;
        const std::string qasm = wl.circuit_qasm;
        const int shots = wl.num_shots; 

        if (backend== "ibm_cloud"){
            qpu_future = std::async(std::launch::async, [qasm, backend, shots]() {      // ISSUE HERE?
                std::string job_id = submit_qpu_job(qasm, backend, shots);
                return poll_qpu_job(job_id);
            });
        } else {   //simulator
            qpu_future = std::async(std::launch::async, mock_qpu, qasm, shots);
        }

    }

    //WAIT FOR BROADCASTS: ensures data arrives before starting CUDA kernls
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    if (rank != 0) {
        wl.circuit_qasm = std::string(qasm_buffer.begin(), qasm_buffer.begin() + qasm_size);
    }

    //splt combined params 
    const int num_params = param_size /2;
    std::vector<double> theta_plus (wl.parameters.begin(), wl.parameters.begin() + num_params);
    std::vector<double> theta_minus (wl.parameters.begin() + num_params, wl.parameters.end());



    // ACCELERATION LAYER: Pauli expectation value eval
    double t_accel_start = MPI_Wtime();
    double e_plus_local = 0.0;
    double e_minus_local = 0.0; 

    int deviceCount= 0;
    cudaGetDeviceCount(&deviceCount);
    const bool use_cuda = wl.requires_gpu && (deviceCount > 0);

    if (use_cuda) {
        // CUDA kernel uses mean-field approximation — not physically correct for entangled ansatzes.
        // For accurate results, use the Python statevector path (simulator mode).
        if (rank == 0) {
            fprintf(stderr, "[Dispatcher] WARNING: CUDA path uses mean-field approximation. "
                    "Results will be approximate. Use simulator backend for exact statevector.\n");
        }
        e_plus_local = compute_expectation_cuda(theta_plus, wl.pauli_terms);
        e_minus_local = compute_expectation_cuda(theta_minus, wl.pauli_terms);

        res.used_path = (size > 1) ? "MPI + CUDA (mean-field approx)" : "Single Rank CUDA (mean-field approx)";
    } else {
        e_plus_local = compute_hamiltonian_expectation(wl.pauli_terms, theta_plus);
        e_minus_local = compute_hamiltonian_expectation(wl.pauli_terms, theta_minus);

        res.used_path = (size > 1) ? "MPI + CPU (mean-field approx)" : "Single Rank CPU (mean-field approx)";
    }
    const double t_accel_end = MPI_Wtime();



    // QPU RESULT: collect 
    double qpu_val = 0.0;
    double t_qpu_wait_ms = 0.0; 

    if (rank == 0 && qpu_future.valid()){
        const double t_wait_start = MPI_Wtime();
        qpu_val = qpu_future.get();
        t_qpu_wait_ms = (MPI_Wtime() - t_wait_start)* 1000.0;
        std::cout << "[Manager] QPU result = " << qpu_val << "  (residual wait = " <<t_qpu_wait_ms << "ms)" << std::endl; 
    }

    MPI_Bcast(&qpu_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 




    // GLOBAL REDUCTION:  non-blocking 
    double e_plus_global = 0.0;
    double e_minus_global = 0.0;
    MPI_Allreduce(&e_plus_local, &e_plus_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&e_minus_local, &e_minus_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
    if (rank ==0){
        std::cout << "[Reduce] E+ Global : " << e_plus_global << "  | E- global: " << e_minus_global << "  QPU= " << qpu_val << std::endl;
    }


    // PACK RESULT: simulator (energy comes form pauli eval), ibm_cloud (energy comes from measured expectation value, repalces estimate)
    if (wl.backend_target == "ibm_cloud") {
        res.energy= qpu_val;
        res.e_minus = qpu_val;      
    } else {
        res.energy = e_plus_global;
        res.e_minus = e_minus_global;
    }   

    res.success_msg = "OK";       //success
    res.execution_time = MPI_Wtime() - start_time;

    double t_accel = t_accel_end - t_accel_start;
    double t_comm = res.execution_time - t_accel;
    const double diff = e_plus_global - e_minus_global;
    res.masking_metric= (t_comm > 1e-9) ? (t_accel/ t_comm) : 0.0;
    res.variance= (diff*diff) / 4.0;

    return res; 
}
    
