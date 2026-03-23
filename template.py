"""
Hybrid Quantum-Classical VQE — Quantum Chemistry Research Template
=================================================

This template demonstrates how to compute the electronic ground-state energy of a molecule using the 
distributed VQE middleware stack. It is designed for quantum chemistry researchers who may 
not be familiar with MPI or Docker.


Before running this file:
    1. Build the Docker image (a one time setup) by running the command below on the terminal:
           make build

    2. Verify the stack is functional and ready for usage:
           make trial                # runs a 7 layer diagnostic test

    3. (Optional) View all available built in molecules:
           make molecules           # prints the live molecule registry

    4. Run this template file:
           make example NP=2         # 2 MPI ranks (recommended minimum ranks)
           make example NP=4         # 4 MPI ranks (more parallelism)

    Or run directly inside Docker (run this command in the terminal):
           docker run --rm vqe-mpi-gpu mpirun --allow-run-as-root -np 2 python3 template.py

           
Where results are stored:
    - Terminal output: printed live during the run
    - Iteration logs: results/simulator/run_<timestamp>.log (every print statement)
    - Structured data:results/simulator/simulator_<timestamp>.json (energies, timing, history)
    - Checkpoints: checkpoints/<molecule>/checkpoint_iter_XXXX.npy (recoverable state)

    
For IBM Quantum (real QPU) runs:
    1. Copy .env.example to .env and fill in your personal IBM credentials (ensure these remain private). 
    2. Run on terminal:  make run-ibm NP=2

    (See the README file for detailed IBM setup instructions).

For full documentation:
    - README.md - project overview, all make targets, architecture
    - MOLECULES.md - built-in molecules, custom input formats, Optional: adding new molecules

    """





import os
import json
import numpy as np
from datetime import datetime
from src.api.interface import HPCHybridStack
from src.api.problems import ChemistryProblem
from src.api.log import init_log, close_log

# Auto-save all terminal output to a log file
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results/simulator", exist_ok=True)
init_log(f"results/simulator/template_{_ts}.log")


# ═══════════════════════════════════════════════════════════════
# Edit this section for your experiment
# ═══════════════════════════════════════════════════════════════

# Molecule Selection  ────────────────────────────────────────

# Option A: Built-in molecule (run 'make molecules' on terminal to see full list of avaliable molecules) 
MOLECULE = "H2"
problem = ChemistryProblem.from_name(MOLECULE)


# Option B: Custom geometry (atom positions in Angstroms, separated by ; and ,)
# problem = ChemistryProblem(
#     "O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
#     name="water",
#     reps=2,                   # ansatz depth (higher= more expressive but slower)
# )

# Option C: Molecule resolver (common names, SMILES strings, PubChem lookup)
# from src.api.molecule_resolver import MoleculeResolver
# resolver = MoleculeResolver(max_qubits=20)
# info= resolver.resolve("methane")       # or "CCO" (SMILES), or raw geometry
# problem = ChemistryProblem(info.geometry, name=info.name)


# Optimizer Settings   ────────────────────────────────────────
MAX_ITERATIONS = 200            # more iterations = better accuracy, longer runtimes (rule of thumb: 200 for H2, 800+ for LiH/BeH2/H2O)
SEED = 42                        # fixed seed for reproducibility 
CONVERGENCE_TOL= 1.6e-3         # chemical accuracy threshold in Hartrees


# Backend Selection ─────────────────────────────────────────
BACKEND = "simulator"       # "simulator" = exact statevector (noiseless, fast) or "ibm_cloud" = real QPU (requires .env credentials files)



# Execution ─────────────────────────────────────────
with HPCHybridStack(backend=BACKEND) as stack:

    # All ranks prepare the problem (needed for MPI-distributed evaluation)
    problem.prepare()

    # Print experiment configuration and circuit (rank 0 only)
    if stack.rank == 0:
        print("VQE Ground-State Energy Computation")
        print("=" * 60)
        print(f"Molecule:       {problem.name}")
        print(f"Backend:        {BACKEND}")
        print(f"MPI ranks:      {stack.size}")
        print(f"GPU:            {'enabled' if stack.use_gpu else 'disabled'}")
        print(f"Max iterations: {MAX_ITERATIONS}")
        print(f"Seed:           {SEED}")
        print(f"Convergence:    {CONVERGENCE_TOL} Ha")
        print()

        # Display the ansatz circuit
        print("ANSATZ CIRCUIT")
        print("-" * 60)
        ansatz = problem.ansatz_circuit
        print(ansatz.draw(output="text", fold=80))
        print(f"\n  Depth: {ansatz.depth()}  |  Gates: {ansatz.size()}  |  Parameters: {ansatz.num_parameters}")
        print()
        print()

        # Display Hamiltonian summary
        print("HAMILTONIAN")
        print("-" * 60)
        n_terms = len(problem.pauli_terms)
        print(f"  {n_terms} Pauli terms, {problem.num_qubits} qubits")
        for op, coeff in problem.pauli_terms[:5]:
            print(f"    {op}  {coeff:+.6f}")
        if n_terms > 7:
            print(f"    ... ({n_terms - 7} more terms)")
        for op, coeff in problem.pauli_terms[-2:]:
            print(f"    {op}  {coeff:+.6f}")
        print()

    # Synchronize before starting VQE
    stack.comm.Barrier()


    # Run VQE optimization
    theta,history = stack.vqe_optimize(
        problem,    
        max_iterations=MAX_ITERATIONS, 
        tolerance=CONVERGENCE_TOL,
        seed=SEED,
        checkpoint_dir=f"checkpoints/{problem.name}",
    )

    # Report results (rank 0 only)
    if stack.rank == 0 and history:
        fci = problem.fci_energy
        final_energy = history[-1]

        # Check if best physical energy was tracked (HWE may cross below FCI)
        best_energy = getattr(stack,'_best_physical_energy', final_energy)
        best_iter = getattr(stack,'_best_physical_iter', len(history))     

        if best_energy > final_energy and fci is not None and final_energy < fci:
            report_energy = best_energy
            report_iter = best_iter
            crossed_fci = True
        else:
            report_energy = final_energy
            report_iter = len(history)
            crossed_fci = False

        print()
        print("RESULTS")
        print("=" * 60)
        print(f"Molecule: {problem.name}")
        print(f"Qubits: {problem.num_qubits}")
        print(f"Pauli terms:{len(problem.pauli_terms)}")
        print(f"Parameters: {problem.num_params}")
        print(f"Ansatz: {problem.ansatz_tier}")
        print(f"Iterations: {len(history)} (best at iter {report_iter})")
        print(f"Final energy: {report_energy:.6f} Ha")


        if fci is not None:
            error = abs(report_energy - fci)
            print(f"FCI reference: {fci:.6f} Ha")
            print(f"Absolute error: {error:.4f} Ha ({error*1000:.1f} mHa)")

            if error < 1.6e-3:
                print(f"Status: CHEMICAL ACCURACY ACHIEVED")
            elif error < 0.01:
                print(f"Status: Near chemical accuracy")
            else:
                print(f"Status: Converging (increase MAX_ITERATIONS)")

            if crossed_fci:
                print(f"Note: HWE ansatz crossed below FCI at iter {report_iter}.")
                print(f"Reported energy is the best physical (above FCI) value.")




        # Save results to JSON
        os.makedirs("results/simulator", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "molecule": problem.name,
            "energy": report_energy,
            "fci": fci,    
            "error": abs(report_energy- fci) if fci else None,
            "iterations":len(history),         
            "best_iter": report_iter,
            "qubits":problem.num_qubits, 
            "pauli_terms": len(problem.pauli_terms),    
            "params": problem.num_params, 
            "ansatz": problem.ansatz_tier, 
            "backend": BACKEND,
            "mpi_ranks": stack.size,
            "gpu": stack.use_gpu,
            "seed":SEED,
            "history":history,
        }    


        out_path = f"results/simulator/template_{problem.name}_{ts}.json"   

        with open(out_path, "w") as f:
            json.dump(result_data, f, indent=2)  

        print(f"\nResults saved to: {out_path}")
        print(f"Checkpoints in: checkpoints/{problem.name}/")
        print(f"Full log saved to: results/simulator/template_{_ts}.log")

close_log()