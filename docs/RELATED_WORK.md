# Related Work — Positioning vs. Existing Quantum–HPC Software Stacks

Positioning of this stack (referred to below as **HPCHybridStack**) against
comparable quantum–HPC middleware. Framed against the openQSE Reference
Architecture from Shehata & Austin (2026), *"Quantum–HPC Software Stacks and the
openQSE Reference Architecture: A Survey"*, plus two widely used single-vendor
simulators (Pennylane Lightning, Qiskit Aer MPI) that sit alongside those
stacks in practice.

The goal is a paper-ready comparison table plus a short paragraph per
competitor calling out **what they do that we don't** and **what we do that
they don't**.

---

## openQSE reference architecture — mapping our layers

Shehata & Austin propose five reference layers:

1. **Application** — chemistry, finance, ML clients calling into the stack.
2. **Compilation / Circuit Transformation** — ansatz construction, gate
   decomposition, transpilation.
3. **Orchestration / Scheduling** — dispatch across CPU / GPU / QPU, MPI
   coordination, job queuing.
4. **Runtime / Execution** — simulator backends, physical-device drivers.
5. **Hardware Abstraction** — device catalogs, calibration data, error models.

Mapping of HPCHybridStack:

| openQSE layer | HPCHybridStack component |
|---|---|
| Application | `ChemistryProblem`, `FinanceProblem`, `benchmarks/*.py` |
| Compilation | Qiskit Nature (JW mapping) + `AnsatzBuilder` (HWE / UCCSD tiers) |
| Orchestration | `HPCHybridStack.run()`, MPI Pauli-term distribution (`hpc_core`) |
| Runtime | Qiskit Aer statevector (CPU/GPU/cuStateVec), IBM Runtime EstimatorV2 |
| Hardware Abstraction | `HardwareProfile.detect()`, `_GPU_DATABASE`, `results_slug()` |

Under this taxonomy we are a **thin, opinionated middleware** across all five
layers, not a full framework — which is deliberate and is the main
positioning point below.

---

## Comparison table

| Feature / Property | HPCHybridStack (this repo) | JHPC-Quantum | Quantum Brilliance (Qristal) | Tierkreis (Quantinuum) | Pennylane Lightning | Qiskit Aer MPI |
|---|---|---|---|---|---|---|
| **Primary intent** | End-to-end VQE benchmark middleware | HPC-integrated hybrid workflow platform | Full SDK (compiler + runtime + hardware) | Higher-order dataflow orchestrator for hybrid workflows | Single-node high-perf simulator | Multi-node distributed simulator |
| **Scope** | Middleware (thin) | Full framework | Full SDK | Orchestration only | Simulator only | Simulator only |
| **CPU + GPU + QPU in one run** | Yes (proven end-to-end, IBM triple-integration) | Partial (GPU via plugin) | Yes (with Qristal hardware) | Delegates to backends | GPU only, no QPU | CPU/GPU only, no QPU |
| **MPI-distributed** | Yes (Pauli-term parallelism) | Yes | No | Via backend | Yes (Lightning-MPI) | Yes |
| **Distributed statevector** | No (per-rank replicated; see `docs/FUTURE_WORK.md`) | Yes (via backend) | No | Via backend | Yes (Lightning-MPI, cuStateVec) | Yes |
| **Auto hardware detection** | Yes (`HardwareProfile.detect()`) | No | Partial | No | No | No |
| **Auto precision selection** | Yes (fp32 for consumer GPUs, fp64 for datacenter) | No | No | No | Manual | Manual |
| **QPU backend plugin system** | No (IBM only; see FUTURE_WORK) | Yes | Vendor-locked to Qristal | Yes (multiple) | No | No |
| **Error mitigation** | T-REx (via IBM Runtime) | Configurable | Yes | Delegates | No | No |
| **Reproducibility (containerized)** | Yes (Docker + install_native.sh + Slurm) | Partial | Yes | Partial | Yes | Yes |
| **Statistical methodology built-in** | Yes (n=5 seeds, best-of-N, chemical-accuracy check) | No | No | No | No | No |
| **Published benchmarks across hardware classes** | Yes (RTX 6000 + A100 + IBM QPU, all committed) | Reference numbers only | Vendor-provided | N/A | Reference | Reference |

---

## Per-competitor notes

### JHPC-Quantum

Full HPC-integrated hybrid workflow platform from the openQSE survey — targets
supercomputing centers, integrates with SLURM/PBS, and supports distributed
statevector via pluggable simulator backends.

**They do that we don't:**
- True distributed statevector (via backend plugin — cuStateVec multi-GPU or
  QuEST-MPI).
- Multi-scheduler support (SLURM + PBS + LSF) as a first-class concern.

**We do that they don't:**
- End-to-end **triple integration** (CPU + GPU + QPU) proven in a single run
  path — JHPC-Quantum treats these as separate workflows.
- **Auto hardware detection + auto precision** — same Python entry point runs
  unchanged on a laptop, a Lambda A100, and an IBM QPU. JHPC-Quantum requires
  manual per-target configuration.
- **Statistical methodology baked in** (n=5 seeds, best-of-N, chemical-accuracy
  check against FCI). JHPC-Quantum is a platform, not a benchmark suite.
- **Full committed reproducible dataset** across RTX 6000 + A100 + IBM QPU.

### Quantum Brilliance (Qristal)

Vendor SDK from Quantum Brilliance — full compiler + runtime + hardware
control, targeting their diamond NV-center devices.

**They do that we don't:**
- Full compilation stack (gate synthesis, hardware-aware optimization).
- Native error mitigation for their specific hardware.
- Formal quantum-classical shared-memory execution model.

**We do that they don't:**
- **Vendor-neutral QPU access** (currently IBM, plugin system in
  `docs/FUTURE_WORK.md` would generalize to IonQ / Rigetti / Braket). Qristal
  is Qristal-hardware-only.
- **Cross-hardware benchmark story** — Qristal publishes vendor numbers on
  their own device; we publish comparable numbers across three hardware
  classes.
- Lighter weight and easier to adopt: `pip install` + Docker, versus a full
  SDK stack.

### Tierkreis (Quantinuum)

Higher-order dataflow orchestrator — treats hybrid quantum-classical workflows
as typed graphs of pure functions, with backends pluggable behind a common
interface.

**They do that we don't:**
- **Formal dataflow model** — every workflow step is a typed pure function;
  scheduling, retry, and parallelism are automatic from the graph structure.
- **Multiple QPU backends** as a first-class abstraction.
- Serializable / migratable workflow state (workflows can pause and resume
  across processes).

**We do that they don't:**
- **Concrete VQE end-to-end path** — Tierkreis is orchestration only; the user
  still has to write the ansatz builder, expectation-value pipeline, optimizer
  loop, MPI dispatch. HPCHybridStack ships all of that.
- **HPC-first design** — MPI is a first-class execution mode (`mpirun -np N`),
  not a plugin. Tierkreis defers all HPC concerns to backends.
- **Hardware auto-detection + auto precision** at the middleware layer.

### Pennylane Lightning (LightningGPU / Lightning-MPI)

Single-node high-performance statevector simulator from Xanadu, with
GPU (cuStateVec) and MPI variants. Not middleware — a simulator backend that
plugs into Pennylane.

**They do that we don't:**
- **True distributed statevector** via cuStateVec multi-GPU API — the exact
  thing our gap H (`docs/KNOWN_GAPS.md`) calls out as our leading missing
  feature.
- Native autodiff / analytic gradients for VQE optimizers (versus our SPSA
  default, see gap J).
- More mature GPU kernel optimization at the simulator layer.

**We do that they don't:**
- **QPU integration** — Pennylane Lightning is GPU/CPU only; running on
  hardware needs a separate Pennylane device.
- **End-to-end benchmark suite + statistical methodology** — Lightning is a
  simulator; we are a benchmark platform that happens to use a simulator.
- **Hardware profile auto-detection** — same code detects the target device
  and picks precision/backend automatically.

**Baseline comparison plan**: `docs/BASELINE_COMPARISON.md` will define an
apples-to-apples wall-clock comparison for H2/LiH/BeH2/H2O across
HPCHybridStack (Aer-GPU) vs Pennylane Lightning-GPU on the same A100.

### Qiskit Aer MPI

The multi-node MPI mode of Qiskit's Aer simulator. Same simulator we use, but
run in its distributed-statevector configuration rather than the
per-rank-replicated mode we use today.

**They do that we don't:**
- **Distributed statevector across MPI ranks** (native Aer feature — not what
  we invoke). Same underlying capability as Lightning-MPI, wrapped in Aer's
  interface.
- **Larger single-molecule ceiling** for a given GPU memory budget — the CO2
  failure mode (`docs/KNOWN_GAPS.md` gap H) would not exist if we used Aer
  MPI's distributed mode.

**We do that they don't:**
- **Middleware around Aer** — Aer MPI is the simulator; we add the ansatz
  builder, chemistry problem definition, optimizer loop, IBM QPU integration,
  hardware profile, statistical methodology, and results/reporting layer.
- **Auto-selection** — Aer MPI requires the user to explicitly configure
  distributed mode; our stack picks backend automatically.
- Reproducible **cross-hardware benchmark suite**.

**Baseline comparison plan**: `docs/BASELINE_COMPARISON.md` will also cover
HPCHybridStack (Aer-GPU, replicated) vs Aer MPI (distributed) on the same
molecules — this directly measures the cost of our current architectural
choice.

---

## Summary — the paper positioning

<!-- paper -->
HPCHybridStack occupies a niche not filled by any of the surveyed stacks: a
**thin, benchmark-oriented middleware** that ships an end-to-end VQE pipeline
(chemistry problem → JW mapping → HWE/UCCSD ansatz → SPSA/Aer/IBM execution →
statistical reporting) with automatic hardware detection and a proven
CPU+GPU+QPU triple-integration. Full-framework offerings such as JHPC-Quantum
and Qristal cover more surface but require per-target configuration and lack
built-in statistical methodology. Orchestration-only frameworks such as
Tierkreis are more general but require the user to supply the entire VQE
pipeline themselves. High-performance simulators such as Pennylane Lightning
and Qiskit Aer MPI beat us on raw simulation ceiling — they implement genuine
distributed statevector where we currently replicate per rank — but do not
address hybrid orchestration or QPU integration at all. The distributed
statevector gap is the leading follow-up item (`docs/FUTURE_WORK.md`).
<!-- /paper -->

---

## Baseline comparison — deferred detail

See `docs/BASELINE_COMPARISON.md` (skeleton pending) for the concrete
apples-to-apples plan vs Pennylane Lightning-GPU and Qiskit Aer MPI, and
`docs/FUTURE_WORK.md` for the distributed-statevector rearchitecture that
would close the raw-performance gap.
