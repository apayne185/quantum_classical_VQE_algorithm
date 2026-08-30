# AWS Deployment (multi-cloud validation)

Bootstrap for running HPCHybridStack on an AWS EC2 GPU instance. Purpose:
**multi-cloud validation** — reproduce a subset of the A100 (Lambda) results on
AWS hardware so the paper can claim vendor-portability, not Lambda-lock-in.

Not for Amazon Braket. Braket is a **QPU service**, not classical GPU support;
integrating it belongs behind the QPU backend plugin system in
`docs/FUTURE_WORK.md`.

Cross-refs: `docs/BASELINE_COMPARISON.md` (the workload to run once
bootstrapped), `docs/GPU_TROUBLESHOOTING.md` (fallback if Aer-GPU refuses to
load on the AWS AMI).

---

## Target instance

Recommended: **g5.xlarge** — NVIDIA A10G 24GB, 4 vCPU, 16 GB RAM,
$1.006/hour on-demand (us-east-1, 2026 pricing — verify current rate before
launching).

Why A10G:
- Datacenter-class GPU (Ampere generation like A100) — falls into the
  `HardwareProfile._GPU_DATABASE` datacenter bucket, so precision auto-selects
  fp64 unchanged.
- 24GB HBM comfortably fits the canonical 4-molecule set (max is H2O @ 14q,
  ~262KB statevector fp64).
- Cheapest g5 tier; g5.2xlarge / g5.12xlarge only needed for scaling runs
  above NP=4.

Alternatives considered:
- **p3.2xlarge** (V100 16GB, ~$3/hr) — older gen, no cost benefit.
- **p4d.24xlarge** (8× A100 40GB, ~$32/hr) — overkill for reproduction;
  reserve for a proper multi-GPU distributed-statevector run once the
  rearchitecture in `docs/FUTURE_WORK.md` lands.
- **g4dn.xlarge** (T4 16GB, ~$0.53/hr) — T4 is workstation-class; would
  trigger fp32 auto-selection and diverge from the Lambda dataset.

## Prerequisites (local machine)

1. `aws-cli` installed and `aws configure` run with a working access key.
2. An EC2 key pair; export its path:
   `export AWS_KEY=~/.ssh/aws-vqe.pem` (chmod 400).
3. A security group allowing SSH from your IP.
4. Optional: `aws sso login` if the account uses SSO.

## What the bootstrap script does

`scripts/aws_deploy.sh` — see file for the full flow. Summary:

1. Launches a g5.xlarge from the Deep Learning AMI (Ubuntu 22.04, CUDA 12.4
   pre-installed).
2. Waits for SSH readiness.
3. Uploads the current repo (`rsync`, excluding `.git`, `results/`,
   `__pycache__`, and `.pubchem_cache`).
4. Runs `install_native.sh` on the instance to build `hpc_core` + the conda
   env.
5. Runs `make pytest` for a smoke check.
6. Prints the SSH command for follow-up interactive work.

Cleanup is manual (`aws ec2 terminate-instances --instance-ids <id>`) —
**deliberate**, so a rsync-back-results step is never skipped. Lost data in
the past because a script auto-terminated (see `.claude-docs/SESSION_HISTORY.md`
mentions of the Lambda drop-mid-CO2 incident — same failure mode).

## Recommended workload once bootstrapped

Estimated cost: ~$2 for a 90-minute session.

1. `make run NP=2 MOLECULES="H2 LiH BeH2 H2O"` — reproduces the seed=42 P=2
   sweep. Compare the resulting JSON energies against
   `results/a100-sxm4-40gb/simulator/simulator_20260727_220513.json` — same
   backend family, different silicon, should agree to <1 mHa.
2. If baseline comparison is also planned this session:
   `python -m benchmarks.baseline_comparison --backend hpchybrid --molecule H2 --max-iters 100 --warmup`
3. **Before terminating**:
   `rsync -av ec2-user@<ip>:~/quantum_classical_VQE_algorithm/results/g5-a10g/ results/g5-a10g/`

## Known gotchas

- **AWS Deep Learning AMI ships its own CUDA at `/usr/local/cuda`** — same
  shadowing hazard as `docs/GPU_TROUBLESHOOTING.md` §2. If Aer-GPU fails to
  load, check `LD_LIBRARY_PATH` before diagnosing anything else.
- **g5 instances default to NVMe scratch at `/opt/dlami/nvme`** — cheap
  storage but not backed up. `results/` should stay on the root EBS volume.
- **Instance limits**: fresh AWS accounts often have a 0 vCPU limit on
  g-class. Request a quota increase to at least 4 vCPU (g5.xlarge) via the
  Service Quotas console — takes 24–48h. Do this before you plan to run.
- **Spot instances** are ~70% cheaper but can be reclaimed with 2min notice.
  For a reproducible baseline, use on-demand. Save spot for scaling sweeps
  where a lost run is recoverable.

## Multi-cloud claim — what this establishes

Running the same code + same molecules + same seed on AWS A10G and Lambda
A100, and getting energies that agree to <1 mHa, demonstrates that
HPCHybridStack's `HardwareProfile.detect()` + `results_slug()` auto-detection
works across vendors without code changes. That's the "runs unchanged from
laptop to datacenter" claim in the README, validated across two clouds.

What it does **not** establish:
- Scaling equivalence — A10G ≠ A100 kernel-for-kernel; wall-clock will
  differ by a factor of ~2×.
- Cost equivalence — different instance economics per cloud.
- Full parity — one instance type per cloud is a spot check, not a matrix.

The paper should frame this as "reproducibility across cloud vendors,"
not "portable performance."
