# Security Policy

## Supported Versions

This is an academic research project (VQE hybrid quantum-classical stack) under active
development. Only the `main` branch is supported with security fixes; feature branches
and archived results are not maintained.

## Reporting a Vulnerability

If you find a security issue (e.g. credential handling, dependency vulnerability,
unsafe deserialization, injection), please report it privately rather than opening a
public issue:

- Preferred: use [GitHub's private vulnerability reporting](../../security/advisories/new)
  for this repository.
- Alternative: email apayne.ieu2022@student.ie.edu with a description of the issue and,
  if possible, steps to reproduce.

Please do not open a public GitHub issue for suspected vulnerabilities until it has
been reviewed and a fix is available.

## Response

This project is maintained by a single researcher, so please allow some time for a
response. I'll acknowledge reports within a reasonable timeframe and aim to fix
confirmed issues before public disclosure.

## Scope Notes

- Credentials (IBM Quantum tokens, HPC cluster access) are always supplied via
  environment variables, never committed to the repository. If you find a committed
  secret, please report it immediately via the channels above.
- This project depends on `qiskit`, `qiskit-ibm-runtime`, `pyscf`, and other
  third-party packages; vulnerabilities in those dependencies should generally be
  reported upstream, but flagging them here is also welcome so this repo can update
  its pins.
