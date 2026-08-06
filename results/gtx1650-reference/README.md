# GTX 1650 — no raw data recovered

This was the preliminary/exploratory GPU (NVIDIA GeForce GTX 1650 Mobile,
Lambda Cloud) used to get the stack working before moving to the RTX 6000
Ada (university cluster) and A100 (Lambda). No raw JSON/log files from this
hardware exist anywhere in this repo or in
`PayneA_HybridVQEStack_ThesisSubmission` (checked full git history — every
committed result there is `gpu=false`, CPU-only). The data is presumed lost.

The only surviving numbers are published in the thesis PDF:
- `sections/methodology.tex` §4.4.6 — experimental configuration
- `sections/results.tex` §4.7 (GPU-Accelerated Results) — wall-clock,
  strong/weak scaling, cuStateVec vs. Aer-thrust backend comparison

Treat those as a fixed, non-reproducible reference only. Per project
decision (2026-08-02): GTX 1650 is excluded from the published RTX 6000 vs.
A100 comparison going forward — it was the "got me started" preliminary
trial, not part of the final hardware comparison.
