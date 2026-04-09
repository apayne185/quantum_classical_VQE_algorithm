# Structured JSON results persistence for  VQE runs 

import json
import os
import subprocess
from datetime import datetime


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_results(data: dict, backend: str, results_dir: str = "results") -> str:
    # Save run results as JSON, returns the file path.
    # Files are organized into subdirectories by backend type:

    # Map backend to subdirectory
    subdir_map = {
        "simulator": "simulator",
        "ibm_cloud": "ibm",
        "serial_baseline": "baseline",
    }
    subdir = subdir_map.get(backend, backend)
    out_dir = os.path.join(results_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{backend}_{ts}.json"
    path = os.path.join(out_dir, filename)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "backend": backend,
        "git_commit": _git_commit(),
        **data,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"[Results] Saved to {path}")
    return path
