"""Thin dual-output logger: writes to both console and file."""

import os
import time

_log_file = None
_start_time = None


def init_log(path: str):
    global _log_file, _start_time
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _log_file = open(path, "w")
    _start_time = time.time()
    log("Log started")


def log(msg: str, tag: str = ""):
    prefix = ""
    if _start_time is not None:
        elapsed = time.time() - _start_time
        prefix = f"[{elapsed:8.2f}s]"
    if tag:
        prefix = f"{prefix}[{tag}]"

    line = f"{prefix} {msg}" if prefix else msg
    print(line)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def close_log():
    global _log_file
    if _log_file is not None:
        log("Log closed")
        _log_file.close()
        _log_file = None
