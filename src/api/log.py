"""Dual-output logger: captures all stdout to both console and log file.

When init_log() is called, ALL print() output (from interface.py, problems.py, etc.)
is automatically duplicated to the log file. No code changes needed in other modules.
"""

import os
import sys
import time

_log_file = None
_start_time = None
_original_stdout = None


class _TeeStream:
    """Writes to both the original stdout and the log file."""
    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        if self._log_file is not None and data:
            self._log_file.write(data)
            self._log_file.flush()

    def flush(self):
        self._original.flush()
        if self._log_file is not None:
            self._log_file.flush()

    # Support any attribute access that stdout might need
    def __getattr__(self, name):
        return getattr(self._original, name)


def init_log(path: str):
    """Start logging: all subsequent print() output goes to both console and file."""
    global _log_file, _start_time, _original_stdout
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _log_file = open(path, "w")
    _start_time = time.time()

    # Redirect stdout through tee
    _original_stdout = sys.stdout
    sys.stdout = _TeeStream(_original_stdout, _log_file)

    log("Log started")


def log(msg: str, tag: str = ""):
    """Write a timestamped message. Also usable standalone, but not required —
    plain print() is captured too when init_log() has been called."""
    prefix = ""
    if _start_time is not None:
        elapsed = time.time() - _start_time
        prefix = f"[{elapsed:8.2f}s]"
    if tag:
        prefix = f"{prefix}[{tag}]"

    line = f"{prefix} {msg}" if prefix else msg
    print(line)


def close_log():
    """Stop logging and restore original stdout."""
    global _log_file, _original_stdout
    if _log_file is not None:
        log("Log closed")
        # Restore original stdout before closing
        if _original_stdout is not None:
            sys.stdout = _original_stdout
            _original_stdout = None
        _log_file.close()
        _log_file = None