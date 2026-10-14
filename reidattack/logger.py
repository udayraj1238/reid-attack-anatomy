"""Dual logger: writes to both stdout AND a persistent log file in /kaggle/working/.

This ensures progress is visible in the output artifacts even when Kaggle batch mode
suppresses real-time log streaming to the Logs tab.

Usage in any script:
    from reidattack.logger import log, setup_logging
    setup_logging("train_resnet50")  # creates /kaggle/working/log_train_resnet50.txt
    log("epoch 1/30 ...")           # prints AND writes to file with flush
"""
import sys
import os
import datetime

_log_file = None
_log_path = None


def setup_logging(tag="pipeline", log_dir=None):
    """Initialize file-based logging alongside stdout.
    
    Args:
        tag: Name tag for the log file (e.g. 'train_resnet50')
        log_dir: Directory for log files. Defaults to /kaggle/working/ if it exists,
                 otherwise current directory.
    """
    global _log_file, _log_path
    
    if log_dir is None:
        log_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    
    os.makedirs(log_dir, exist_ok=True)
    _log_path = os.path.join(log_dir, f"log_{tag}.txt")
    _log_file = open(_log_path, "a", encoding="utf-8")
    
    # Write header
    header = f"\n{'='*60}\nLog started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTag: {tag}\n{'='*60}\n"
    _log_file.write(header)
    _log_file.flush()
    print(header, end="", flush=True)
    
    return _log_path


def log(msg=""):
    """Print to stdout AND write to log file with immediate flush."""
    line = str(msg)
    print(line, flush=True)
    sys.stdout.flush()
    
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()
        os.fsync(_log_file.fileno())  # Force OS-level flush to disk


def get_log_path():
    """Return the current log file path."""
    return _log_path
