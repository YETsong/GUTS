"""Small helpers shared by the dataset code: paths and logs, run ids, seeds, pickles."""

import json
import os
import pickle
import random
import re
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def repo_relpath(path: str) -> str:
    """`path` relative to the repository root (paths outside the repository stay absolute).
    Used for log lines and for every path stored in a result file; read back with resolve_repo_path."""
    if not path:
        return path
    full = os.path.abspath(path)
    try:
        rel = os.path.relpath(full, _REPO_ROOT)
    except ValueError:
        return full
    return full if rel == ".." or rel.startswith(".." + os.sep) else rel


def relpath_for_log(path: str) -> str:
    """`path` for a log line (see repo_relpath)."""
    return repo_relpath(path)


def resolve_repo_path(path: str) -> str:
    """Absolute path of a stored path: relative paths are relative to the repository root."""
    if not path:
        return path
    return os.path.abspath(path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path))


def timestamp_id() -> str:
    return datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")


def random_seed() -> int:
    """A fresh 31-bit seed from the OS entropy pool (valid for random, numpy and torch)."""
    return int.from_bytes(os.urandom(4), byteorder="big") & 0x7FFFFFFF


def resolve_seed(seed: Optional[int]) -> int:
    """`seed` when given, otherwise a fresh random seed (the caller records it)."""
    return random_seed() if seed is None else int(seed)


def seed_mismatch_warning(data_path: str, seed: int) -> Optional[str]:
    """A warning when the data path's file or folder name carries `seed<N>` with N != seed.

    Only a check: the seed always comes from options/config, never from a name."""
    name = os.path.normpath(data_path or "")
    for part in (os.path.basename(name), os.path.basename(os.path.dirname(name))):
        found = re.findall(r"seed(\d+)", part)
        if found:
            if int(found[-1]) != int(seed):
                return (f"[WARN] seed {seed} differs from seed{found[-1]} in the data path name "
                        f"({data_path}); pass the matching seed if this data was built for it")
            return None
    return None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_pickle(obj, path: str):
    # Write to a temporary file and rename, so an interrupted save never leaves a
    # truncated file at `path`.
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def append_log(path: str, message: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def append_log_synced(path: str, message: str) -> None:
    """Append a line and fsync, so logs survive a crash of a parallel worker."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
        f.flush()
        os.fsync(f.fileno())


def timestamp_line(message: str) -> str:
    return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"


def run_logger(run_log_path: str):
    """log(msg): print a timestamped line and append it to the run log (run_log.txt).
    Scripts print only progress, results, warnings and errors; details go to their own files."""
    def log(message: str) -> None:
        line = timestamp_line(message)
        print(line, flush=True)
        append_log_synced(run_log_path, line)
    return log


def compact_json(obj) -> str:
    """json.dumps with indent=2, but pure-number arrays stay on one line."""
    raw = json.dumps(obj, indent=2, default=str)
    return re.sub(
        r'\[\n(\s+-?\d+(?:\.\d+)?,?\n)+\s*\]',
        lambda m: '[' + ', '.join(re.findall(r'-?\d+(?:\.\d+)?', m.group(0))) + ']',
        raw,
    )
