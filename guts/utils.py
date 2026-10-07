# python -c "from guts.utils import compute_binary_metrics; import numpy as np; print(compute_binary_metrics(np.array([0,1]), np.array([0.2,0.8])))"

import os
import random
import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def find_latest_run(root: str, pattern: str) -> Optional[str]:
    import glob as _glob

    candidates = [p for p in _glob.glob(os.path.join(root, pattern)) if os.path.isdir(p)]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return os.path.abspath(candidates[0])


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def relpath_for_log(path: str) -> str:
    if not path:
        return path
    try:
        return os.path.relpath(os.path.abspath(path), _PROJECT_ROOT)
    except ValueError:
        return path


def timestamp_id() -> str:
    return datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    metrics = {
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    unique = np.unique(y_true)
    if unique.size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return metrics


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


def estimate_shard_weight(processed_dir: str) -> int:
    meta_path = os.path.join(processed_dir, "next_location_meta.pt")
    if not os.path.exists(meta_path):
        return 1
    try:
        meta = torch.load(meta_path, map_location="cpu")
        daily_graphs = meta.get("daily_graphs")
        if daily_graphs:
            total_edges = sum(int(entry.get("num_edges", 0)) for entry in daily_graphs)
            if total_edges:
                return int(total_edges)
        counts = meta.get("counts", {})
        steps = counts.get("split_steps", {}).get("train")
        if steps:
            return int(steps)
        sessions = counts.get("split_sessions", {}).get("train")
        if sessions:
            return int(sessions)
    except Exception:
        pass
    return 1


def balanced_lanes(weighted_items, num_lanes: int) -> list:
    lanes = [[] for _ in range(num_lanes)]
    loads = [0] * num_lanes
    for item, weight in sorted(weighted_items, key=lambda x: x[1], reverse=True):
        idx = loads.index(min(loads))
        lanes[idx].append(item)
        loads[idx] += weight
    return lanes


def append_log_synced(path: str, message: str) -> None:
    """Append a line and fsync, so logs survive a crash of a parallel worker."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
        f.flush()
        os.fsync(f.fileno())


def compact_json(obj) -> str:
    """json.dumps with indent=2, but pure-number arrays stay on one line."""
    import json
    import re

    raw = json.dumps(obj, indent=2, default=str)
    return re.sub(
        r'\[\n(\s+-?\d+(?:\.\d+)?,?\n)+\s*\]',
        lambda m: '[' + ', '.join(re.findall(r'-?\d+(?:\.\d+)?', m.group(0))) + ']',
        raw,
    )
