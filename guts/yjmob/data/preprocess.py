"""Build the next-location (next-grid) dataset from raw YJMob CSV records.

Steps:
  1. Grid bounds and day range (infer_ranges); node id = (x - x_min) * y_size + (y - y_min).
  2. Per user, records in time order are cut into sessions at a day change or a gap
     longer than --session_gap time slots; consecutive repeated cells are merged and
     sessions shorter than --min_compressed_len cells are dropped.
  3. Each user's sessions are split chronologically 70/10/20 into train/val/test.
  4. One graph per day from the train sessions only (edges = cell transitions,
     weighted by count; six log1p node features).
  5. The train sessions grouped by day (train_trajs/): the traj hyperedges and
     their users read by the GUTS partitioner.

Outputs under <out_root>/processed/<out_name>/: sequences/{train,val,test}/part_*.pt,
daily_graphs/day_XXXX.pt, train_trajs/{meta.pkl,day_XXXX.pkl},
next_location_meta.{pt,json}, grid_info.pickle.

The --exclude_*_from options build the clean data for retraining after a deletion;
its train_trajs then hold only the remaining sessions. With --stream, one clean
dataset per step k (<out_name>/k001..) is built without the first k requests of a
stream forget set, as unlearn_stream.py reads them; the CSVs are read once and
finished steps are skipped.

Example (the YJMob-14d data used by configs/yjmob_14d.yaml):
    python guts/yjmob/data/preprocess.py --dataset_name dataset1 \
        --day_range 0 13 --out_name yjmob_1_14d
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.method.reporting import output_row
from guts.method.utils import ensure_dir, relpath_for_log, save_pickle


DEFAULT_RAW_DIR = os.path.join(_REPO_ROOT, "processed", "yjmob", "raw")
DEFAULT_OUT_ROOT = _REPO_ROOT
_NEXT_LOCATION_OUT_NAME = {"dataset1": "yjmob_1", "dataset2": "yjmob_2"}


@dataclass
class Session:
    """One trajectory session; `cells` is the compressed sequence, raw_* keep every record."""
    uid: int
    day: int
    start_time: int
    end_time: int
    cells: List[int]
    raw_cells: List[int]
    raw_times: List[int]


def clear_previous_build_outputs(out_dir: str) -> None:
    """Remove only generated files inside the selected dataset output directory."""
    for split in ("train", "val", "test"):
        split_dir = os.path.join(out_dir, "sequences", split)
        if os.path.isdir(split_dir):
            for name in os.listdir(split_dir):
                if name.startswith("part_") and name.endswith(".pt"):
                    os.remove(os.path.join(split_dir, name))

    graph_dir = os.path.join(out_dir, "daily_graphs")
    if os.path.isdir(graph_dir):
        for name in os.listdir(graph_dir):
            if name.startswith("day_") and name.endswith(".pt"):
                os.remove(os.path.join(graph_dir, name))

    for name in (
        "day_index_train.pt",
        "day_index_val.pt",
        "day_index_test.pt",
    ):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)


class SequencePartWriter:
    """Buffers sessions of one split and writes them as sequences/<split>/part_*.pt."""

    def __init__(self, out_dir: str, split: str, max_sessions_per_part: int) -> None:
        self.out_dir = ensure_dir(os.path.join(out_dir, "sequences", split))
        self.split = split
        self.max_sessions_per_part = int(max_sessions_per_part)
        self.buffer: List[Dict] = []
        self.parts: List[Dict] = []
        self.part_idx = 0
        self.session_count = 0
        self.step_count = 0

    def add(self, session: Session) -> None:
        obj = {
            "uid": int(session.uid),
            "day": int(session.day),
            "start_time": int(session.start_time),
            "end_time": int(session.end_time),
            "cells": [int(c) for c in session.cells],
        }
        self.buffer.append(obj)
        self.session_count += 1
        self.step_count += max(0, len(session.cells) - 1)
        if len(self.buffer) >= self.max_sessions_per_part:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        fname = f"part_{self.part_idx:05d}.pt"
        path = os.path.join(self.out_dir, fname)
        torch.save(self.buffer, path)
        steps = int(sum(max(0, len(s["cells"]) - 1) for s in self.buffer))
        self.parts.append(
            {
                "path": os.path.relpath(path, start=os.path.dirname(os.path.dirname(self.out_dir))),
                "sessions": int(len(self.buffer)),
                "steps": steps,
            }
        )
        self.buffer = []
        self.part_idx += 1

    def finish(self) -> Dict:
        self.flush()
        return {
            "split": self.split,
            "parts": self.parts,
            "sessions": int(self.session_count),
            "steps": int(self.step_count),
        }


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"{name} column not found. candidates={candidates}, actual={list(df.columns)}")


def _read_normalized_csv(
    path: str,
    chunksize: int,
    exclude_users: Optional[set] = None,
    day_range: Optional[Tuple[int, int]] = None,
) -> Iterable[pd.DataFrame]:
    """Read the CSV in chunks with columns renamed to uid/day/time/x/y, after the
    uid and day filters."""
    for chunk in pd.read_csv(path, chunksize=chunksize):
        uid_col = _resolve_column(chunk, ["uid", "user_id"], "uid")
        day_col = _resolve_column(chunk, ["d", "day", "date_id"], "day")
        time_col = _resolve_column(chunk, ["t", "time", "time_slot"], "time")
        x_col = _resolve_column(chunk, ["x", "origin_x", "grid_x"], "x")
        y_col = _resolve_column(chunk, ["y", "origin_y", "grid_y"], "y")
        out = chunk[[uid_col, day_col, time_col, x_col, y_col]].copy()
        out.columns = ["uid", "day", "time", "x", "y"]
        # int32 is enough for YJMob (uid < 1e5, day < 75, time < 48, x/y <= 200) and halves memory.
        out = out.astype({"uid": "int32", "day": "int32", "time": "int32", "x": "int32", "y": "int32"})
        if exclude_users:
            out = out[~out["uid"].isin(exclude_users)]
        if day_range is not None:
            lo, hi = day_range
            out = out[(out["day"] >= lo) & (out["day"] <= hi)]
        yield out


def infer_ranges(
    csv_paths: List[str],
    chunksize: int,
    exclude_users: Optional[set] = None,
    fixed_bounds: Optional[Dict] = None,
    day_range: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Grid bounds (or fixed_bounds, to keep node ids of an original build), day
    range, row and user counts."""
    x_min = y_min = day_min = None
    x_max = y_max = day_max = None
    rows = 0
    users = set()
    for path in csv_paths:
        for chunk in _read_normalized_csv(path, chunksize, exclude_users, day_range):
            if chunk.empty:
                continue
            rows += int(len(chunk))
            users.update(int(u) for u in chunk["uid"].unique().tolist())
            cd_min, cd_max = int(chunk["day"].min()), int(chunk["day"].max())
            day_min = cd_min if day_min is None else min(day_min, cd_min)
            day_max = cd_max if day_max is None else max(day_max, cd_max)
            if fixed_bounds is not None:
                continue
            cx_min, cx_max = int(chunk["x"].min()), int(chunk["x"].max())
            cy_min, cy_max = int(chunk["y"].min()), int(chunk["y"].max())
            x_min = cx_min if x_min is None else min(x_min, cx_min)
            x_max = cx_max if x_max is None else max(x_max, cx_max)
            y_min = cy_min if y_min is None else min(y_min, cy_min)
            y_max = cy_max if y_max is None else max(y_max, cy_max)
    if fixed_bounds is not None:
        x_min, x_max = fixed_bounds["x_min"], fixed_bounds["x_max"]
        y_min, y_max = fixed_bounds["y_min"], fixed_bounds["y_max"]
    if x_min is None:
        raise ValueError("No records found.")
    x_size = int(x_max - x_min + 1)
    y_size = int(y_max - y_min + 1)
    return {
        "rows": int(rows),
        "num_users": int(len(users)),
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "x_size": x_size,
        "y_size": y_size,
        "num_nodes": int(x_size * y_size),
        "day_min": int(day_min),
        "day_max": int(day_max),
        "days": list(range(int(day_min), int(day_max) + 1)),
    }


def node_ids_from_xy(df: pd.DataFrame, x_min: int, y_min: int, y_size: int) -> np.ndarray:
    return ((df["x"].to_numpy(dtype=np.int64) - x_min) * y_size + (df["y"].to_numpy(dtype=np.int64) - y_min)).astype(np.int64)


def compress_consecutive(raw_cells: List[int]) -> List[int]:
    """Merge consecutive repeated cells (A, A, B, A -> A, B, A)."""
    cells: List[int] = []
    prev: Optional[int] = None
    for c in raw_cells:
        c = int(c)
        if prev is None or c != prev:
            cells.append(c)
            prev = c
    return cells


def split_user_sessions(records: pd.DataFrame, session_gap: int, min_compressed_len: int) -> List[Session]:
    """Cut one user's records into sessions at a day change or a time gap above
    session_gap (<= 0 disables the gap rule); keep sessions with at least
    min_compressed_len cells after compression."""
    sessions: List[Session] = []
    if records.empty:
        return sessions
    records = records.sort_values(["day", "time"]).reset_index(drop=True)
    uid = int(records.loc[0, "uid"])
    cur_day: Optional[int] = None
    cur_cells: List[int] = []
    cur_times: List[int] = []

    def close_current() -> None:
        if not cur_cells:
            return
        cells = compress_consecutive(cur_cells)
        if len(cells) >= min_compressed_len:
            sessions.append(
                Session(
                    uid=uid,
                    day=int(cur_day),
                    start_time=int(cur_times[0]),
                    end_time=int(cur_times[-1]),
                    cells=cells,
                    raw_cells=list(cur_cells),
                    raw_times=list(cur_times),
                )
            )

    prev_time: Optional[int] = None
    for row in records.itertuples(index=False):
        day = int(row.day)
        time = int(row.time)
        cell = int(row.node_id)
        should_break = cur_day is None or day != cur_day
        if prev_time is not None and not should_break and session_gap > 0 and time - prev_time > session_gap:
            should_break = True
        if should_break:
            close_current()
            cur_day = day
            cur_cells = []
            cur_times = []
        cur_cells.append(cell)
        cur_times.append(time)
        prev_time = time
    close_current()
    return sessions


def split_name_for_session_index(idx: int, total: int) -> str:
    """Split of a user's idx-th session (chronological): first 70% train, next 10%
    val, rest test; at least one train session, and one test session when total >= 2."""
    if total <= 0:
        return "test"
    train_count = int(round(total * 0.7))
    val_count = int(round(total * 0.1))
    if train_count < 1:
        train_count = 1
    test_count = total - train_count - val_count
    if total >= 2 and test_count < 1:
        deficit = 1 - test_count
        take = min(deficit, train_count - 1)
        train_count -= take
        deficit -= take
        if deficit > 0:
            val_count = max(0, val_count - deficit)
    train_end = train_count
    val_end = train_count + val_count
    if idx < train_end:
        return "train"
    if idx < val_end:
        return "val"
    return "test"


def apply_deletion_to_session(
    sess: Session,
    min_compressed_len: int,
    exclude_node_ids: Optional[set] = None,
    exclude_edges: Optional[set] = None,
    exclude_trajs: Optional[set] = None,
) -> List[Session]:
    """Apply a deletion to one session whose split is already fixed: drop it (path),
    remove deleted cells (node), or cut it at deleted transitions (edge). The pieces
    are recompressed; short ones are dropped."""
    if exclude_trajs and (sess.uid, sess.day, sess.start_time, sess.end_time) in exclude_trajs:
        return []
    cells, times = list(sess.raw_cells), list(sess.raw_times)
    if exclude_node_ids:
        kept = [(c, t) for c, t in zip(cells, times) if int(c) not in exclude_node_ids]
        cells, times = [c for c, _ in kept], [t for _, t in kept]
    pieces: List[Tuple[List[int], List[int]]] = []
    cur_c: List[int] = []
    cur_t: List[int] = []
    for c, t in zip(cells, times):
        if exclude_edges and cur_c and (int(cur_c[-1]), int(c)) in exclude_edges:
            pieces.append((cur_c, cur_t))
            cur_c, cur_t = [], []
        cur_c.append(c)
        cur_t.append(t)
    if cur_c:
        pieces.append((cur_c, cur_t))
    out: List[Session] = []
    for pc, pt in pieces:
        compressed = compress_consecutive(pc)
        if len(compressed) >= min_compressed_len:
            out.append(Session(uid=sess.uid, day=sess.day, start_time=int(pt[0]), end_time=int(pt[-1]),
                               cells=compressed, raw_cells=list(pc), raw_times=list(pt)))
    return out


def add_train_session_to_graph(
    session: Session,
    visit_counts: np.ndarray,
    dwell_sums: np.ndarray,
    edge_counts_by_day: Dict[int, Counter],
) -> None:
    """Add a train session's raw visits, dwell time per cell run, and compressed
    transitions to its day's graph statistics."""
    day = int(session.day)
    for c in session.raw_cells:
        visit_counts[day, int(c)] += 1

    run_cell = session.raw_cells[0]
    run_start = session.raw_times[0]
    run_end = session.raw_times[0]
    for c, t in zip(session.raw_cells[1:], session.raw_times[1:]):
        if c == run_cell:
            run_end = t
            continue
        dwell_sums[day, int(run_cell)] += max(0.0, float(run_end - run_start))
        run_cell = c
        run_start = t
        run_end = t
    dwell_sums[day, int(run_cell)] += max(0.0, float(run_end - run_start))

    counter = edge_counts_by_day[day]
    for src, dst in zip(session.cells[:-1], session.cells[1:]):
        if int(src) != int(dst):
            counter[(int(src), int(dst))] += 1


def iter_user_records(
    csv_paths: List[str],
    ranges: Dict,
    chunksize: int,
    exclude_users: Optional[set] = None,
    day_range: Optional[Tuple[int, int]] = None,
) -> Iterable[pd.DataFrame]:
    """Yield each user's records (with node_id) in CSV order.

    The CSV must be ordered by uid: only the last uid of a chunk is carried into the
    next chunk, so a uid whose rows are not contiguous would be split in two; that
    stops with an error. The YJMob CSVs are ordered by uid."""
    seen_uids: set = set()

    def checked(user_df: pd.DataFrame) -> pd.DataFrame:
        uid = int(user_df["uid"].iloc[0])
        if uid in seen_uids:
            raise ValueError(
                f"uid {uid} appears again after its rows were processed: the CSV must be "
                "ordered by uid (sort it by uid, then day and time, and rerun)."
            )
        seen_uids.add(uid)
        return user_df

    carry = pd.DataFrame()
    for path in csv_paths:
        for chunk in _read_normalized_csv(path, chunksize, exclude_users, day_range):
            if chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["node_id"] = node_ids_from_xy(chunk, ranges["x_min"], ranges["y_min"], ranges["y_size"])
            if not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
                carry = pd.DataFrame()
            chunk.sort_values(["uid", "day", "time"], inplace=True, kind="mergesort")
            last_uid = int(chunk["uid"].iloc[-1])
            complete = chunk[chunk["uid"] != last_uid]
            carry = chunk[chunk["uid"] == last_uid].copy()
            for _, user_df in complete.groupby("uid", sort=False):
                if not user_df.empty:
                    yield checked(user_df)
    if not carry.empty:
        for _, user_df in carry.groupby("uid", sort=False):
            if not user_df.empty:
                yield checked(user_df)


def process_csvs(
    csv_paths: List[str],
    out_dir: str,
    ranges: Dict,
    chunksize: int,
    session_gap: int,
    min_compressed_len: int,
    max_sessions_per_part: int,
    exclude_users: Optional[set] = None,
    day_range: Optional[Tuple[int, int]] = None,
    exclude_node_ids: Optional[set] = None,
    exclude_edges: Optional[set] = None,
    exclude_trajs: Optional[set] = None,
) -> Dict:
    """Stream the CSVs and write the dataset (sequence parts, daily graphs, meta)."""
    user_sessions = (
        split_user_sessions(user_df, session_gap=session_gap, min_compressed_len=min_compressed_len)
        for user_df in iter_user_records(csv_paths, ranges, chunksize, exclude_users, day_range)
    )
    return write_dataset(out_dir, user_sessions, ranges, min_compressed_len, max_sessions_per_part,
                         exclude_users, exclude_node_ids, exclude_edges, exclude_trajs)


def write_dataset(
    out_dir: str,
    user_sessions: Iterable[List[Session]],
    ranges: Dict,
    min_compressed_len: int,
    max_sessions_per_part: int,
    exclude_users: Optional[set] = None,
    exclude_node_ids: Optional[set] = None,
    exclude_edges: Optional[set] = None,
    exclude_trajs: Optional[set] = None,
) -> Dict:
    """Write sequence parts, daily graphs, meta and train_trajs from each user's sessions
    (before deletion, in user order; users of exclude_users already left out). Splits are
    decided on these sessions, then node / edge / traj deletions are applied per session."""
    ensure_dir(out_dir)
    clear_previous_build_outputs(out_dir)
    writers = {
        split: SequencePartWriter(out_dir, split, max_sessions_per_part)
        for split in ("train", "val", "test")
    }

    day_count = int(ranges["day_max"]) + 1
    num_nodes = int(ranges["num_nodes"])
    visit_counts = np.zeros((day_count, num_nodes), dtype=np.float32)
    dwell_sums = np.zeros((day_count, num_nodes), dtype=np.float32)
    edge_counts_by_day: Dict[int, Counter] = defaultdict(Counter)
    split_session_counts = Counter()
    split_step_counts = Counter()
    user_count = 0
    deletion_mode = bool(exclude_node_ids or exclude_edges or exclude_trajs)

    for sessions in user_sessions:
        user_count += 1
        total = len(sessions)
        if deletion_mode:
            labeled = [(split_name_for_session_index(idx, total), piece)
                       for idx, sess in enumerate(sessions)
                       for piece in apply_deletion_to_session(sess, min_compressed_len, exclude_node_ids,
                                                              exclude_edges, exclude_trajs)]
        else:
            labeled = [(split_name_for_session_index(idx, total), sess) for idx, sess in enumerate(sessions)]
        for split, sess in labeled:
            writers[split].add(sess)
            split_session_counts[split] += 1
            split_step_counts[split] += max(0, len(sess.cells) - 1)
            if split == "train":
                add_train_session_to_graph(sess, visit_counts, dwell_sums, edge_counts_by_day)

    seq_index = {split: writer.finish() for split, writer in writers.items()}

    graph_dir = ensure_dir(os.path.join(out_dir, "daily_graphs"))
    # Node features (stored as log1p): raw visits, incoming/outgoing transition counts,
    # distinct predecessors/successors, and mean dwell time per visit.
    feature_names = ["visit_count", "in_count", "out_count", "unique_in", "unique_out", "dwell_time"]
    daily_index = []

    for day in ranges["days"]:
        day = int(day)
        edge_counter = edge_counts_by_day.get(day, Counter())
        feat = np.zeros((num_nodes, len(feature_names)), dtype=np.float32)
        feat[:, 0] = visit_counts[day]
        if edge_counter:
            src = np.fromiter((k[0] for k in edge_counter.keys()), dtype=np.int64)
            dst = np.fromiter((k[1] for k in edge_counter.keys()), dtype=np.int64)
            weight = np.fromiter(edge_counter.values(), dtype=np.float32)
            np.add.at(feat[:, 1], dst, weight)
            np.add.at(feat[:, 2], src, weight)
            for node, cnt in Counter(dst.tolist()).items():
                feat[int(node), 3] = float(cnt)
            for node, cnt in Counter(src.tolist()).items():
                feat[int(node), 4] = float(cnt)
            edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
            edge_weight = torch.from_numpy(weight).float()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        denom = np.maximum(visit_counts[day], 1.0)
        feat[:, 5] = dwell_sums[day] / denom
        x_raw = torch.from_numpy(feat).float()
        x = torch.log1p(x_raw)
        active = torch.nonzero(x_raw.sum(dim=1) > 0, as_tuple=False).view(-1).long()
        obj = {
            "day": day,
            "x": x,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "active_nodes": active,
            "feature_names": feature_names,
        }
        path = os.path.join(graph_dir, f"day_{day:04d}.pt")
        torch.save(obj, path)
        daily_index.append(
            {
                "day": day,
                "path": os.path.relpath(path, start=out_dir),
                "num_edges": int(edge_index.size(1)),
                "num_active_nodes": int(active.numel()),
            }
        )

    meta = {
        "task": "next_grid_prediction",
        "num_nodes": int(num_nodes),
        "grid_info": {
            k: ranges[k]
            for k in ("x_min", "x_max", "y_min", "y_max", "x_size", "y_size")
        },
        "days": ranges["days"],
        "day_min": int(ranges["day_min"]),
        "day_max": int(ranges["day_max"]),
        "feature_names": feature_names,
        "split_policy": "per-user chronological sessions, 70/10/20",
        "deletion_split_policy": ("split decided on pre-deletion sessions; deletion applied per session "
                                  "(original split preserved)") if (exclude_node_ids or exclude_edges or exclude_trajs) else None,
        "graph_policy": "day-wise train split transitions and train split raw visits only",
        "sequence_index": seq_index,
        "daily_graphs": daily_index,
        "counts": {
            "raw_rows": int(ranges["rows"]),
            "users_seen": int(user_count),
            "split_sessions": {k: int(v) for k, v in split_session_counts.items()},
            "split_steps": {k: int(v) for k, v in split_step_counts.items()},
        },
        "excluded_users_count": int(len(exclude_users)) if exclude_users else 0,
        "excluded_nodes_count": int(len(exclude_node_ids)) if exclude_node_ids else 0,
        "excluded_edges_count": int(len(exclude_edges)) if exclude_edges else 0,
        "excluded_trajs_count": int(len(exclude_trajs)) if exclude_trajs else 0,
    }
    torch.save(meta, os.path.join(out_dir, "next_location_meta.pt"))
    with open(os.path.join(out_dir, "next_location_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    save_pickle(meta["grid_info"], os.path.join(out_dir, "grid_info.pickle"))
    write_train_trajs(out_dir)
    return meta


TRAIN_TRAJS_DIR = "train_trajs"


def write_train_trajs(out_dir: str) -> Dict:
    """Group the train sessions (sequences/train) by day into train_trajs/day_XXXX.pkl,
    each a list of {"uid", "nodes"} in session order, plus train_trajs/meta.pkl with the
    day files relative to that directory. The partitioner turns each session into one
    hyperedge, so the hyperedges always match the training data, including clean builds."""
    meta = torch.load(os.path.join(out_dir, "next_location_meta.pt"), map_location="cpu")
    by_day: Dict[int, List[Dict]] = defaultdict(list)
    for part in meta["sequence_index"]["train"]["parts"]:
        for sess in torch.load(os.path.join(out_dir, part["path"]), map_location="cpu"):
            by_day[int(sess["day"])].append({"uid": int(sess["uid"]), "nodes": list(sess["cells"])})

    trajs_dir = ensure_dir(os.path.join(out_dir, TRAIN_TRAJS_DIR))
    day_files: Dict[int, str] = {}
    for day in sorted(by_day):
        name = f"day_{day:04d}.pkl"
        save_pickle(by_day[day], os.path.join(trajs_dir, name))
        day_files[day] = name
    trajs_meta = {
        "day_files": day_files,
        "num_trajs": int(sum(len(v) for v in by_day.values())),
    }
    save_pickle(trajs_meta, os.path.join(trajs_dir, "meta.pkl"))
    return trajs_meta


def _load_records(path: str) -> list:
    with open(path) as f:
        return json.load(f)["records"]


def _exclusions(records: Dict[str, list]) -> Dict[str, Optional[set]]:
    """Deletion targets of each forget-set type (None when the type is not given)."""
    return {
        "users": {int(r["uid"]) for r in records["users"]} if "users" in records else None,
        "nodes": {int(r["nid"]) for r in records["nodes"]} if "nodes" in records else None,
        "edges": {(int(r["src"]), int(r["dst"])) for r in records["edges"]} if "edges" in records else None,
        "trajs": ({(int(r["uid"]), int(r["day"]), int(r["start_time"]), int(r["end_time"])) for r in records["trajs"]}
                  if "trajs" in records else None),
    }


def _counts_line(meta: dict) -> str:
    c = meta["counts"]
    sess = c["split_sessions"]
    return (f"users {c['users_seen']:,}   sessions  train {sess['train']:,}   "
            f"val {sess['val']:,}   test {sess['test']:,}")


def _scan_line(ranges: Dict) -> str:
    return (f"{ranges['rows']:,} rows   {ranges['num_users']:,} users   days {ranges['day_min']}-{ranges['day_max']}"
            f"   grid {ranges['x_size']} x {ranges['y_size']}")


def build_stream(csv_paths: List[str], out_dir: str, key: str, records: list, fixed_bounds: Dict,
                 day_range: Optional[Tuple[int, int]], args: argparse.Namespace) -> None:
    """Build <out_dir>/k001..kNNN, step k without the first k records, reading the CSVs once.

    Every user's sessions are kept in memory with its row count and day range; each step
    drops the deleted users and recomputes the row, user and day counts from them, so a
    step equals a separate build with the first k records. Finished steps are skipped."""
    todo = []
    for k in range(1, len(records) + 1):
        step_dir = os.path.join(out_dir, f"k{k:03d}")
        if not os.path.exists(os.path.join(step_dir, "next_location_meta.pt")):
            todo.append((k, step_dir))
    if len(todo) < len(records):
        print(output_row("stream", f"{len(records) - len(todo)}/{len(records)} steps already built"), flush=True)
    if not todo:
        return

    t0 = time.time()
    base_ranges = infer_ranges(csv_paths, chunksize=args.chunksize, fixed_bounds=fixed_bounds, day_range=day_range)
    users = []  # (uid, rows, first day, last day, sessions)
    for user_df in iter_user_records(csv_paths, base_ranges, args.chunksize, None, day_range):
        users.append((int(user_df["uid"].iloc[0]), len(user_df), int(user_df["day"].min()), int(user_df["day"].max()),
                      split_user_sessions(user_df, session_gap=args.session_gap, min_compressed_len=args.min_compressed_len)))
    print(output_row("scan", f"{_scan_line(base_ranges)}   ({time.time() - t0:.0f}s)"), flush=True)
    print(output_row("stream", f"building {len(todo)} steps (step k: first k {key} removed)"), flush=True)

    for k, step_dir in todo:
        excl = _exclusions({key: records[:k]})
        kept = [u for u in users if not excl["users"] or u[0] not in excl["users"]]
        day_min, day_max = min(u[2] for u in kept), max(u[3] for u in kept)
        ranges = dict(base_ranges, rows=sum(u[1] for u in kept), num_users=len(kept),
                      day_min=day_min, day_max=day_max, days=list(range(day_min, day_max + 1)))
        meta = write_dataset(step_dir, (u[4] for u in kept), ranges, args.min_compressed_len, args.max_sessions_per_part,
                             excl["users"], excl["nodes"], excl["edges"], excl["trajs"])
        print(output_row(f"k{k:03d}", _counts_line(meta)), flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the YJMob next-location dataset.")
    p.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR)
    p.add_argument("--dataset_name", type=str, default="dataset1")
    p.add_argument("--csv_files", nargs="+", default=None)
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    p.add_argument("--out_name", type=str, default="")
    p.add_argument("--chunksize", type=int, default=1_000_000)
    p.add_argument("--session_gap", type=int, default=6, help="session break gap in time slots (<=0: off)")
    p.add_argument("--min_compressed_len", type=int, default=3)
    p.add_argument("--max_sessions_per_part", type=int, default=50_000)
    p.add_argument("--day_range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                   help="keep days LO..HI")
    p.add_argument("--fixed_grid_info_from", type=str, default="",
                   help="original next_location_meta.pt; keeps node ids (clean builds)")
    p.add_argument("--exclude_users_from", type=str, default="", help="forget-set JSON: drop users")
    p.add_argument("--exclude_nodes_from", type=str, default="",
                   help="forget-set JSON: drop cells (needs --fixed_grid_info_from)")
    p.add_argument("--exclude_edges_from", type=str, default="", help="forget-set JSON: cut transitions")
    p.add_argument("--exclude_trajs_from", type=str, default="", help="forget-set JSON: drop trajs (sessions)")
    p.add_argument("--stream", action="store_true",
                   help="stream clean data: <out_name>/k001.. without the first k requests of the forget set")
    p.add_argument("--train_trajs_only", action="store_true",
                   help="only (re)write train_trajs/ of an existing build")
    return p.parse_args(argv)


def main(argv=None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)
    out_name = args.out_name or _NEXT_LOCATION_OUT_NAME.get(args.dataset_name, f"{args.dataset_name}_Next_location")
    out_dir = os.path.join(args.out_root, "processed", out_name)
    if args.train_trajs_only:
        trajs_meta = write_train_trajs(out_dir)
        print(output_row("done", f"{trajs_meta['num_trajs']:,} train trajs -> {relpath_for_log(os.path.join(out_dir, TRAIN_TRAJS_DIR))}"),
              flush=True)
        return

    if args.csv_files is None:
        if args.dataset_name == "dataset1":
            csv_files = ["yjmob100k-dataset1.csv"]
        elif args.dataset_name == "dataset2":
            csv_files = ["yjmob100k-dataset2.csv"]
        else:
            raise ValueError("Use --csv_files for custom dataset_name.")
    else:
        csv_files = args.csv_files
    csv_paths = [f if os.path.isabs(f) else os.path.join(args.raw_dir, f) for f in csv_files]
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    fixed_bounds: Optional[Dict] = None
    if args.fixed_grid_info_from:
        orig_meta = torch.load(args.fixed_grid_info_from, map_location="cpu")
        gi = orig_meta["grid_info"]
        fixed_bounds = {
            "x_min": gi["x_min"], "x_max": gi["x_max"],
            "y_min": gi["y_min"], "y_max": gi["y_max"],
        }
    if args.exclude_nodes_from and not fixed_bounds:
        raise ValueError("--exclude_nodes_from requires --fixed_grid_info_from (node ids must match the original grid mapping).")

    day_range: Optional[Tuple[int, int]] = None
    if args.day_range is not None:
        day_range = (int(args.day_range[0]), int(args.day_range[1]))

    forget_sets = {key: _load_records(getattr(args, f"exclude_{key}_from"))
                   for key in ("users", "nodes", "edges", "trajs") if getattr(args, f"exclude_{key}_from")}
    print(f"preprocess  {args.dataset_name} -> {relpath_for_log(out_dir)}", flush=True)
    options = ([f"days {day_range[0]}-{day_range[1]}"] if day_range else []) \
        + [f"exclude {k} {len(v):,}" for k, v in forget_sets.items()] \
        + ([f"grid from {relpath_for_log(args.fixed_grid_info_from)}"] if fixed_bounds else []) \
        + (["stream"] if args.stream else [])
    if options:
        print(output_row("options", "   ".join(options)), flush=True)

    def build(build_dir: str, records: Dict[str, list]) -> None:
        exclusions = _exclusions(records)
        t0 = time.time()
        ranges = infer_ranges(csv_paths, chunksize=args.chunksize, exclude_users=exclusions["users"],
                              fixed_bounds=fixed_bounds, day_range=day_range)
        print(output_row("scan", f"{_scan_line(ranges)}   ({time.time() - t0:.0f}s)"), flush=True)
        meta = process_csvs(
            csv_paths=csv_paths,
            out_dir=ensure_dir(build_dir),
            ranges=ranges,
            chunksize=args.chunksize,
            session_gap=args.session_gap,
            min_compressed_len=args.min_compressed_len,
            max_sessions_per_part=args.max_sessions_per_part,
            exclude_users=exclusions["users"],
            day_range=day_range,
            exclude_node_ids=exclusions["nodes"],
            exclude_edges=exclusions["edges"],
            exclude_trajs=exclusions["trajs"],
        )
        print(output_row("done", f"{_counts_line(meta)}   ({time.time() - t0:.0f}s)"), flush=True)

    if not args.stream:
        build(out_dir, forget_sets)
        return

    # Stream clean data: <out_name>/k001..kNNN, step k without the first k requests of the forget set.
    if len(forget_sets) != 1:
        raise ValueError("--stream needs exactly one --exclude_*_from (the stream forget set)")
    if not fixed_bounds:
        raise ValueError("--stream needs --fixed_grid_info_from (every step keeps the original node ids)")
    (key, records), = forget_sets.items()
    build_stream(csv_paths, out_dir, key, records, fixed_bounds, day_range, args)

if __name__ == "__main__":
    main()
