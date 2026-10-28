"""Write one processed directory per shard for guts/yjmob/model/train_single.py.

For every shard the output mirrors a preprocess.py build:
  shard_<k>/daily_graphs/day_XXXX.pt : the shard's subgraph of that day. x has one row
                                       per active node, in the order of active_nodes
                                       (sorted global cell ids); edge_index uses global ids.
  shard_<k>/sequences/<split>/part_*.pt: the session segments that start inside the shard.
  shard_<k>/next_location_meta.pt     : the source meta with this shard's index.
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from guts.method.utils import ensure_dir, repo_relpath


def build_edge_weight_map(sample) -> Dict[Tuple[int, int], float] | None:
    """(src, dst) -> transition count of the day graph, or None without weights."""
    if sample is None or sample.edge_weight is None or sample.edge_weight.numel() == 0:
        return None
    src_list = sample.edge_index[0].tolist()
    dst_list = sample.edge_index[1].tolist()
    w_list = sample.edge_weight.tolist()
    return dict(zip(zip(src_list, dst_list), w_list))


def shard_graph_to_global(
    shard,
    num_nodes: int,
    day: int,
    edge_weight_map: Dict[Tuple[int, int], float] | None = None,
) -> dict:
    """A shard's day subgraph (local node ids) in global node ids, as saved on disk.

    Only the shard's nodes get a row of x (ordered like active_nodes); num_nodes is the
    grid size the global ids index into.
    """
    uids = shard.uids.long()
    order = torch.argsort(uids)
    active_nodes = uids[order]
    if active_nodes.numel() and int(active_nodes[-1]) >= int(num_nodes):
        raise ValueError(f"node id {int(active_nodes[-1])} out of range for num_nodes={num_nodes}")
    x = shard.x[order]
    edge_index = uids[shard.edge_index]

    if edge_weight_map is not None:
        edge_weight = torch.tensor(
            [edge_weight_map.get((int(s), int(d)), 1.0)
             for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist())],
            dtype=torch.float32,
        )
    else:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    return {
        "day": int(day),
        "x": x,
        # int32 on disk (node ids < 40,000); readers upcast to int64 before indexing.
        "edge_index": edge_index.contiguous().to(torch.int32),
        "edge_weight": edge_weight,
        "active_nodes": active_nodes.to(torch.int32),
    }


def split_session_by_shard(cells: List[int], shard_of: Dict[int, int]) -> List[Tuple[int, List[int]]]:
    """Cut a session into (shard, segment) pairs: maximal runs of consecutive transitions
    whose source cells lie in the same shard. A run's last transition may leave the shard
    (its target is predicted); transitions from unassigned cells are skipped."""
    segments: List[Tuple[int, List[int]]] = []
    run_shard = None
    current: List[int] = []
    for src, dst in zip(cells[:-1], cells[1:]):
        shard = shard_of.get(int(src))
        if shard != run_shard or shard is None:
            if current:
                segments.append((run_shard, current))
            run_shard, current = shard, ([int(src)] if shard is not None else [])
        if shard is not None:
            current.append(int(dst))
    if current:
        segments.append((run_shard, current))
    return segments


class _ShardSequenceWriter:
    """Buffers per-(shard, split) sessions and writes them as sequences/<split>/part_*.pt."""

    def __init__(self, out_dir: str, max_sessions_per_part: int) -> None:
        self.out_dir = out_dir
        self.max_sessions_per_part = int(max_sessions_per_part)
        self.buffers: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
        self.part_idx: Dict[Tuple[int, str], int] = defaultdict(int)
        self.parts: Dict[Tuple[int, str], List[dict]] = defaultdict(list)

    def add(self, shard_id: int, split: str, sess: dict) -> None:
        key = (shard_id, split)
        self.buffers[key].append(sess)
        if len(self.buffers[key]) >= self.max_sessions_per_part:
            self._flush(key)

    def _flush(self, key: Tuple[int, str]) -> None:
        buffer = self.buffers[key]
        if not buffer:
            return
        shard_id, split = key
        name = f"part_{self.part_idx[key]:05d}.pt"
        torch.save(buffer, os.path.join(self.out_dir, f"shard_{shard_id}", "sequences", split, name))
        self.parts[key].append({
            "path": f"sequences/{split}/{name}",
            "sessions": len(buffer),
            "steps": int(sum(max(0, len(sess["cells"]) - 1) for sess in buffer)),
        })
        self.buffers[key] = []
        self.part_idx[key] += 1

    def finish(self, shard_ids: List[int], splits: List[str]) -> Dict[int, dict]:
        """Flush the rest; returns {shard: {split: {parts, sessions, steps}}}."""
        for key in list(self.buffers):
            self._flush(key)
        return {
            shard_id: {
                split: {
                    "parts": self.parts[(shard_id, split)],
                    "sessions": sum(p["sessions"] for p in self.parts[(shard_id, split)]),
                    "steps": sum(p["steps"] for p in self.parts[(shard_id, split)]),
                }
                for split in splits
            }
            for shard_id in shard_ids
        }


def _iter_source_sessions(source_dir: str, parts: List[dict], days: set):
    """Yield the source sessions of `days` from the given sequence parts."""
    for part in parts:
        for sess in torch.load(os.path.join(source_dir, part["path"]), map_location="cpu"):
            if int(sess["day"]) in days:
                yield sess


def build_sequences(
    source_dir: str,
    out_dir: str,
    source_meta: dict,
    splits: List[str],
    days: set,
    by_day: dict,
    shard_ids: List[int],
    max_sessions_per_part: int,
    max_source_parts: int,
    log_fn=None,
) -> Dict[int, dict]:
    """Split every source session of `days` into per-shard segments (by that day's
    partition) and write the segments of `shard_ids` as sequence parts; returns their
    sequence index. Sessions are always split with the full partition, so writing a
    subset of shards gives exactly those shards' part of a full build."""
    log = log_fn if log_fn is not None else (lambda msg: print(msg, flush=True))
    # Each node belongs to exactly one shard per day.
    shard_of_by_day: Dict[int, Dict[int, int]] = {
        day: {int(n): shard_id for shard_id, shard in by_day[day]["shards"].items() for n in shard.uids.tolist()}
        for day in days
    }
    for shard_id in shard_ids:
        for split in splits:
            ensure_dir(os.path.join(out_dir, f"shard_{shard_id}", "sequences", split))

    shard_set = set(shard_ids)
    writer = _ShardSequenceWriter(out_dir, max_sessions_per_part)
    done = []
    for split in splits:
        parts = source_meta["sequence_index"][split]["parts"]
        if max_source_parts > 0:
            parts = parts[:max_source_parts]
        n_sessions = 0
        for sess in _iter_source_sessions(source_dir, parts, days):
            n_sessions += 1
            day = int(sess["day"])
            segment_count: Dict[int, int] = defaultdict(int)
            for shard_id, segment in split_session_by_shard(sess["cells"], shard_of_by_day[day]):
                if shard_id not in shard_set:
                    segment_count[shard_id] += 1
                    continue
                writer.add(shard_id, split, {
                    "uid": int(sess["uid"]),
                    "day": day,
                    "start_time": int(sess.get("start_time", -1)),
                    "end_time": int(sess.get("end_time", -1)),
                    "source_segment": segment_count[shard_id],
                    "cells": segment,
                })
                segment_count[shard_id] += 1
        done.append(f"{split} {n_sessions:,}")
    log(f"  sessions split  {'  '.join(done)}  -> {len(shard_ids)} shards written")
    return writer.finish(shard_ids, splits)


def write_outputs(out_dir, source_processed_dir, source_meta, by_day, shard_ids, seq_index):
    """Write each shard's daily graphs and next_location_meta.pt."""
    ensure_dir(out_dir)
    num_nodes = int(source_meta["num_nodes"])
    feature_names = list(source_meta["feature_names"])
    day_edge_weight_maps = {
        day: build_edge_weight_map(data.get("sample")) for day, data in by_day.items()
    }
    for shard_id in shard_ids:
        shard_dir = ensure_dir(os.path.join(out_dir, f"shard_{shard_id}"))
        daily_dir = ensure_dir(os.path.join(shard_dir, "daily_graphs"))
        daily_graph_entries = []
        for day, data in sorted(by_day.items()):
            graph = shard_graph_to_global(
                data["shards"][shard_id], num_nodes, day, edge_weight_map=day_edge_weight_maps[day],
            )
            torch.save(graph, os.path.join(daily_dir, f"day_{day:04d}.pt"))
            daily_graph_entries.append({
                "day": int(day),
                "path": f"daily_graphs/day_{day:04d}.pt",
                "num_edges": int(graph["edge_index"].size(1)),
                "num_active_nodes": int(graph["active_nodes"].numel()),
            })

        meta = copy.deepcopy(source_meta)
        meta["task"] = "sharded_next_grid_prediction"
        meta["node_feat_dim"] = len(feature_names)
        meta["days"] = sorted(int(day) for day in by_day)
        meta["day_min"] = min(meta["days"]) if meta["days"] else 0
        meta["day_max"] = max(meta["days"]) if meta["days"] else 0
        meta["sequence_index"] = seq_index[shard_id]
        meta["source_processed_dir"] = source_processed_dir
        meta["daily_graphs"] = daily_graph_entries
        torch.save(meta, os.path.join(shard_dir, "next_location_meta.pt"))


def materialize_graphsage_shards(
    source_processed_dir: str,
    out_dir: str,
    days: List[int],
    by_day: dict,
    splits: List[str] | None = None,
    max_sessions_per_part: int = 50000,
    max_source_parts: int = 0,
    log_fn=None,
    shard_ids: List[int] | None = None,
):
    """Write sharded processed dirs for guts/yjmob/model/train_single.py.

    ``by_day`` must be ``{day: {"shards": shards, "sample": sample}}`` from the
    pipeline's partition step, so partitioning is not repeated here. ``shard_ids``
    limits the output to those shards (default: all); the unlearning rebuild uses it.
    """
    if splits is None:
        splits = ["train", "val"]
    source_meta = torch.load(os.path.join(source_processed_dir, "next_location_meta.pt"), map_location="cpu")
    days = sorted(set(int(day) for day in days))
    if shard_ids is None:
        shard_ids = sorted(next(iter(by_day.values()))["shards"])
    shard_ids = sorted(int(k) for k in shard_ids)
    seq_index = build_sequences(
        source_dir=source_processed_dir,
        out_dir=out_dir,
        source_meta=source_meta,
        splits=list(splits),
        days=set(days),
        by_day=by_day,
        shard_ids=shard_ids,
        max_sessions_per_part=max_sessions_per_part,
        max_source_parts=max_source_parts,
        log_fn=log_fn,
    )
    write_outputs(out_dir, source_processed_dir, source_meta, by_day, shard_ids, seq_index)
    return {
        "out_dir": repo_relpath(out_dir),
        "num_shards": len(shard_ids),
        "shards": {
            int(shard_id): {
                "num_nodes": int(source_meta["num_nodes"]),
                "steps": {split: int(seq_index[shard_id][split]["steps"]) for split in splits},
                "sessions": {split: int(seq_index[shard_id][split]["sessions"]) for split in splits},
            }
            for shard_id in shard_ids
        },
    }


def estimate_shard_weight(processed_dir: str) -> int:
    """Training load of a shard for GPU balancing: its day-graph edges (else train steps or
    sessions); 1 when the shard has no meta yet."""
    meta_path = os.path.join(processed_dir, "next_location_meta.pt")
    if not os.path.exists(meta_path):
        return 1
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
    return 1
