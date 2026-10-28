"""Build yjmob forget sets (node / edge / user / traj) from the original train sessions.

Each train session is one traj. Targets are drawn stratified by a per-type measure
(node out-degree, edge weight, user traj count, traj length) with the stratum bounds
read from a strata file:

  strata  compute quartile bounds of each measure on a dataset and save the strata file
  batch   one forget set per percentage:  <out_root>/<Type>/forget_<type>_seed<S>_<P>pct.json
  stream  n targets in shuffled order for unlearn_stream.py:  .../forget_<type>_seed<S>_n<N>_stream.json

Unlearning and the clean data build read only the target keys of each record
(uid / nid / src, dst / uid, day, start_time, end_time); the other fields describe the target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.method.forget_sampling import NUM_STRATA, quartile_bounds, sample_from_strata, stratum_labels, stratum_of  # noqa: E402
from guts.method.reporting import output_row  # noqa: E402
from guts.method.utils import relpath_for_log, repo_relpath, resolve_seed  # noqa: E402

DIR_NAMES = {"node": "Node", "edge": "Edge", "user": "User", "traj": "Traj"}
STREAM_TYPES = ("user", "node", "traj")  # types unlearn_stream.py supports


# ─────────────────────────────────────────────────────────────────────────────
# Train session index
# ─────────────────────────────────────────────────────────────────────────────

class TrainSessionIndex:
    """Users, cells (nodes) and transitions (edges) mapped to the trajs containing them.

    Traj ids follow the part file order and the session order inside each part.
    """

    def __init__(self, source_dir: str) -> None:
        train_dir = os.path.join(source_dir, "sequences", "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"{relpath_for_log(train_dir)} not found — source_dir must be a preprocessed dataset")
        self.traj_meta: Dict[int, dict] = {}
        user_to_trajs: Dict[int, List[int]] = defaultdict(list)
        node_to_trajs: Dict[int, List[int]] = defaultdict(list)
        edge_to_trajs: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        traj_id = 0
        for fname in sorted(f for f in os.listdir(train_dir) if f.endswith(".pt")):
            for sess in torch.load(os.path.join(train_dir, fname), map_location="cpu"):
                uid = int(sess["uid"])
                cells = [int(c) for c in sess["cells"]]
                self.traj_meta[traj_id] = {
                    "uid": uid, "day": int(sess["day"]),
                    "start_time": int(sess["start_time"]), "end_time": int(sess["end_time"]),
                    "cells": cells,
                }
                user_to_trajs[uid].append(traj_id)
                for c in set(cells):
                    node_to_trajs[c].append(traj_id)
                for e in set(zip(cells[:-1], cells[1:])):
                    edge_to_trajs[e].append(traj_id)
                traj_id += 1
        self.user_to_trajs = dict(user_to_trajs)
        self.node_to_trajs = dict(node_to_trajs)
        self.edge_to_trajs = dict(edge_to_trajs)
        print(output_row("index", f"{len(self.traj_meta):,} train trajs   {len(self.user_to_trajs):,} users   "
                         f"{len(self.node_to_trajs):,} nodes   {len(self.edge_to_trajs):,} edges"), flush=True)

    def traj_edges(self, traj_id: int) -> List[Tuple[int, int]]:
        """Transitions of a traj in order (repeats kept)."""
        cells = self.traj_meta[traj_id]["cells"]
        return list(zip(cells[:-1], cells[1:]))

    def step_count(self, traj_ids: List[int]) -> int:
        return sum(len(self.traj_meta[t]["cells"]) - 1 for t in traj_ids)

    def node_out_degree(self) -> Dict[int, int]:
        """Transitions leaving each cell, over all trajs."""
        degree: Dict[int, int] = defaultdict(int)
        for t in self.traj_meta:
            for src, _ in self.traj_edges(t):
                degree[src] += 1
        return dict(degree)

    def edge_weight(self) -> Dict[Tuple[int, int], int]:
        """Occurrences of each transition, over all trajs."""
        weight: Dict[Tuple[int, int], int] = defaultdict(int)
        for t in self.traj_meta:
            for e in self.traj_edges(t):
                weight[e] += 1
        return dict(weight)


# ─────────────────────────────────────────────────────────────────────────────
# Per-type pool, measure and record
# ─────────────────────────────────────────────────────────────────────────────

Spec = Tuple[list, Callable, Callable]  # (targets, measure(target) -> int, record(target, stratum) -> dict)


def _node_spec(idx: TrainSessionIndex) -> Spec:
    degree = idx.node_out_degree()

    def record(nid, stratum):
        trajs = idx.node_to_trajs[nid]
        return {"nid": nid, "out_degree": degree.get(nid, 0), "stratum": stratum,
                "affected_traj_count": len(trajs), "affected_step_count": idx.step_count(trajs)}
    return list(idx.node_to_trajs), lambda nid: degree.get(nid, 0), record


def _edge_spec(idx: TrainSessionIndex) -> Spec:
    weight = idx.edge_weight()

    def record(edge, stratum):
        trajs = idx.edge_to_trajs[edge]
        return {"src": edge[0], "dst": edge[1], "weight": weight[edge], "stratum": stratum,
                "affected_traj_count": len(trajs), "affected_step_count": idx.step_count(trajs)}
    return list(weight), lambda edge: weight[edge], record


def _user_spec(idx: TrainSessionIndex) -> Spec:
    def record(uid, stratum):
        trajs = idx.user_to_trajs[uid]
        nodes, edges = set(), set()
        for t in trajs:
            nodes.update(idx.traj_meta[t]["cells"])
            edges.update(idx.traj_edges(t))
        return {"uid": uid, "traj_count": len(trajs), "stratum": stratum, "step_count": idx.step_count(trajs),
                "affected_node_count": len(nodes), "affected_edge_count": len(edges)}
    return list(idx.user_to_trajs), lambda uid: len(idx.user_to_trajs[uid]), record


def _traj_spec(idx: TrainSessionIndex) -> Spec:
    def record(traj_id, stratum):
        m = idx.traj_meta[traj_id]
        return {"uid": m["uid"], "day": m["day"], "start_time": m["start_time"], "end_time": m["end_time"],
                "length": len(m["cells"]), "stratum": stratum,
                "affected_node_count": len(set(m["cells"])), "affected_edge_count": len(set(idx.traj_edges(traj_id)))}
    return list(idx.traj_meta), lambda traj_id: len(idx.traj_meta[traj_id]["cells"]), record


SPECS = {"node": _node_spec, "edge": _edge_spec, "user": _user_spec, "traj": _traj_spec}
MEASURES = {"node": "out_degree", "edge": "weight", "user": "traj_count", "traj": "length"}


# ─────────────────────────────────────────────────────────────────────────────
# Strata
# ─────────────────────────────────────────────────────────────────────────────

def resolve_strata_path(path: str) -> str:
    """A bare file name lives next to this script."""
    return path if os.path.dirname(path) else os.path.join(_HERE, path)


def load_strata(path: str, source_dir: str, types: List[str]) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"strata file not found: {relpath_for_log(path)} (create it with the 'strata' command)")
    with open(path, encoding="utf-8") as f:
        strata = json.load(f)
    missing = [t for t in types if t not in strata.get("bounds", {})]
    if missing:
        raise ValueError(f"strata file {relpath_for_log(path)} has no bounds for {missing}")
    # source_dir is stored relative to the repository root, so the file is portable.
    if os.path.abspath(os.path.join(_REPO_ROOT, strata["source_dir"])) != os.path.abspath(source_dir):
        print(f"[WARN] strata file was computed on {strata['source_dir']}, not on {relpath_for_log(source_dir)}; "
              "using it anyway", flush=True)
    return strata


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def _stratified(idx: TrainSessionIndex, forget_type: str, bounds: List[int]):
    """(pool size, strata buckets, record(target) -> dict)."""
    pool, measure, record = SPECS[forget_type](idx)
    labels = stratum_labels(bounds)
    stratum = {}
    buckets: Dict[int, list] = defaultdict(list)
    for item in pool:
        s = stratum_of(measure(item), bounds)
        stratum[item] = s
        buckets[s].append(item)
    return len(pool), buckets, lambda item: record(item, labels[stratum[item]])


def _save(out_dir: str, fname: str, data: dict, size_label: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(output_row(data["forget_type"], f"{size_label:<8}{data['target_count']:>8,} targets -> {relpath_for_log(path)}"),
          flush=True)


def generate_batch(idx, forget_type: str, bounds: List[int], percents: List[float], seed: int, out_root: str, strata_file: str) -> None:
    """One forget set per percentage; each draws from a fresh RNG(seed), independent of the others."""
    pool_size, buckets, record = _stratified(idx, forget_type, bounds)
    for pct in percents:
        n = max(1, math.ceil(pool_size * pct / 100))
        chosen = sample_from_strata(buckets, pool_size, n, random.Random(seed))
        _save(os.path.join(out_root, DIR_NAMES[forget_type]), f"forget_{forget_type}_seed{seed}_{pct:g}pct.json", {
            "forget_type": forget_type, "percent": pct, "target_count": len(chosen), "seed": seed,
            "strata": {"file": repo_relpath(strata_file), "measure": MEASURES[forget_type], "upper_bounds": bounds},
            "records": [record(x) for x in chosen],
        }, f"{pct:g}%")


def generate_stream(idx, forget_type: str, bounds: List[int], n: int, seed: int, out_root: str, strata_file: str) -> None:
    """n targets, stratified, then shuffled; unlearn_stream.py deletes them one at a time in this order."""
    pool_size, buckets, record = _stratified(idx, forget_type, bounds)
    rng = random.Random(seed)
    chosen = sample_from_strata(buckets, pool_size, n, rng)
    rng.shuffle(chosen)
    _save(os.path.join(out_root, DIR_NAMES[forget_type]), f"forget_{forget_type}_seed{seed}_n{n}_stream.json", {
        "forget_type": forget_type, "order": "shuffled", "target_count": len(chosen), "seed": seed,
        "strata": {"file": repo_relpath(strata_file), "measure": MEASURES[forget_type], "upper_bounds": bounds},
        "records": [record(x) for x in chosen],
    }, f"n{n}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compute_strata(idx: TrainSessionIndex, source_dir: str) -> dict:
    bounds = {}
    for t, spec in SPECS.items():
        pool, measure, _ = spec(idx)
        values = [measure(x) for x in pool]
        bounds[t] = quartile_bounds(values)
        counts = np.bincount([stratum_of(v, bounds[t]) for v in values], minlength=len(bounds[t]) + 1)
        sizes = "   ".join(f"{lab} {100 * c / len(values):.1f}%" for lab, c in zip(stratum_labels(bounds[t]), counts))
        print(output_row(t, f"{MEASURES[t]:<14}{sizes}"), flush=True)
    return {"source_dir": os.path.relpath(os.path.abspath(source_dir), _REPO_ROOT), "method": f"quantile, {NUM_STRATA} strata (equal bounds merged)",
            "measures": MEASURES, "bounds": bounds}


def parse_args(argv=None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source_dir", default=os.path.join(_REPO_ROOT, "processed", "yjmob_1_14d"),
                        help="original preprocessed dataset")
    common.add_argument("--strata", default="strata.json",
                        help="strata file (a bare file name is next to this script)")
    sampling = argparse.ArgumentParser(add_help=False)
    sampling.add_argument("--seed", type=int, default=None,
                          help="sampling seed (default: random; it is part of the output file name)")
    sampling.add_argument("--out_root", default=os.path.join(_REPO_ROOT, "processed", "forget_sets", "yjmob"))

    p = argparse.ArgumentParser(description="Build yjmob forget sets.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("strata", parents=[common], help="compute and save the strata file")
    batch = sub.add_parser("batch", parents=[common, sampling], help="forget sets by percentage")
    batch.add_argument("--percent", type=float, nargs="+", default=[1.0],
                       help="share of targets to delete, in percent (e.g. 0.05 1 10)")
    batch.add_argument("--types", nargs="+", choices=list(SPECS), default=list(SPECS))
    stream = sub.add_parser("stream", parents=[common, sampling], help="ordered forget sets for unlearn_stream.py")
    stream.add_argument("--n", type=int, default=20, help="number of requests")
    stream.add_argument("--types", nargs="+", choices=list(STREAM_TYPES), default=list(STREAM_TYPES))
    return p.parse_args(argv)


def main(argv=None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)
    source_dir = os.path.abspath(args.source_dir)
    strata_path = resolve_strata_path(args.strata)

    if args.command == "strata":
        print(f"strata  source {relpath_for_log(source_dir)}", flush=True)
        idx = TrainSessionIndex(source_dir)
        strata = compute_strata(idx, source_dir)
        with open(strata_path, "w", encoding="utf-8") as f:
            json.dump(strata, f, indent=2)
        print(output_row("done", f"-> {relpath_for_log(strata_path)}"), flush=True)
        return

    if args.command == "batch":
        bad = [p for p in args.percent if not 0 < p <= 100]
        if bad:
            raise ValueError(f"--percent must be in (0, 100]: {bad}")
    elif args.n < 1:
        raise ValueError("--n must be at least 1")
    strata = load_strata(strata_path, source_dir, args.types)
    seed = resolve_seed(args.seed)
    print(f"forget sets  {args.command}  seed {seed}" + (" (random)" if args.seed is None else "")
          + f"  source {relpath_for_log(source_dir)}  strata {relpath_for_log(strata_path)}", flush=True)
    idx = TrainSessionIndex(source_dir)
    for t in args.types:
        if args.command == "batch":
            generate_batch(idx, t, strata["bounds"][t], args.percent, seed, args.out_root, strata_path)
        else:
            generate_stream(idx, t, strata["bounds"][t], args.n, seed, args.out_root, strata_path)


if __name__ == "__main__":
    main()
