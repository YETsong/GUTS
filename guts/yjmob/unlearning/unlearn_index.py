"""Index from forget-set targets to the shards whose training data contain them.

Built from the train sessions of a run's shard data (Sharding/processed/shard_<k>/);
session cells are global cell ids. A shard is affected by a target when one of its
train session segments contains it: a user (uid), a cell (node, including the
predicted last cell of a segment), a transition (edge) or a session, identified by
(uid, day, start_time, end_time) (path). Only these shards are retrained.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

PathKey = Tuple[int, int, int, int]  # (uid, day, start_time, end_time)
_FORMAT = 2


class GraphSAGEUnlearnIndex:
    def __init__(self) -> None:
        self.format = _FORMAT
        self.state_dir: str = ""
        self.num_shards: int = 0
        self.num_paths: int = 0
        self.uid_to_shards: Dict[int, Set[int]] = {}
        self.node_to_shards: Dict[int, Set[int]] = {}
        self.edge_to_shards: Dict[Tuple[int, int], Set[int]] = {}
        self.path_to_shards: Dict[PathKey, Set[int]] = {}

    @classmethod
    def build(cls, state_dir: str, sharding_dir: Optional[str] = None, log_fn=None) -> "GraphSAGEUnlearnIndex":
        log = log_fn if log_fn is not None else (lambda msg: print(msg, flush=True))
        self = cls()
        self.state_dir = os.path.abspath(state_dir)
        processed_root = os.path.join(sharding_dir or os.path.join(state_dir, "Sharding"), "processed")
        if not os.path.isdir(processed_root):
            raise FileNotFoundError(f"shard data not found: {processed_root}")
        shard_dirs = sorted(
            (d for d in os.listdir(processed_root) if d.startswith("shard_")),
            key=lambda d: int(d.split("_", 1)[1]),
        )
        if not shard_dirs:
            raise FileNotFoundError(f"no shard_<k>/ directories under {processed_root}")
        self.num_shards = len(shard_dirs)
        log(f"[UnlearnIndex] {processed_root}: {self.num_shards} shards")

        uid_to_shards: Dict[int, Set[int]] = defaultdict(set)
        node_to_shards: Dict[int, Set[int]] = defaultdict(set)
        edge_to_shards: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        path_to_shards: Dict[PathKey, Set[int]] = defaultdict(set)
        for shard_dir in shard_dirs:
            shard_id = int(shard_dir.split("_", 1)[1])
            processed = os.path.join(processed_root, shard_dir)
            meta = torch.load(os.path.join(processed, "next_location_meta.pt"), map_location="cpu")
            for part_info in meta["sequence_index"]["train"]["parts"]:
                part_path = os.path.join(processed, part_info["path"])
                if not os.path.exists(part_path):
                    raise FileNotFoundError(f"registered session part not found: {part_path}")
                for sess in torch.load(part_path, map_location="cpu"):
                    uid = int(sess["uid"])
                    cells = [int(c) for c in sess["cells"]]
                    uid_to_shards[uid].add(shard_id)
                    path_to_shards[(uid, int(sess["day"]), int(sess["start_time"]), int(sess["end_time"]))].add(shard_id)
                    for c in cells:
                        node_to_shards[c].add(shard_id)
                    for edge in zip(cells[:-1], cells[1:]):
                        edge_to_shards[edge].add(shard_id)
                    self.num_paths += 1

        self.uid_to_shards = dict(uid_to_shards)
        self.node_to_shards = dict(node_to_shards)
        self.edge_to_shards = dict(edge_to_shards)
        self.path_to_shards = dict(path_to_shards)
        log(f"[UnlearnIndex] {self.summary()}")
        return self

    @staticmethod
    def _union(table: dict, keys: Iterable) -> List[int]:
        shards: Set[int] = set()
        for key in keys:
            shards.update(table.get(key, ()))
        return sorted(shards)

    def affected_shards_for_uids(self, uids: Iterable[int]) -> List[int]:
        return self._union(self.uid_to_shards, uids)

    def affected_shards_for_nodes(self, nodes: Iterable[int]) -> List[int]:
        return self._union(self.node_to_shards, nodes)

    def affected_shards_for_edges(self, edges: Iterable[Tuple[int, int]]) -> List[int]:
        return self._union(self.edge_to_shards, edges)

    def affected_shards_for_path_keys(self, keys: Iterable[PathKey]) -> List[int]:
        return self._union(self.path_to_shards, keys)

    def summary(self) -> dict:
        return {
            "num_shards": self.num_shards,
            "train_segments": self.num_paths,
            "uids": len(self.uid_to_shards),
            "nodes": len(self.node_to_shards),
            "edges": len(self.edge_to_shards),
            "paths": len(self.path_to_shards),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[UnlearnIndex] saved → {path}", flush=True)

    @staticmethod
    def load(path: str) -> "GraphSAGEUnlearnIndex":
        with open(path, "rb") as f:
            idx = pickle.load(f)
        if getattr(idx, "format", 1) != _FORMAT:
            raise ValueError(f"{path} is an old unlearn index format; rebuild it (unlearn.py --rebuild_index)")
        return idx


def _cli():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Build the unlearn index of a GUTS yjmob run.")
    p.add_argument("--state_dir", required=True, help="source shard-runs directory")
    p.add_argument("--sharded-dir", dest="sharded_dir", default="", help="Sharding/ root (default: <state_dir>/Sharding)")
    p.add_argument("--out", default="", help="output path (default: <sharding_dir>/graphsage_unlearn_index.pkl)")
    args = p.parse_args()

    sharding_dir = os.path.abspath(args.sharded_dir) if args.sharded_dir else os.path.join(args.state_dir, "Sharding")
    idx = GraphSAGEUnlearnIndex.build(args.state_dir, sharding_dir=sharding_dir)
    idx.save(args.out or os.path.join(sharding_dir, "graphsage_unlearn_index.pkl"))
    print(json.dumps(idx.summary(), indent=2))


if __name__ == "__main__":
    _cli()
