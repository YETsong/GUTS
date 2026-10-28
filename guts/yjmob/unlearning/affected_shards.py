"""Index from forget-set targets to the shards whose training data contain them.

Built from the train sessions of a run's shard data (Sharding/processed/shard_<k>/);
session cells are global cell ids. A shard is affected by a target when one of its
train session segments contains it: a user, a cell (node, including the predicted
last cell of a segment), a transition (edge) or a traj (one session), identified by
(uid, day, start_time, end_time). Only these shards are retrained. Built on every
unlearning run (about 40 s for YJMob-14d, K=130); not cached.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import torch

from guts.method.utils import relpath_for_log, repo_relpath

TrajKey = Tuple[int, int, int, int]  # (uid, day, start_time, end_time)


class AffectedShardIndex:
    def __init__(self) -> None:
        self.num_shards: int = 0
        self.num_segments: int = 0
        self.user_to_shards: Dict[int, Set[int]] = {}
        self.node_to_shards: Dict[int, Set[int]] = {}
        self.edge_to_shards: Dict[Tuple[int, int], Set[int]] = {}
        self.traj_to_shards: Dict[TrajKey, Set[int]] = {}

    @classmethod
    def build(cls, sharding_dir: str, log_fn=None) -> "AffectedShardIndex":
        log = log_fn if log_fn is not None else (lambda msg: print(msg, flush=True))
        t0 = time.time()
        self = cls()
        processed_root = os.path.join(sharding_dir, "processed")
        if not os.path.isdir(processed_root):
            raise FileNotFoundError(f"shard data not found: {relpath_for_log(processed_root)}")
        shard_dirs = sorted(
            (d for d in os.listdir(processed_root) if d.startswith("shard_")),
            key=lambda d: int(d.split("_", 1)[1]),
        )
        if not shard_dirs:
            raise FileNotFoundError(f"no shard_<k>/ directories under {relpath_for_log(processed_root)}")
        self.num_shards = len(shard_dirs)

        user_to_shards: Dict[int, Set[int]] = defaultdict(set)
        node_to_shards: Dict[int, Set[int]] = defaultdict(set)
        edge_to_shards: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        traj_to_shards: Dict[TrajKey, Set[int]] = defaultdict(set)
        for shard_dir in shard_dirs:
            shard_id = int(shard_dir.split("_", 1)[1])
            processed = os.path.join(processed_root, shard_dir)
            meta = torch.load(os.path.join(processed, "next_location_meta.pt"), map_location="cpu")
            for part_info in meta["sequence_index"]["train"]["parts"]:
                part_path = os.path.join(processed, part_info["path"])
                if not os.path.exists(part_path):
                    raise FileNotFoundError(f"registered session part not found: {relpath_for_log(part_path)}")
                for sess in torch.load(part_path, map_location="cpu"):
                    uid = int(sess["uid"])
                    cells = [int(c) for c in sess["cells"]]
                    user_to_shards[uid].add(shard_id)
                    traj_to_shards[(uid, int(sess["day"]), int(sess["start_time"]), int(sess["end_time"]))].add(shard_id)
                    for c in cells:
                        node_to_shards[c].add(shard_id)
                    for edge in zip(cells[:-1], cells[1:]):
                        edge_to_shards[edge].add(shard_id)
                    self.num_segments += 1

        self.user_to_shards = dict(user_to_shards)
        self.node_to_shards = dict(node_to_shards)
        self.edge_to_shards = dict(edge_to_shards)
        self.traj_to_shards = dict(traj_to_shards)
        log(f"  index: {self.num_shards} shards, {self.num_segments:,} train segments  {time.time() - t0:.0f}s")
        return self

    @staticmethod
    def _union(table: dict, keys: Iterable) -> List[int]:
        shards: Set[int] = set()
        for key in keys:
            shards.update(table.get(key, ()))
        return sorted(shards)

    def affected_shards_for_users(self, uids: Iterable[int]) -> List[int]:
        return self._union(self.user_to_shards, uids)

    def affected_shards_for_nodes(self, nodes: Iterable[int]) -> List[int]:
        return self._union(self.node_to_shards, nodes)

    def affected_shards_for_edges(self, edges: Iterable[Tuple[int, int]]) -> List[int]:
        return self._union(self.edge_to_shards, edges)

    def affected_shards_for_trajs(self, keys: Iterable[TrajKey]) -> List[int]:
        return self._union(self.traj_to_shards, keys)

    def affected_shards_for_records(self, forget_type: str, records: List[dict]) -> List[int]:
        """Affected shards of the records of a forget set JSON (forget_sets.py output)."""
        if forget_type == "user":
            return self.affected_shards_for_users(int(r["uid"]) for r in records)
        if forget_type == "node":
            return self.affected_shards_for_nodes(int(r["nid"]) for r in records)
        if forget_type == "edge":
            return self.affected_shards_for_edges((int(r["src"]), int(r["dst"])) for r in records)
        if forget_type == "traj":
            return self.affected_shards_for_trajs(
                (int(r["uid"]), int(r["day"]), int(r["start_time"]), int(r["end_time"])) for r in records)
        raise ValueError(f"Unknown forget_type: {forget_type}")

    def summary(self) -> dict:
        return {
            "num_shards": self.num_shards,
            "train_segments": self.num_segments,
            "users": len(self.user_to_shards),
            "nodes": len(self.node_to_shards),
            "edges": len(self.edge_to_shards),
            "trajs": len(self.traj_to_shards),
        }


def load_forget_records(forget_set_path: str) -> List[dict]:
    with open(forget_set_path) as f:
        return json.load(f)["records"]


def affected_shards(sharding_dir: str, forget_type: str, forget_set_path: str, log_fn=None) -> dict:
    """Affected shards of one forget set for the run whose Sharding/ dir is given (no retraining)."""
    records = load_forget_records(forget_set_path)
    idx = AffectedShardIndex.build(sharding_dir, log_fn=log_fn)
    shards = idx.affected_shards_for_records(forget_type, records)
    return {
        "sharding_dir": repo_relpath(sharding_dir),
        "forget_type": forget_type,
        "forget_set_path": repo_relpath(forget_set_path),
        "num_records": len(records),
        "num_shards": idx.num_shards,
        "num_affected": len(shards),
        "affected_shards": shards,
    }
