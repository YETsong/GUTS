from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch


class RawUnlearnIndex:
    def __init__(self) -> None:
        self.source_dir: str = ""

        self.path_meta: Dict[int, dict] = {}
        self.path_to_edges: Dict[int, List[Tuple[int, int]]] = {}
        self.uid_to_paths: Dict[int, List[int]] = {}
        self.day_to_paths: Dict[int, List[int]] = {}
        self.node_to_paths: Dict[int, List[int]] = {}
        self.edge_to_paths: Dict[Tuple[int, int], List[int]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        source_dir: str,
        split: str = "train",
        log_fn=None,
    ) -> "RawUnlearnIndex":
        def _log(msg: str):
            (log_fn or print)(msg, flush=True)

        self = cls()
        self.source_dir = os.path.abspath(source_dir)

        train_dir = os.path.join(source_dir, "sequences", split)
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"{train_dir} not found — check that source_dir is the original next-grid processed directory")

        uid_to_paths: Dict[int, List[int]] = defaultdict(list)
        day_to_paths: Dict[int, List[int]] = defaultdict(list)
        node_to_paths: Dict[int, List[int]] = defaultdict(list)
        edge_to_paths: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        path_id = 0
        part_files = sorted(f for f in os.listdir(train_dir) if f.endswith(".pt"))
        _log(f"[RawUnlearnIndex] source_dir={source_dir}  parts={len(part_files)}")

        for i, fname in enumerate(part_files):
            sessions = torch.load(os.path.join(train_dir, fname), map_location="cpu")
            for sess in sessions:
                uid = int(sess["uid"])
                day = int(sess.get("day", -1))
                start_time = int(sess.get("start_time", -1))
                end_time = int(sess.get("end_time", -1))
                cells = [int(c) for c in sess["cells"]]
                edges = [(cells[i2], cells[i2 + 1]) for i2 in range(len(cells) - 1)]

                self.path_meta[path_id] = {
                    "uid": uid, "day": day,
                    "start_time": start_time, "end_time": end_time,
                    "cells": cells,
                }
                self.path_to_edges[path_id] = edges

                uid_to_paths[uid].append(path_id)
                day_to_paths[day].append(path_id)
                for c in set(cells):
                    node_to_paths[c].append(path_id)
                for e in set(edges):
                    edge_to_paths[e].append(path_id)

                path_id += 1

            if (i + 1) % 10 == 0 or i == len(part_files) - 1:
                _log(f"  [{i + 1}/{len(part_files)}] parts scanned, paths so far={path_id}")

        self.uid_to_paths = dict(uid_to_paths)
        self.day_to_paths = dict(day_to_paths)
        self.node_to_paths = dict(node_to_paths)
        self.edge_to_paths = dict(edge_to_paths)

        _log(f"[RawUnlearnIndex] total paths       : {len(self.path_meta)}")
        _log(f"[RawUnlearnIndex] unique uids        : {len(self.uid_to_paths)}")
        _log(f"[RawUnlearnIndex] unique days         : {len(self.day_to_paths)}")
        _log(f"[RawUnlearnIndex] unique global nodes : {len(self.node_to_paths)}")
        _log(f"[RawUnlearnIndex] unique global edges  : {len(self.edge_to_paths)}")

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Stats helpers
    # ─────────────────────────────────────────────────────────────────────────

    def uid_stats(self) -> Dict[int, dict]:
        stats = {}
        for uid, pids in self.uid_to_paths.items():
            sessions = len(pids)
            steps = sum(max(0, len(self.path_meta[pid]["cells"]) - 1) for pid in pids)
            stats[uid] = {"session_count": sessions, "step_count": steps}
        return stats

    def day_stats(self) -> Dict[int, dict]:
        stats = {}
        for day, pids in self.day_to_paths.items():
            sessions = len(pids)
            steps = sum(max(0, len(self.path_meta[pid]["cells"]) - 1) for pid in pids)
            stats[day] = {"session_count": sessions, "step_count": steps}
        return stats

    def day_to_uid_set(self) -> Dict[int, Set[int]]:
        out: Dict[int, Set[int]] = defaultdict(set)
        for pid, m in self.path_meta.items():
            out[m["day"]].add(m["uid"])
        return dict(out)

    def node_out_degree(self) -> Dict[int, int]:
        degree: Dict[int, int] = defaultdict(int)
        for edges in self.path_to_edges.values():
            for src, _ in edges:
                degree[src] += 1
        return dict(degree)

    def edge_weight(self) -> Dict[Tuple[int, int], int]:
        weight: Dict[Tuple[int, int], int] = defaultdict(int)
        for edges in self.path_to_edges.values():
            for e in edges:
                weight[e] += 1
        return dict(weight)

    def summary(self) -> dict:
        return {
            "total_paths": len(self.path_meta),
            "unique_uids": len(self.uid_to_paths),
            "unique_days": len(self.day_to_paths),
            "unique_global_nodes": len(self.node_to_paths),
            "unique_global_edges": len(self.edge_to_paths),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[RawUnlearnIndex] saved → {path}", flush=True)

    @staticmethod
    def load(path: str) -> "RawUnlearnIndex":
        with open(path, "rb") as f:
            return pickle.load(f)
