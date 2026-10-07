"""GUTS partitioner for static graphs (Cora): every edge becomes a size-2 hyperedge."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import torch

from guts.partition.partitioner import GUTSPartitioner
from guts.partition.trajectory_paths import TrajectoryPaths


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PathCollection:
    """
    All extracted paths for a single graph snapshot.

    Attributes:
        paths     : List of node-ID tuples, e.g. (u, v) or (u, v, w).
                    Node IDs are local to the snapshot (same as in edge_index).
        day       : Source snapshot day (for provenance / caching).
        edge_index: Original edge_index this collection was derived from.
        num_nodes : Number of nodes in the source snapshot.
    """
    paths: List[Tuple[int, ...]]
    day: int
    edge_index: torch.Tensor   # [2, E]
    num_nodes: int

    def __len__(self) -> int:
        return len(self.paths)

    def path_lengths(self) -> List[int]:
        """Return list of path lengths (in edges = len(path) - 1)."""
        return [len(p) - 1 for p in self.paths]


# ──────────────────────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────────────────────

class PathExtractor:

    def __init__(self, max_path_length: int = 1, path_sample_ratio: float = 1.0) -> None:
        self.max_path_length = max_path_length
        self.path_sample_ratio = path_sample_ratio

    # ── Main extraction entry point ───────────────────────────────────────────

    def extract(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        day: int = -1,
        seed_ratio: Optional[float] = None,
    ) -> PathCollection:
        ratio = seed_ratio if seed_ratio is not None else self.path_sample_ratio

        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        num_edges = len(src_list)

        if num_edges == 0:
            return PathCollection(paths=[], day=day,
                                  edge_index=edge_index, num_nodes=num_nodes)

        # Build out-adjacency list: adj[u] = list of out-neighbors of u
        adj: defaultdict = defaultdict(list)
        for u, v in zip(src_list, dst_list):
            adj[u].append(v)

        # Sample seed edge indices
        num_seeds = max(1, int(np.ceil(num_edges * ratio)))
        num_seeds = min(num_seeds, num_edges)
        seed_indices = random.sample(range(num_edges), num_seeds)

        # Collect paths via randomized single-extension per seed
        paths_set: Set[Tuple[int, ...]] = set()
        max_L = self.max_path_length

        for idx in seed_indices:
            u, v = src_list[idx], dst_list[idx]

            # 1-edge path
            p1 = (u, v)
            paths_set.add(p1)

            if max_L < 2:
                continue

            # 2-edge extension: pick one random neighbor of v (not u)
            v_neighbors = [w for w in adj[v] if w != u]
            if not v_neighbors:
                continue
            w = random.choice(v_neighbors)
            p2 = (u, v, w)
            paths_set.add(p2)

            if max_L < 3:
                continue

            # 3-edge extension: pick one random neighbor of w (not in {u,v,w})
            visited2 = {u, v, w}
            w_neighbors = [x for x in adj[w] if x not in visited2]
            if not w_neighbors:
                continue
            x = random.choice(w_neighbors)
            p3 = (u, v, w, x)
            paths_set.add(p3)

            # For max_path_length > 3 (unlikely given scale, but defensive)
            if max_L >= 4:
                curr_path = p3
                curr_node = x
                for _ in range(max_L - 3):
                    visited_n = set(curr_path)
                    nn = [nb for nb in adj[curr_node] if nb not in visited_n]
                    if not nn:
                        break
                    nxt = random.choice(nn)
                    curr_path = curr_path + (nxt,)
                    paths_set.add(curr_path)
                    curr_node = nxt

        return PathCollection(
            paths=list(paths_set),
            day=day,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )


class SyntheticTrajectoryExtractor:

    def __init__(self, cfg) -> None:
        self._extractor = PathExtractor()

    def extract(self, sample):
        day = int(getattr(sample, "day", 0))
        pc = self._extractor.extract(sample.edge_index, int(sample.num_nodes), day=day)
        return TrajectoryPaths(paths=pc.paths, path_uids=[None] * len(pc.paths))


class CoraGUTSPartitioner(GUTSPartitioner):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._trajectory_extractor = SyntheticTrajectoryExtractor(cfg)


def make_cora_partitioner(cfg):
    strategy = getattr(cfg.partition, "strategy", "guts")
    if strategy != "guts":
        raise ValueError(f"Unknown partition.strategy: {strategy}")
    return CoraGUTSPartitioner(cfg)
