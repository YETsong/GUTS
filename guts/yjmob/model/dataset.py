"""Lazy session dataset and day-grouped sampler for next-grid prediction.

A processed directory (the preprocess output or one shard directory written by
guts/yjmob/data/shard_data.py) holds session parts under ``sequences/<split>/``
and, for training, rolling graphs listed in ``next_location_meta.pt`` under
``"rolling_graphs"`` (written by guts/yjmob/data/rolling_graphs.py).
"""

from __future__ import annotations

import bisect
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

_GRAPH_CACHE_SIZE = 8


def local_graph(graph: Dict, num_nodes: int) -> Dict:
    """A saved shard graph with edges relabeled to rows of x.

    Saved graphs keep one row of x per active node (sorted global ids) and
    global edge ids. ``node_lookup`` maps a global cell id to its row, or -1 for
    cells absent from the graph, whose graph embedding is then zero. Training
    and the shard routing evaluation both read graphs through this function.
    """
    active = graph["active_nodes"].long()
    if graph["x"].size(0) != active.numel():
        raise ValueError(
            f"x has {graph['x'].size(0)} rows but there are {active.numel()} active nodes "
            "(expected one row per active node)"
        )
    lookup = torch.full((int(num_nodes),), -1, dtype=torch.long)
    lookup[active] = torch.arange(active.numel(), dtype=torch.long)
    edge_local = lookup[graph["edge_index"].long()]
    valid = (edge_local[0] >= 0) & (edge_local[1] >= 0)
    edge_weight = graph.get("edge_weight")
    return {
        "x": graph["x"].to(torch.float32),
        "edge_index": edge_local[:, valid].contiguous(),
        "edge_weight": edge_weight[valid].contiguous() if edge_weight is not None else None,
        "node_lookup": lookup,
    }


class NextGridSessionDataset(Dataset):
    """Returns one compressed session for teacher-forced sequence training."""

    def __init__(self, processed_dir: str, split: str, part_cache_size: int = 128) -> None:
        self.processed_dir = os.path.abspath(processed_dir)
        self.split = split
        self.meta = torch.load(os.path.join(self.processed_dir, "next_location_meta.pt"), map_location="cpu")
        split_index = self.meta["sequence_index"][split]
        self.parts = split_index["parts"]
        self.part_cum_sessions: List[int] = []
        total = 0
        for part in self.parts:
            total += int(part["sessions"])
            self.part_cum_sessions.append(total)
        self.total_sessions = total
        self.total_steps = int(split_index["steps"])
        self._part_cache_idx = -1
        self._part_cache: List[Dict] = []
        self._part_lru_cache: OrderedDict[int, List[Dict]] = OrderedDict()
        self._graph_cache: OrderedDict[Tuple[int, int], Dict] = OrderedDict()
        self.max_part_cache_size = max(1, int(part_cache_size))
        self._rolling_graph_index: Dict[Tuple[int, int], dict] = {
            (int(entry["target_day"]), int(w)): entry
            for w, entries in self.meta.get("rolling_graphs", {}).items()
            for entry in entries
        }

    def __len__(self) -> int:
        return int(self.total_sessions)

    @property
    def num_nodes(self) -> int:
        return int(self.meta["num_nodes"])

    @property
    def node_feat_dim(self) -> int:
        return int(len(self.meta["feature_names"]))

    @property
    def has_rolling_graphs(self) -> bool:
        return bool(self._rolling_graph_index)

    def _load_part(self, part_idx: int) -> None:
        if part_idx == self._part_cache_idx:
            return
        if part_idx in self._part_lru_cache:
            self._part_lru_cache.move_to_end(part_idx)
            self._part_cache = self._part_lru_cache[part_idx]
            self._part_cache_idx = int(part_idx)
            return
        self._part_cache = torch.load(os.path.join(self.processed_dir, self.parts[part_idx]["path"]), map_location="cpu")
        self._part_lru_cache[int(part_idx)] = self._part_cache
        while len(self._part_lru_cache) > self.max_part_cache_size:
            self._part_lru_cache.popitem(last=False)
        self._part_cache_idx = int(part_idx)

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= self.total_sessions:
            raise IndexError(idx)
        part_idx = bisect.bisect_right(self.part_cum_sessions, idx)
        prev = self.part_cum_sessions[part_idx - 1] if part_idx > 0 else 0
        self._load_part(part_idx)
        sess = self._part_cache[idx - prev]
        cells = torch.tensor(sess["cells"], dtype=torch.long)
        return {
            "uid": int(sess["uid"]),
            "day": int(sess["day"]),
            "input_cells": cells[:-1],
            "target_cells": cells[1:],
            "length": int(cells.numel() - 1),
        }

    def get_rolling_graph(self, target_day: int, window_size: int) -> Dict:
        """Rolling graph of target_day (the previous window_size days merged), see local_graph."""
        key = (int(target_day), int(window_size))
        if key not in self._graph_cache:
            entry = self._rolling_graph_index.get(key)
            if entry is None:
                raise KeyError(f"No rolling graph for target_day={target_day}, window_size={window_size}")
            graph = torch.load(os.path.join(self.processed_dir, entry["path"]), map_location="cpu")
            if len(self._graph_cache) >= _GRAPH_CACHE_SIZE:
                self._graph_cache.popitem(last=False)
            self._graph_cache[key] = local_graph(graph, self.num_nodes)
        return self._graph_cache[key]


def build_day_index(processed_dir: str, split: str, meta: dict) -> Dict[int, List[int]]:
    """Scan all session parts and return {day: [global_session_indices]}.

    Result is cached to <processed_dir>/day_index_<split>.pt so the expensive
    scan only happens once.
    """
    cache_path = os.path.join(processed_dir, f"day_index_{split}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    day_to_indices: Dict[int, List[int]] = {}
    global_idx = 0
    for part_info in meta["sequence_index"][split]["parts"]:
        sessions = torch.load(os.path.join(processed_dir, part_info["path"]), map_location="cpu")
        for sess in sessions:
            day_to_indices.setdefault(int(sess["day"]), []).append(global_idx)
            global_idx += 1

    torch.save(day_to_indices, cache_path)
    return day_to_indices


class DayBatchSampler(Sampler):
    """Yields batches of session indices that never mix days.

    Every session of a batch is encoded with the rolling graph of the batch's
    day, so a batch must not cross a day boundary (otherwise sessions of one day
    would use another day's graph, possibly built from later days). Day order
    and within-day session order are shuffled each epoch when ``shuffle`` is set.

    Under DDP a day's sessions are strided across ranks; no session is
    duplicated. With ``equal_steps`` (training) every rank yields the same
    number of batches per day, so ranks with fewer sessions get empty batches
    (``[]``) that only join the gradient all-reduce. Evaluation does not need
    this, since ranks combine their totals only once at the end.

    Usage with DataLoader:
        sampler = DayBatchSampler(day_index, batch_size=bs, rank=rank, world_size=ws)
        loader  = DataLoader(ds, batch_sampler=sampler, ...)
    """

    def __init__(
        self,
        day_to_indices: Dict[int, List[int]],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        equal_steps: bool = True,
    ) -> None:
        self.day_to_indices = {int(d): list(v) for d, v in day_to_indices.items() if v}
        self.batch_size = int(batch_size)
        self.rank = rank
        self.world_size = max(int(world_size), 1)
        self.shuffle = shuffle
        self.seed = seed
        self.equal_steps = bool(equal_steps)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _num_batches(self, n_day: int) -> int:
        """Batches this rank yields for a day of n_day sessions."""
        if self.equal_steps:
            per_rank = -(-n_day // self.world_size)            # the largest rank share
        else:
            per_rank = len(range(self.rank, n_day, self.world_size))
        return -(-per_rank // self.batch_size)

    def __len__(self) -> int:
        return sum(self._num_batches(len(v)) for v in self.day_to_indices.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        days = sorted(self.day_to_indices.keys())
        if self.shuffle:
            rng.shuffle(days)
        for day in days:
            indices = list(self.day_to_indices[day])
            if self.shuffle:
                rng.shuffle(indices)
            rank_indices = indices[self.rank::self.world_size]
            for b in range(self._num_batches(len(indices))):
                yield rank_indices[b * self.batch_size:(b + 1) * self.batch_size]


def collate_next_grid_sessions(batch: List[Dict]) -> Dict:
    """Pad sessions to one batch; an empty batch (DDP filler) gives zero-sized tensors."""
    max_len = max((int(item["length"]) for item in batch), default=0)
    inputs = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), max_len), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    lengths = torch.empty((len(batch),), dtype=torch.long)
    days = torch.empty((len(batch),), dtype=torch.long)
    uids = torch.empty((len(batch),), dtype=torch.long)
    for i, item in enumerate(batch):
        n = int(item["length"])
        inputs[i, :n] = item["input_cells"]
        targets[i, :n] = item["target_cells"]
        mask[i, :n] = True
        lengths[i] = n
        days[i] = int(item["day"])
        uids[i] = int(item["uid"])
    return {
        "input_cells": inputs,
        "target_cells": targets,
        "mask": mask,
        "lengths": lengths,
        "day": days,
        "uid": uids,
    }
