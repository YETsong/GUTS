"""
Trajectory path extractor for GUTS.

Loads the day's train sessions written by guts/yjmob/data/preprocess.py
(<data.nextgrid_processed_dir>/train_paths/) and returns them as TrajectoryPaths.
Each path becomes one hyperedge.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrajectoryPaths:
    paths: List[Tuple[int, ...]]
    path_uids: List[Optional[int]]   # None when paths carry no user (static graphs)

    def filter_to(self, num_nodes: int) -> "TrajectoryPaths":
        """Drop paths shorter than two points or with a node outside [0, num_nodes)."""
        keep = [
            idx for idx, path in enumerate(self.paths)
            if len(path) >= 2 and all(0 <= int(n) < int(num_nodes) for n in path)
        ]
        if len(keep) == len(self.paths):
            return self
        return TrajectoryPaths(
            paths=[self.paths[i] for i in keep],
            path_uids=[self.path_uids[i] for i in keep],
        )


class TrajectoryPathExtractor:
    """Load the train sessions of a sample's day from the data directory."""

    def __init__(self, cfg) -> None:
        self.path_index_path = os.path.join(cfg.data.nextgrid_processed_dir, "train_paths", "meta.pkl")
        self.min_points = 2
        self._index: Optional[Dict] = None
        self._day_cache: Dict[int, List[Dict]] = {}

    def extract(self, sample) -> TrajectoryPaths:
        paths: List[Tuple[int, ...]] = []
        path_uids: List[Optional[int]] = []
        for item in self._paths_for_day(int(sample.day)):
            nodes = tuple(int(v) for v in item["nodes"])
            if len(nodes) < self.min_points:
                continue
            paths.append(nodes)
            path_uids.append(int(item["uid"]))
        return TrajectoryPaths(paths=paths, path_uids=path_uids)

    def _load_index(self) -> Dict:
        if self._index is not None:
            return self._index
        if not os.path.exists(self.path_index_path):
            raise FileNotFoundError(
                f"{self.path_index_path} not found. Build it with "
                "guts/yjmob/data/preprocess.py (or --train_paths_only for an existing build)."
            )
        with open(self.path_index_path, "rb") as f:
            self._index = pickle.load(f)
        return self._index

    def _paths_for_day(self, day: int) -> List[Dict]:
        if day in self._day_cache:
            return self._day_cache[day]
        day_file = self._load_index().get("day_files", {}).get(day)
        if day_file is None:
            # The index has no paths for this day.
            self._day_cache[day] = []
            return []
        # Relative day files are resolved against the index file's directory.
        if not os.path.isabs(day_file):
            day_file = os.path.join(os.path.dirname(os.path.abspath(self.path_index_path)), day_file)
        if not os.path.exists(day_file):
            raise FileNotFoundError(
                f"Trajectory path file for day {day} listed in the index is missing: {day_file}"
            )
        with open(day_file, "rb") as f:
            paths = pickle.load(f)
        self._day_cache[day] = paths
        return paths
