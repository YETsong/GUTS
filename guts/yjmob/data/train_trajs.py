"""Inputs of the GUTS partitioner read from a yjmob processed directory.

  - TrajectoryExtractor: the day's train sessions (<processed_dir>/train_trajs/, written
    by preprocess.py) as Trajectories, one hyperedge per session.
  - load_grid_info: the grid origin and height used by temporal subgraph assignment to
    map a cell id to its (x, y) coordinate.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

from guts.method.partition.trajectories import Trajectories


def load_grid_info(processed_dir: str) -> Optional[Dict[str, int]]:
    """Grid origin and height for node-id -> (x, y); None when the dataset has no grid."""
    if not processed_dir:
        return None
    path = os.path.join(processed_dir, "grid_info.pickle")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        info = pickle.load(f)
    return {
        "x_min": int(info["x_min"]),
        "y_min": int(info["y_min"]),
        "y_size": int(info["y_size"]),
    }


class TrajectoryExtractor:
    """Load one day's train sessions from <processed_dir>/train_trajs/."""

    def __init__(self, processed_dir: str) -> None:
        self.index_path = os.path.join(processed_dir, "train_trajs", "meta.pkl")
        self.min_points = 2
        self._index: Optional[Dict] = None
        self._day_cache: Dict[int, List[Dict]] = {}

    def extract(self, day: int) -> Trajectories:
        trajs: List[Tuple[int, ...]] = []
        traj_uids: List[Optional[int]] = []
        for item in self._trajs_for_day(int(day)):
            nodes = tuple(int(v) for v in item["nodes"])
            if len(nodes) < self.min_points:
                continue
            trajs.append(nodes)
            traj_uids.append(int(item["uid"]))
        return Trajectories(trajs=trajs, traj_uids=traj_uids)

    def _load_index(self) -> Dict:
        if self._index is not None:
            return self._index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"{self.index_path} not found. Build it with "
                "guts/yjmob/data/preprocess.py (or --train_trajs_only for an existing build)."
            )
        with open(self.index_path, "rb") as f:
            self._index = pickle.load(f)
        return self._index

    def _trajs_for_day(self, day: int) -> List[Dict]:
        if day in self._day_cache:
            return self._day_cache[day]
        day_file = self._load_index().get("day_files", {}).get(day)
        if day_file is None:
            # The index has no trajs for this day.
            self._day_cache[day] = []
            return []
        # Relative day files are resolved against the index file's directory.
        if not os.path.isabs(day_file):
            day_file = os.path.join(os.path.dirname(os.path.abspath(self.index_path)), day_file)
        if not os.path.exists(day_file):
            raise FileNotFoundError(
                f"Trajectory file for day {day} listed in the index is missing: {day_file}"
            )
        with open(day_file, "rb") as f:
            trajs = pickle.load(f)
        self._day_cache[day] = trajs
        return trajs
