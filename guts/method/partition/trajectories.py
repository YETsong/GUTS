"""
Trajectories: the hyperedges given to the GUTS partitioner.

Each traj is a node sequence and becomes one hyperedge. Datasets build them from
their own data (yjmob: guts/yjmob/data/train_trajs.py; static graphs: one traj
per edge in guts/static_graph/stage_partition.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Trajectories:
    trajs: List[Tuple[int, ...]]
    traj_uids: List[Optional[int]]   # None when trajs carry no user (static graphs)

    def filter_to(self, num_nodes: int) -> "Trajectories":
        """Drop trajs shorter than two points or with a node outside [0, num_nodes)."""
        keep = [
            idx for idx, traj in enumerate(self.trajs)
            if len(traj) >= 2 and all(0 <= int(n) < int(num_nodes) for n in traj)
        ]
        if len(keep) == len(self.trajs):
            return self
        return Trajectories(
            trajs=[self.trajs[i] for i in keep],
            traj_uids=[self.traj_uids[i] for i in keep],
        )
