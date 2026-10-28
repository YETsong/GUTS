"""Temporal Subgraph Assignment (paper §4.4).

Each day is partitioned independently, so shard index k on one day need not
cover the same region or users as index k on the previous day. This step maps
the day's K subgraphs one-to-one onto K persistent shard models by Hungarian
matching on

    Sim(i, m) = 1/3 * (J_node(i, m) + C_spatial(i, m) + J_user(i, m))      (Eq. 25)

  J_node   : Jaccard similarity of node sets.
  C_spatial: 1 / (1 + Euclidean distance) between grid-coordinate centroids
             (constant when no grid is available, e.g. static graphs).
  J_user   : Jaccard similarity of the user sets of the subgraphs' trajs.

The first day sets the references as is. Afterwards the node and user references
of a model are the unions over the last `window_size` days assigned to it, and
its spatial reference is an exponential moving average with decay
SPATIAL_REFERENCE_DECAY. The references persist across calls, so one assigner
must see the days in order.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

# Equal weights of the three similarities (Eq. 25).
SIM_WEIGHT = 1.0 / 3.0
# Decay of the spatial-centroid reference (EMA).
SPATIAL_REFERENCE_DECAY = 0.8

Descriptor = Dict[str, object]  # {"node_set": set, "spatial_centroid": ndarray, "uid_set": set}


class TemporalSubgraphAssigner:
    """Keeps the per-model references and relabels each day's partition."""

    def __init__(self, window_size: int, grid_info: Optional[Dict[str, int]]) -> None:
        self.grid_info = grid_info
        self.reference: Optional[Dict[int, Descriptor]] = None
        # Node/user sets of the last `window_size` days, per model slot.
        self.history: Deque[Dict[int, Dict[str, Set]]] = deque(maxlen=max(1, int(window_size)))

    def assign(self, partition, tc):
        """Relabel `partition` so shard ids are model slots; returns the same object."""
        K = len(partition.shards_nodes)
        if K == 0:
            return partition

        descriptors = self._build_descriptors(partition, tc)

        if self.reference is None:
            # First day: identity assignment, references seeded from this day.
            self.reference = {m: descriptors[m] for m in range(K)}
            self.history.append({
                m: {"node_set": descriptors[m]["node_set"], "uid_set": descriptors[m]["uid_set"]}
                for m in range(K)
            })
            partition.quality_metrics["temporal_assignment_score"] = 1.0
            return partition

        sim = self._similarity_matrix(descriptors, K)
        row_ind, col_ind = linear_sum_assignment(-sim)
        old_to_model = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
        assignment_score = float(sim[row_ind, col_ind].mean())

        self._relabel(partition, old_to_model)
        self._update_reference(descriptors, old_to_model, K)
        partition.quality_metrics["temporal_assignment_score"] = assignment_score
        return partition

    def _build_descriptors(self, partition, tc) -> Dict[int, Descriptor]:
        node_uids: Dict[int, Set[int]] = defaultdict(set)
        for traj, uid in zip(tc.trajs, tc.traj_uids):
            if uid is None:
                continue
            for n in traj:
                node_uids[int(n)].add(uid)

        grid_info = self.grid_info

        def _node_coord(node_id: int) -> np.ndarray:
            y_size = grid_info["y_size"]
            x_coord = node_id // y_size + grid_info["x_min"]
            y_coord = node_id % y_size + grid_info["y_min"]
            return np.asarray([x_coord, y_coord], dtype=np.float32)

        descriptors: Dict[int, Descriptor] = {}
        for shard_id, nodes in partition.shards_nodes.items():
            node_set = set(int(n) for n in nodes)
            if grid_info is not None and node_set:
                centroid = np.stack([_node_coord(n) for n in node_set]).mean(axis=0)
            else:
                centroid = np.zeros(2, dtype=np.float32)
            uid_set: Set[int] = set()
            for n in node_set:
                uid_set |= node_uids.get(n, set())
            descriptors[shard_id] = {
                "node_set": node_set,
                "spatial_centroid": np.asarray(centroid, dtype=np.float32),
                "uid_set": uid_set,
            }
        return descriptors

    def _similarity_matrix(self, descriptors: Dict[int, Descriptor], K: int) -> np.ndarray:
        sim = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            di = descriptors[i]
            for m in range(K):
                rm = self.reference[m]
                sim[i, m] = (
                    SIM_WEIGHT * self._jaccard(di["node_set"], rm["node_set"])
                    + SIM_WEIGHT * self._inv_dist(di["spatial_centroid"], rm["spatial_centroid"])
                    + SIM_WEIGHT * self._jaccard(di["uid_set"], rm["uid_set"])
                )
        return sim

    def _update_reference(
        self, descriptors: Dict[int, Descriptor], old_to_model: Dict[int, int], K: int
    ) -> None:
        rho = SPATIAL_REFERENCE_DECAY
        model_to_old = {m: o for o, m in old_to_model.items()}

        self.history.append({
            m: {
                "node_set": descriptors[model_to_old[m]]["node_set"],
                "uid_set": descriptors[model_to_old[m]]["uid_set"],
            }
            for m in range(K)
        })

        new_ref: Dict[int, Descriptor] = {}
        for m in range(K):
            z = descriptors[model_to_old[m]]
            prev = self.reference[m]
            windowed_node_set: Set = set()
            windowed_uid_set: Set = set()
            for day_entry in self.history:
                entry = day_entry.get(m)
                if entry:
                    windowed_node_set |= entry["node_set"]
                    windowed_uid_set |= entry["uid_set"]
            new_ref[m] = {
                "node_set": windowed_node_set,
                "spatial_centroid": rho * prev["spatial_centroid"] + (1 - rho) * z["spatial_centroid"],
                "uid_set": windowed_uid_set,
            }
        self.reference = new_ref

    @staticmethod
    def _relabel(partition, old_to_model: Dict[int, int]) -> None:
        perm = lambda s: old_to_model.get(s, s)
        partition.node_to_shard = {n: perm(s) for n, s in partition.node_to_shard.items()}
        partition.shards_nodes = {perm(s): nodes for s, nodes in partition.shards_nodes.items()}
        partition.shards_edges = {perm(s): ei for s, ei in partition.shards_edges.items()}

    @staticmethod
    def _jaccard(a: Set, b: Set) -> float:
        if not a and not b:
            return 1.0
        union = len(a | b)
        if union == 0:
            return 0.0
        return len(a & b) / union

    @staticmethod
    def _inv_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + d)
