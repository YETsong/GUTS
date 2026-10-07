"""
GUTS partitioning: trajectory hypergraph partitioning weighted by deletion
influence(DI), with boundary refinement and temporal subgraph assignment.

Each day graph is partitioned in four stages (paper §4.1-§4.4):

  1. Deletion Influence Modeling (§4.1) — DI(v) is a weighted sum of trajectory
     statistics (TF, UC*, class heterophily; degree for static graphs). It needs
     no sampled deletion requests.
  2. Trajectory Hypergraph Partitioning (§4.2) — each trajectory is a hyperedge
     whose KM1 weight grows with the DI of its members, so KaHyPar avoids
     splitting high-influence trajectories across shards.
  3. Boundary Refinement (§4.3, refinement.py) — boundary nodes with the largest
     positive gain are moved (single moves, then pairwise swaps) to reduce KM1 and
     cross-shard transitions, optionally with a label-distribution term, under the
     shard-size imbalance refine_epsilon (0.05, the same as KaHyPar's epsilon).
  4. Temporal Subgraph Assignment (§4.4, assignment.py) — the day's subgraphs are matched to
     persistent shard models by Hungarian matching on node/spatial/user
     similarity. The reference descriptors are instance state kept across fit()
     calls, so one partitioner instance must handle the days in order.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from guts.partition.assignment import TemporalSubgraphAssigner, load_grid_info
from guts.partition.refinement import BoundaryRefiner
from guts.partition.trajectory_paths import (
    TrajectoryPathExtractor,
    TrajectoryPaths,
)
from guts.shard import build_shard_edges

import kahypar as _kahypar


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Hyperedge weight range: w(e) = 1 + round(KAPPA * q(e)) in {1, ..., 10} (Eq. 23).
KAPPA = 9


# ─────────────────────────────────────────────────────────────────────────────
# Partition result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GUTSPartition:
    """One day's partition. Shard ids are model slots after temporal subgraph assignment."""
    shards_nodes: Dict[int, List[int]]     # shard_id → sorted global node ids
    shards_edges: Dict[int, torch.Tensor]  # shard_id → intra-shard edge_index (local ids)
    node_to_shard: Dict[int, int] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Isolated-node placement
# ─────────────────────────────────────────────────────────────────────────────

def complete_isolated_nodes_fair(
    node_to_shard: Dict[int, int],
    isolated_nodes: List[int],
    node_labels: Dict[int, int],
    num_shards: int,
) -> None:
    """
    Place isolated nodes (no edge in the day graph, so not KaHyPar vertices) into
    shards, in place.

    Repeatedly picks the shard furthest below the target size (total nodes / K)
    and gives it a node of the class it lacks most relative to the global label
    distribution. Nodes left over once every shard reaches the target go to the
    currently smallest shard.
    """
    if not isolated_nodes:
        return

    total_nodes = len(node_to_shard) + len(isolated_nodes)
    num_classes = max(node_labels.values()) + 1
    global_counts = np.zeros(num_classes, dtype=np.int64)
    for lbl in node_labels.values():
        global_counts[lbl] += 1
    global_frac = global_counts / max(total_nodes, 1)

    queue: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for v in isolated_nodes:
        queue[node_labels[v]].append(v)
    remaining = len(isolated_nodes)

    shard_counts = np.zeros(num_shards, dtype=np.int64)
    shard_class_counts = np.zeros((num_shards, num_classes), dtype=np.int64)
    for v, s in node_to_shard.items():
        if 0 <= s < num_shards:
            shard_counts[s] += 1
            shard_class_counts[s, node_labels[v]] += 1

    target_size = total_nodes / num_shards

    while remaining > 0:
        deficits = target_size - shard_counts
        k = int(np.argmax(deficits))
        if deficits[k] <= 0:
            break

        expected = global_frac * max(int(shard_counts[k]), 1)
        class_deficits = expected - shard_class_counts[k]
        order = np.argsort(-class_deficits)

        placed = False
        for c in order:
            c = int(c)
            if queue[c]:
                v = queue[c].pop()
                node_to_shard[v] = k
                shard_counts[k] += 1
                shard_class_counts[k, c] += 1
                remaining -= 1
                placed = True
                break
        if not placed:
            break

    for c in range(num_classes):
        for v in queue[c]:
            k = int(np.argmin(shard_counts))
            node_to_shard[v] = k
            shard_counts[k] += 1
            shard_class_counts[k, c] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Partitioner
# ─────────────────────────────────────────────────────────────────────────────

class GUTSPartitioner:
    """GUTS partitioner (see the module docstring for the four stages).

    Output fields are compatible with ShardBuilder and the unlearning index.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.pcfg = cfg.partition
        self._trajectory_extractor = TrajectoryPathExtractor(cfg)

        # §4.4 state persists across fit() calls for successive days
        # (guts/yjmob/pipeline.py builds one partitioner per run).
        self._assigner = TemporalSubgraphAssigner(
            window_size=cfg.temporal_graph.window_size,
            grid_info=load_grid_info(cfg.data.nextgrid_processed_dir),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, sample, node_labels: Optional[np.ndarray] = None) -> GUTSPartition:
        partition, pc = self.fit_unassigned(sample, node_labels)
        return self.subgraph_assign(partition, pc)

    def fit_unassigned(
        self, sample, node_labels: Optional[np.ndarray] = None
    ) -> Tuple[GUTSPartition, TrajectoryPaths]:
        K = int(self.pcfg.num_shards)
        epsilon = float(self.pcfg.epsilon)
        seed = int(self.cfg.training.seed)
        label_array: Optional[np.ndarray] = None
        if node_labels is not None:
            if torch.is_tensor(node_labels):
                label_array = node_labels.detach().cpu().numpy()
            else:
                label_array = np.asarray(node_labels)
            label_array = label_array.reshape(-1).astype(np.int64, copy=False)
            if len(label_array) < int(sample.num_nodes):
                raise ValueError(
                    "node_labels must cover every node: "
                    f"got {len(label_array)}, expected at least {sample.num_nodes}"
                )

        pc = self._extract_paths(sample)
        active_nodes = self._active_nodes(sample)

        # §4.1: Deletion Influence Modeling
        di = self._compute_deletion_influence_scores(
            pc, active_nodes, sample.edge_index, int(sample.num_nodes), node_labels=label_array
        )

        # §4.2: Trajectory Hypergraph Partitioning (DI-weighted hyperedges + KaHyPar)
        local_to_global, hyperedges = self._build_hyperedges(pc, active_nodes)
        hyperedge_weights = self._build_hyperedge_weights(
            hyperedges, di, local_to_global, use_di_weighting=bool(self.pcfg.use_di)
        )

        node_to_local_shard = self._run_kahypar(
            hyperedges=hyperedges,
            num_vertices=len(local_to_global),
            K=K,
            epsilon=epsilon,
            seed=seed,
            hyperedge_weights=hyperedge_weights,
        )

        node_to_shard: Dict[int, int] = {
            int(local_to_global[local_id]): int(shard)
            for local_id, shard in node_to_local_shard.items()
        }

        # Nodes without edges are not KaHyPar vertices. With node labels (static graphs)
        if self.pcfg.include_isolated_nodes and label_array is not None:
            isolated_nodes = [
                v for v in range(int(sample.num_nodes)) if v not in node_to_shard
            ]
            if isolated_nodes:
                node_label_map = {v: int(label_array[v]) for v in range(int(sample.num_nodes))}
                complete_isolated_nodes_fair(node_to_shard, isolated_nodes, node_label_map, K)

        pre_refine_counts = np.bincount(
            np.fromiter(node_to_shard.values(), dtype=np.int64),
            minlength=K,
        )
        pre_refine_max_shard_size = (
            int(pre_refine_counts.max()) if pre_refine_counts.size else 0
        )

        # §4.3: Boundary Refinement (guts/partition/refinement.py)
        before_refine = self._quality_metrics(pc, node_to_shard, sample.edge_index, K, label_array)
        refine_stats: Dict[str, Any] = {}
        if self.pcfg.use_boundary_refinement:
            refine_started = time.perf_counter()
            node_to_shard, refine_stats = BoundaryRefiner(
                self.pcfg, pc, sample.edge_index, hyperedges, local_to_global,
                node_to_shard, di, K, label_array,
            ).run()
            refine_stats["pre_refine_max_shard_size"] = pre_refine_max_shard_size
            refine_stats["wall_time_s"] = time.perf_counter() - refine_started

        shards_nodes = self._build_shards_nodes(node_to_shard, K)
        shards_edges = build_shard_edges(sample.edge_index, shards_nodes, sample.num_nodes)

        quality_metrics = self._quality_metrics(pc, node_to_shard, sample.edge_index, K, label_array)
        for key, value in refine_stats.items():
            if isinstance(value, (float, np.floating)):
                value = round(float(value), 6)
            quality_metrics[f"refine_{key}"] = value
        for key in before_refine:
            quality_metrics[f"refine_{key}_before"] = before_refine[key]
            quality_metrics[f"refine_{key}_after"] = quality_metrics[key]

        partition = GUTSPartition(
            shards_nodes=shards_nodes,
            shards_edges=shards_edges,
            node_to_shard=node_to_shard,
            quality_metrics=quality_metrics,
        )
        return partition, pc

    def subgraph_assign(self, partition: GUTSPartition, pc: TrajectoryPaths) -> GUTSPartition:
        """§4.4 (guts/partition/assignment.py); days must be passed in order."""
        if self.pcfg.use_temporal_subgraph_assignment:
            partition = self._assigner.assign(partition, pc)
        return partition

    def save(self, partition: GUTSPartition, path: str) -> None:
        from guts.utils import save_pickle
        save_pickle(partition, path)

    def load(self, path: str) -> GUTSPartition:
        from guts.utils import load_pickle
        return load_pickle(path)

    # ──────────────────────────────────────────────────────────────────────
    # Path extraction
    # ──────────────────────────────────────────────────────────────────────

    def _extract_paths(self, sample) -> TrajectoryPaths:
        return self._trajectory_extractor.extract(sample).filter_to(int(sample.num_nodes))

    @staticmethod
    def _active_nodes(sample) -> Set[int]:
        """Nodes with at least one edge in the day graph (the KaHyPar vertices)."""
        src = sample.edge_index[0].detach().cpu().tolist()
        dst = sample.edge_index[1].detach().cpu().tolist()
        num_nodes = int(sample.num_nodes)
        return {int(n) for n in src + dst if 0 <= int(n) < num_nodes}

    # ──────────────────────────────────────────────────────────────────────
    # §4.1: Deletion Influence Modeling
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_tf(pc: TrajectoryPaths, active_nodes: Set[int], num_nodes: int) -> np.ndarray:
        """TF(v) (Eq. 14): number of path transitions incident to v."""
        tf: Dict[int, int] = defaultdict(int)
        for path in pc.paths:
            for src, dst in zip(path, path[1:]):
                src, dst = int(src), int(dst)
                if src in active_nodes:
                    tf[src] += 1
                if dst != src and dst in active_nodes:
                    tf[dst] += 1
        out = np.zeros(num_nodes, dtype=np.float32)
        for v, count in tf.items():
            out[v] = float(count)
        return out

    @staticmethod
    def _compute_uc_star(pc: TrajectoryPaths, active_nodes: Set[int], num_nodes: int) -> np.ndarray:
        """UC*(v) (Eq. 15-16): probability that two visits to v drawn without replacement
        come from the same user, sum_u c_u(c_u - 1) / (T(T - 1))."""
        user_visits: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for path, uid in zip(pc.paths, pc.path_uids):
            if uid is None:
                continue
            for v in path:
                v = int(v)
                if v in active_nodes:
                    user_visits[v][int(uid)] += 1
        out = np.zeros(num_nodes, dtype=np.float32)
        for v, counts in user_visits.items():
            total = sum(counts.values())
            if total > 1:
                out[v] = sum(c * (c - 1) for c in counts.values()) / (total * (total - 1))
        return out

    @staticmethod
    def _compute_ch(
        node_labels: np.ndarray,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> np.ndarray:
        """CH(v) (Eq. 17), used when node classes are available (the static graphs here):
        the mean over v's neighbours of 1[y_v != y_w] * (1/n(y_v) + 1/n(y_w)) / 2, where
        n(c) is the number of nodes of class c."""
        labels = np.asarray(node_labels).reshape(-1)
        counts = np.bincount(labels[labels >= 0].astype(np.int64)) if labels.size else np.array([])
        freq = {int(c): max(int(n), 1) for c, n in enumerate(counts)}

        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        score = np.zeros(num_nodes, dtype=np.float64)
        degree = np.zeros(num_nodes, dtype=np.float64)
        for u, v in zip(src.tolist(), dst.tolist()):
            if u == v or not (0 <= u < len(labels)) or not (0 <= v < len(labels)):
                continue
            lu, lv = int(labels[u]), int(labels[v])
            degree[u] += 1
            if lu != lv and lu >= 0 and lv >= 0:
                score[u] += 0.5 * (1.0 / freq.get(lu, 1) + 1.0 / freq.get(lv, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(degree > 0, score / degree, 0.0)
        return out.astype(np.float32)

    @staticmethod
    def _compute_tf_static(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
        """TF for static graphs: the node degree (self-loops excluded)."""
        src = edge_index[0].cpu().numpy().astype(np.int64)
        dst = edge_index[1].cpu().numpy().astype(np.int64)
        keep = (src != dst) & (src >= 0) & (src < num_nodes)
        return np.bincount(src[keep], minlength=num_nodes).astype(np.float32)

    def _compute_deletion_influence_scores(
        self,
        pc: TrajectoryPaths,
        active_nodes: Set[int],
        edge_index: torch.Tensor,
        num_nodes: int,
        node_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """DI(v) = alpha * TF + beta * UC* + gamma * CH (Eq. 19), each term max-normalised
        (Eq. 18). For static graphs TF is the node degree, weighted by di_alpha_static.
        Terms with a zero coefficient are not computed."""
        alpha = float(self.pcfg.di_alpha)
        beta = float(self.pcfg.di_beta)
        gamma = float(self.pcfg.di_gamma)
        alpha_static = float(self.pcfg.di_alpha_static)
        if gamma and node_labels is None:
            raise ValueError("partition.di_gamma > 0 needs node labels (class heterophily).")

        di = np.zeros(num_nodes, dtype=np.float32)
        if alpha:
            di = di + alpha * self._normalise(self._compute_tf(pc, active_nodes, num_nodes))
        if beta:
            di = di + beta * self._normalise(self._compute_uc_star(pc, active_nodes, num_nodes))
        if gamma:
            di = di + gamma * self._normalise(self._compute_ch(node_labels, edge_index, num_nodes))
        if alpha_static:
            di = di + alpha_static * self._normalise(self._compute_tf_static(edge_index, num_nodes))
        return di

    # ──────────────────────────────────────────────────────────────────────
    # §4.2: Trajectory Hypergraph Partitioning — weighted hyperedges
    # ──────────────────────────────────────────────────────────────────────

    def _build_hyperedges(
        self,
        pc: TrajectoryPaths,
        active_nodes: Set[int],
    ) -> Tuple[List[int], List[List[int]]]:
        """Map active nodes to local vertex ids 0..n-1 and turn each path with at least
        two distinct active nodes into a hyperedge (Eq. 21). Every active node is a vertex, so
        KaHyPar assigns all of them (nodes in no hyperedge are placed freely)."""
        global_to_local: Dict[int, int] = {}
        local_to_global: List[int] = []

        for node in sorted(active_nodes):
            global_to_local[int(node)] = len(local_to_global)
            local_to_global.append(int(node))

        hyperedges: List[List[int]] = []
        for path in pc.paths:
            local_ids = list(dict.fromkeys(
                global_to_local[int(n)] for n in path
                if int(n) in global_to_local
            ))
            if len(local_ids) >= 2:
                hyperedges.append(local_ids)

        return local_to_global, hyperedges

    @staticmethod
    def _build_hyperedge_weights(
        hyperedges: List[List[int]],
        di: np.ndarray,
        local_to_global: List[int],
        use_di_weighting: bool,
    ) -> List[int]:
        """Hyperedge weight w(e) = 1 + round(KAPPA * q(e)) in {1, ..., 10} (Eq. 23), where
        q(e) is the percentile rank of the mean DI of e's nodes (Eq. 22). Weights are all 1 when
        DI weighting is off or every score is equal (the rank would only follow the
        hyperedge index). Vertex weights stay 1: they measure node count, not importance."""
        weights = [1] * len(hyperedges)
        if not use_di_weighting or not hyperedges:
            return weights
        scores = np.asarray(
            [float(np.mean([float(di[local_to_global[l]]) for l in he])) for he in hyperedges],
            dtype=np.float64,
        )
        if np.ptp(scores) == 0.0:
            return weights
        n = len(scores)
        order = np.argsort(scores, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        percentiles = ranks / max(n - 1, 1)
        return (1 + np.round(KAPPA * percentiles).astype(np.int64)).tolist()

    # ──────────────────────────────────────────────────────────────────────
    # §4.2: Trajectory Hypergraph Partitioning — KaHyPar
    # ──────────────────────────────────────────────────────────────────────

    def _run_kahypar(
        self,
        hyperedges: List[List[int]],
        num_vertices: int,
        K: int,
        epsilon: float,
        seed: int,
        hyperedge_weights: List[int],
    ) -> Dict[int, int]:
        """Local vertex id -> block, by direct K-way weighted-KM1 initial partitioning
        (kahypar_km1_initial_only.ini). Without hyperedges, vertices go round-robin."""
        if not hyperedges or num_vertices == 0:
            return {i: i % K for i in range(num_vertices)}

        hyperedge_indices = [0]
        hyperedge_vertices = []
        for he in hyperedges:
            hyperedge_vertices.extend(he)
            hyperedge_indices.append(len(hyperedge_vertices))

        num_hyperedges = len(hyperedges)
        vertex_weights = [1] * num_vertices

        hypergraph = _kahypar.Hypergraph(
            num_vertices,
            num_hyperedges,
            hyperedge_indices,
            hyperedge_vertices,
            K,
            hyperedge_weights,
            vertex_weights,
        )

        context = _kahypar.Context()
        ini_path = os.path.join(os.path.dirname(__file__), "kahypar_km1_initial_only.ini")
        if not os.path.exists(ini_path):
            raise FileNotFoundError(f"KaHyPar INI config not found: {ini_path}")
        context.loadINIconfiguration(ini_path)
        context.setK(K)
        context.setEpsilon(epsilon)
        context.setSeed(seed)
        context.suppressOutput(True)

        _kahypar.partition(hypergraph, context)

        node_to_local_shard: Dict[int, int] = {
            i: int(hypergraph.blockID(i)) for i in range(num_vertices)
        }
        return node_to_local_shard

    # ──────────────────────────────────────────────────────────────────────
    # Shard construction and quality metrics
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_shards_nodes(node_to_shard: Dict[int, int], K: int) -> Dict[int, List[int]]:
        """shard id -> sorted global node ids, for every shard 0..K-1."""
        shards_nodes: Dict[int, List[int]] = {sid: [] for sid in range(K)}
        for node, shard in node_to_shard.items():
            shards_nodes[shard].append(node)
        return {sid: sorted(nodes) for sid, nodes in shards_nodes.items()}

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        """Divide by the maximum (all zeros when the maximum is ~0)."""
        values = values.astype(np.float32)
        vmax = float(values.max()) if values.size else 0.0
        if vmax <= 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return values / vmax

    @staticmethod
    def _quality_metrics(
        pc: TrajectoryPaths,
        node_to_shard: Dict[int, int],
        edge_index: torch.Tensor,
        K: int,
        node_labels: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Partition quality, chosen by whether the graph has node labels.

        Trajectory graphs (no labels) — the two measures reported in the paper (Sec. 6):
          shards_per_path: mean connectivity lambda_e of the path hyperedges (Eq. 11),
                           i.e. the mean number of shards a path spans.
          cstr           : share of graph edges (transitions) whose endpoints lie in
                           different shards (an unassigned endpoint counts as different).
        Static node-classification graphs (labels):
          balance : sum_k |n_k - N/K| / (2N) over shard node counts n_k (0 = equal sizes).
          fairness: mean over shards of the total-variation distance between the shard's
                    label distribution and the global one (0 = same everywhere; an
                    empty shard counts as 0).
        """
        if node_labels is not None:
            return GUTSPartitioner._label_quality_metrics(node_to_shard, K, node_labels)

        touched_counts = [
            len({node_to_shard[int(n)] for n in path if int(n) in node_to_shard})
            for path in pc.paths
        ]
        shards_per_path = float(np.mean(touched_counts)) if touched_counts else 0.0

        src = edge_index[0].cpu().numpy().astype(np.int64)
        dst = edge_index[1].cpu().numpy().astype(np.int64)
        if len(src) == 0:
            cstr = 0.0
        else:
            size = max(int(src.max()), int(dst.max()), max(node_to_shard, default=-1)) + 1
            shard_of = np.full(size, -1, dtype=np.int64)  # -1 = unassigned
            for node, shard in node_to_shard.items():
                shard_of[node] = shard
            cross = int(np.count_nonzero(shard_of[src] != shard_of[dst]))
            cstr = cross / len(src)

        return {
            "shards_per_path": round(shards_per_path, 6),
            "cstr": round(float(cstr), 6),
        }

    @staticmethod
    def _label_quality_metrics(
        node_to_shard: Dict[int, int], K: int, node_labels: np.ndarray
    ) -> Dict[str, float]:
        """Node-count balance and label fairness (see _quality_metrics)."""
        nodes = np.fromiter(node_to_shard.keys(), dtype=np.int64)
        shards = np.fromiter(node_to_shard.values(), dtype=np.int64)
        counts = np.bincount(shards, minlength=K).astype(np.float64)
        n_total = float(counts.sum())
        balance = float(np.abs(counts - n_total / K).sum() / (2.0 * n_total)) if n_total else 0.0

        labels = np.asarray(node_labels).reshape(-1)[nodes]
        valid = labels >= 0
        labels, shards = labels[valid], shards[valid]
        if labels.size == 0:
            return {"balance": round(balance, 6), "fairness": 0.0}
        classes, label_idx = np.unique(labels, return_inverse=True)
        table = np.zeros((K, len(classes)), dtype=np.float64)
        np.add.at(table, (shards, label_idx), 1.0)
        global_dist = table.sum(axis=0) / table.sum()
        shard_sizes = table.sum(axis=1)
        tv = np.zeros(K, dtype=np.float64)
        nonempty = shard_sizes > 0
        tv[nonempty] = 0.5 * np.abs(
            table[nonempty] / shard_sizes[nonempty, None] - global_dist
        ).sum(axis=1)
        return {"balance": round(balance, 6), "fairness": round(float(tv.mean()), 6)}

def make_partitioner(cfg):
    """Return the partitioner selected by cfg.partition.strategy."""
    strategy = cfg.partition.strategy
    if strategy == "guts":
        return GUTSPartitioner(cfg)
    raise ValueError(f"Unknown partition.strategy: {strategy}")
