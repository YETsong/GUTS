"""Boundary Refinement (paper §4.3).

Starting from the KaHyPar assignment, boundary nodes are moved between shards to
maximise (Eq. 24)

    Gain(v) = -[(1 + lambda_di * DI(v)) * dKM1(v) + lambda_c * dCSTR(v)] + lambda_l * G_label(v)

  dKM1   : change in KM1 over the hyperedges containing v. Hyperedges count with
           unit weight here; the DI weights already shaped the KaHyPar objective.
  dCSTR  : change in the number of v's graph edges that cross shards.
  G_label: reduction in the total-variation distance between the class
           distributions of the two shards and the global one (needs labels).

Each pass scores every boundary node (a node with a neighbour or a hyperedge in
another shard) by its best admissible single move and keeps the top
`refine_candidate_max_ratio` of the positive-gain nodes. For each candidate, the
best admissible single move is applied; if none has a positive gain, a swap with
a neighbour in a candidate shard is tried. A move is admissible when it keeps
local connectivity (>= 2 neighbours in the source shard, >= 1 in the destination)
and the destination below the shard-size cap ceil(n / K * (1 + refine_epsilon)),
where n is the number of partitioned nodes. Moved nodes cannot return to their source
shard for REFINE_TABU_TENURE passes. Passes stop when nothing improves. With
`refine_class_balance_passes`, extra passes then move nodes by G_label alone.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

REFINE_MIN_GAIN = 1e-8
REFINE_SWAP_CANDIDATE_LIMIT = 20
REFINE_TABU_TENURE = 3


class BoundaryRefiner:
    """Refines one day's node-to-shard assignment; call run() once."""

    def __init__(
        self,
        pcfg,
        pc,
        edge_index: torch.Tensor,
        hyperedges: List[List[int]],
        local_to_global: List[int],
        node_to_shard: Dict[int, int],
        di: np.ndarray,
        K: int,
        node_labels: Optional[np.ndarray],
    ) -> None:
        self.lambda_c = float(pcfg.lambda_c)
        self.lambda_di = float(pcfg.lambda_di)
        self.lambda_l = float(pcfg.lambda_l)
        self.max_passes = int(pcfg.refine_max_passes)
        self.candidate_max_ratio = float(pcfg.refine_candidate_max_ratio)
        self.class_balance_passes = bool(pcfg.refine_class_balance_passes)
        refine_epsilon = float(pcfg.refine_epsilon)
        if self.lambda_l < 0:
            raise ValueError(f"partition.lambda_l must be >= 0, got {self.lambda_l}")
        if not 0.0 <= self.candidate_max_ratio <= 1.0:
            raise ValueError(
                "partition.refine_candidate_max_ratio must be in [0, 1], "
                f"got {self.candidate_max_ratio}"
            )
        self.use_label_distribution = self.lambda_l > 0

        self.K = K
        self.di = di
        self.node_to_shard = dict(node_to_shard)

        # Hyperedges in global ids and their per-shard node counts.
        self.global_hyperedges = [set(local_to_global[l] for l in he) for he in hyperedges]
        self.node_to_hyperedges: Dict[int, List[int]] = defaultdict(list)
        for hidx, he in enumerate(self.global_hyperedges):
            for v in he:
                self.node_to_hyperedges[v].append(hidx)
        self.hyperedge_shard_counts: List[Dict[int, int]] = []
        for he in self.global_hyperedges:
            counts: Dict[int, int] = {}
            for v in he:
                s = self.node_to_shard.get(v)
                if s is not None:
                    counts[s] = counts.get(s, 0) + 1
            self.hyperedge_shard_counts.append(counts)

        # Graph neighbours and their per-shard counts.
        src = edge_index[0].cpu().numpy().tolist()
        dst = edge_index[1].cpu().numpy().tolist()
        self.node_incident: Dict[int, List[int]] = defaultdict(list)
        for u, v in zip(src, dst):
            if u == v:
                continue
            self.node_incident[u].append(v)
            self.node_incident[v].append(u)
        self.neighbor_shard_count: Dict[int, Dict[int, int]] = {}
        for v, neighbors in self.node_incident.items():
            counts = {}
            for u in neighbors:
                s = self.node_to_shard.get(u)
                if s is not None:
                    counts[s] = counts.get(s, 0) + 1
            self.neighbor_shard_count[v] = counts

        # Shard sizes and the size cap.
        self.shard_node_counts = np.zeros(K, dtype=np.int64)
        for s in self.node_to_shard.values():
            if 0 <= s < K:
                self.shard_node_counts[s] += 1
        self.counts_before = self.shard_node_counts.copy()
        self.cap = int(np.ceil(len(self.node_to_shard) / K * (1.0 + refine_epsilon)))

        # Users of each node and the shards each user's paths touch (destination shortlist).
        self.uid_to_shards: Dict[int, Set[int]] = defaultdict(set)
        self.node_uids: Dict[int, Set[int]] = defaultdict(set)
        for path, uid in zip(pc.paths, pc.path_uids):
            if uid is None:
                continue
            path_shards = {self.node_to_shard[int(n)] for n in path if int(n) in self.node_to_shard}
            self.uid_to_shards[uid] |= path_shards
            for n in path:
                self.node_uids[int(n)].add(uid)

        self._init_labels(node_labels)
        self.tabu: Dict[Tuple[int, int], int] = {}  # (node, from_shard) -> pass it becomes legal again

        self.single_moves = 0
        self.swap_moves = 0
        self.gain_sum = 0.0
        self.label_gain_sum = 0.0
        self.label_improving_moves = 0
        self.label_only_passes = 0
        self.label_only_single_moves = 0
        self.label_only_gain_sum = 0.0

    # ── Class-distribution term ───────────────────────────────────────────────

    def _init_labels(self, node_labels: Optional[np.ndarray]) -> None:
        K = self.K
        self.label_indices: Dict[int, int] = {}
        self.shard_label_counts = np.zeros((K, 0), dtype=np.int64)
        self.shard_label_totals = np.zeros(K, dtype=np.int64)
        self.global_label_dist = np.zeros(0, dtype=np.float64)
        self.labelled_node_count = 0
        if node_labels is not None:
            valid_nodes = [
                node for node in self.node_to_shard
                if 0 <= node < len(node_labels) and int(node_labels[node]) >= 0
            ]
            unique_labels = sorted({int(node_labels[node]) for node in valid_nodes})
            label_to_col = {label: idx for idx, label in enumerate(unique_labels)}
            if unique_labels:
                self.shard_label_counts = np.zeros((K, len(unique_labels)), dtype=np.int64)
                global_counts = np.zeros(len(unique_labels), dtype=np.int64)
                for node in valid_nodes:
                    col = label_to_col[int(node_labels[node])]
                    shard = self.node_to_shard[node]
                    self.label_indices[node] = col
                    self.shard_label_counts[shard, col] += 1
                    self.shard_label_totals[shard] += 1
                    global_counts[col] += 1
                self.labelled_node_count = int(global_counts.sum())
                self.global_label_dist = global_counts.astype(np.float64) / self.labelled_node_count
        if self.use_label_distribution and self.labelled_node_count == 0:
            raise ValueError(
                "partition.lambda_l > 0 requires non-negative node_labels "
                "for the partitioned nodes"
            )
        self.label_gain_scale = self.labelled_node_count / K
        self.label_loss_before = self.label_distribution_loss()

    def shard_label_loss(
        self,
        shard: int,
        counts: Optional[np.ndarray] = None,
        total: Optional[int] = None,
    ) -> float:
        if self.labelled_node_count == 0:
            return 0.0
        values = self.shard_label_counts[shard] if counts is None else counts
        size = int(self.shard_label_totals[shard]) if total is None else int(total)
        if size <= 0:
            return 0.0
        return 0.5 * float(np.abs(values / float(size) - self.global_label_dist).sum())

    def label_distribution_loss(self) -> float:
        if self.labelled_node_count == 0:
            return 0.0
        return sum(self.shard_label_loss(shard) for shard in range(self.K)) / self.K

    def label_move_gain(self, v: int, a: int, b: int) -> float:
        col = self.label_indices.get(v)
        if col is None or a == b:
            return 0.0
        before = self.shard_label_loss(a) + self.shard_label_loss(b)
        counts_a = self.shard_label_counts[a].copy()
        counts_b = self.shard_label_counts[b].copy()
        counts_a[col] -= 1
        counts_b[col] += 1
        after = (
            self.shard_label_loss(a, counts_a, self.shard_label_totals[a] - 1)
            + self.shard_label_loss(b, counts_b, self.shard_label_totals[b] + 1)
        )
        return self.label_gain_scale * (before - after)

    def label_swap_gain(self, v: int, u: int, a: int, b: int) -> float:
        col_v, col_u = self.label_indices.get(v), self.label_indices.get(u)
        if col_v is None or col_u is None or col_v == col_u or a == b:
            return 0.0
        before = self.shard_label_loss(a) + self.shard_label_loss(b)
        counts_a = self.shard_label_counts[a].copy()
        counts_b = self.shard_label_counts[b].copy()
        counts_a[col_v] -= 1
        counts_b[col_v] += 1
        counts_b[col_u] -= 1
        counts_a[col_u] += 1
        after = (
            self.shard_label_loss(a, counts_a, self.shard_label_totals[a])
            + self.shard_label_loss(b, counts_b, self.shard_label_totals[b])
        )
        return self.label_gain_scale * (before - after)

    # ── Gain of a move ────────────────────────────────────────────────────────

    def deg_within(self, v: int, shard_id: int) -> int:
        return self.neighbor_shard_count.get(v, {}).get(shard_id, 0)

    def dest_candidates(self, v: int) -> Set[int]:
        """Shards of v's neighbours, of its hyperedges and of its users' paths."""
        cur = self.node_to_shard[v]
        cands: Set[int] = set()
        for u in self.node_incident.get(v, ()):
            s = self.node_to_shard.get(u)
            if s is not None and s != cur:
                cands.add(s)
        for hidx in self.node_to_hyperedges.get(v, ()):
            for s in self.hyperedge_shard_counts[hidx].keys():
                if s != cur:
                    cands.add(s)
        for uid in self.node_uids.get(v, ()):
            for s in self.uid_to_shards.get(uid, ()):
                if s != cur:
                    cands.add(s)
        return cands

    def km1_delta_prep(self, v: int, a: int) -> Tuple[int, Dict[int, int]]:
        """Per-node counts that give dKM1 for any destination b in O(1):
        dKM1(v: a -> b) = #{h : v's shard a keeps other nodes} - #{h : b already in h}."""
        c_a_still = 0
        b_hit_count: Dict[int, int] = {}
        for hidx in self.node_to_hyperedges.get(v, ()):
            counts = self.hyperedge_shard_counts[hidx]
            if counts.get(a, 0) > 1:
                c_a_still += 1
            for s in counts.keys():
                b_hit_count[s] = b_hit_count.get(s, 0) + 1
        return c_a_still, b_hit_count

    def move_gain(
        self,
        v: int,
        a: int,
        b: int,
        prep: Tuple[int, Dict[int, int]],
        include_label_gain: bool = True,
    ) -> float:
        c_a_still, b_hit_count = prep
        delta_km1 = float(c_a_still - b_hit_count.get(b, 0))
        delta_cstr = float(self.deg_within(v, a) - self.deg_within(v, b))
        km1_scale = 1.0 + self.lambda_di * float(self.di[v]) if self.lambda_di else 1.0
        gain = -(km1_scale * delta_km1 + self.lambda_c * delta_cstr)
        if include_label_gain and self.use_label_distribution:
            gain += self.lambda_l * self.label_move_gain(v, a, b)
        return gain

    def feasible_moves(
        self, v: int, a: int, dests: Set[int], prep: Tuple[int, Dict[int, int]]
    ) -> List[Tuple[float, int]]:
        """(gain, b) for every admissible single move of v out of shard a."""
        if self.deg_within(v, a) < 2:
            return []
        return [
            (self.move_gain(v, a, b, prep), b)
            for b in dests
            if self.deg_within(v, b) >= 1 and self.shard_node_counts[b] + 1 <= self.cap
        ]

    # ── Applying moves ────────────────────────────────────────────────────────

    def _apply_move(self, v: int, a: int, b: int) -> None:
        """Move v from a to b in every index except the shard sizes."""
        self.node_to_shard[v] = b
        for hidx in self.node_to_hyperedges.get(v, ()):
            counts = self.hyperedge_shard_counts[hidx]
            counts[a] -= 1
            if counts[a] == 0:
                del counts[a]
            counts[b] = counts.get(b, 0) + 1
        for w in self.node_incident.get(v, ()):
            counts = self.neighbor_shard_count.get(w)
            if counts is None:
                continue
            if a in counts:
                counts[a] -= 1
                if counts[a] == 0:
                    del counts[a]
            counts[b] = counts.get(b, 0) + 1
        col = self.label_indices.get(v)
        if col is not None and a != b:
            self.shard_label_counts[a, col] -= 1
            self.shard_label_counts[b, col] += 1
            self.shard_label_totals[a] -= 1
            self.shard_label_totals[b] += 1

    def _apply_single_move(self, v: int, a: int, b: int, tabu_until: int) -> None:
        self._apply_move(v, a, b)
        self.shard_node_counts[a] -= 1
        self.shard_node_counts[b] += 1
        self.tabu[(v, a)] = tabu_until

    # ── Passes ────────────────────────────────────────────────────────────────

    def structural_candidates(self) -> Set[int]:
        """Boundary nodes: a neighbour or a hyperedge lies in another shard."""
        candidates: Set[int] = set()
        for v, neighbors in self.node_incident.items():
            if v not in self.node_to_shard:
                continue
            shard = self.node_to_shard[v]
            if any(self.node_to_shard.get(u, shard) != shard for u in neighbors):
                candidates.add(v)
        for hidx, hyperedge in enumerate(self.global_hyperedges):
            if len(self.hyperedge_shard_counts[hidx]) > 1:
                candidates.update(v for v in hyperedge if v in self.node_to_shard)
        return candidates

    def best_single_gain(self, v: int) -> Optional[float]:
        a = self.node_to_shard[v]
        moves = self.feasible_moves(v, a, self.dest_candidates(v), self.km1_delta_prep(v, a))
        return max(gain for gain, _ in moves) if moves else None

    def _try_swap(self, v: int, a: int, dests: Set[int], prep_v, pass_idx: int) -> bool:
        if self.deg_within(v, a) < 2:
            return False
        for b in dests:
            swap_candidates = [
                u for u in self.node_incident.get(v, ())
                if self.node_to_shard.get(u) == b
            ][:REFINE_SWAP_CANDIDATE_LIMIT]
            if not swap_candidates:
                continue
            gain_v = self.move_gain(v, a, b, prep_v, include_label_gain=False)
            for u in swap_candidates:
                if self.deg_within(u, b) < 2 or self.deg_within(u, a) < 1:
                    continue
                if self.tabu.get((u, b), -1) > pass_idx:
                    continue
                gain_u = self.move_gain(u, b, a, self.km1_delta_prep(u, b), include_label_gain=False)
                label_gain = self.label_swap_gain(v, u, a, b)
                swap_gain = gain_v + gain_u
                if self.use_label_distribution:
                    swap_gain += self.lambda_l * label_gain
                if swap_gain > REFINE_MIN_GAIN:
                    # A swap keeps both shard sizes.
                    self._apply_move(v, a, b)
                    self._apply_move(u, b, a)
                    self.tabu[(v, a)] = pass_idx + REFINE_TABU_TENURE
                    self.tabu[(u, b)] = pass_idx + REFINE_TABU_TENURE
                    self.swap_moves += 1
                    self.gain_sum += swap_gain
                    self.label_gain_sum += label_gain
                    if label_gain > 0:
                        self.label_improving_moves += 1
                    return True
        return False

    def _structural_passes(self) -> Dict[str, int]:
        initial: Dict[str, int] = {}
        candidate_evaluations = 0
        passes_run = 0
        for pass_idx in range(self.max_passes):
            passes_run += 1
            structural = self.structural_candidates()
            positive = []
            for v in structural:
                gain = self.best_single_gain(v)
                if gain is not None and gain > REFINE_MIN_GAIN:
                    positive.append((gain, v))
            positive.sort(key=lambda item: (-item[0], item[1]))
            max_candidates = int(np.ceil(len(structural) * self.candidate_max_ratio))
            candidates = [v for _, v in positive[:max_candidates]]
            if pass_idx == 0:
                initial = {
                    "structural_candidates_initial": len(structural),
                    "positive_candidates_initial": len(positive),
                    "selected_candidates_initial": len(candidates),
                }
            candidate_evaluations += len(candidates)

            improved = False
            for v in candidates:
                a = self.node_to_shard[v]
                if self.tabu.get((v, a), -1) > pass_idx:
                    continue
                dests = self.dest_candidates(v)
                prep_v = self.km1_delta_prep(v, a)
                moves = self.feasible_moves(v, a, dests, prep_v)
                moves.sort(key=lambda item: (-item[0], item[1]))
                if moves and moves[0][0] > REFINE_MIN_GAIN:
                    gain, b = moves[0]
                    label_gain = self.label_move_gain(v, a, b)
                    self._apply_single_move(v, a, b, pass_idx + REFINE_TABU_TENURE)
                    self.single_moves += 1
                    self.gain_sum += gain
                    self.label_gain_sum += label_gain
                    if label_gain > 0:
                        self.label_improving_moves += 1
                    improved = True
                    continue
                if self._try_swap(v, a, dests, prep_v, pass_idx):
                    improved = True
            if not improved:
                break
        return {
            "structural_candidates_initial": initial.get("structural_candidates_initial", 0),
            "positive_candidates_initial": initial.get("positive_candidates_initial", 0),
            "selected_candidates_initial": initial.get("selected_candidates_initial", 0),
            "candidate_evaluations": candidate_evaluations,
            "passes": passes_run,
        }

    def _best_label_move(self, v: int, a: int) -> Tuple[Optional[float], Optional[int]]:
        best_gain, best_b = None, None
        for b in self.dest_candidates(v):
            if self.shard_node_counts[b] + 1 > self.cap:
                continue
            g = self.lambda_l * self.label_move_gain(v, a, b)
            if best_gain is None or g > best_gain:
                best_gain, best_b = g, b
        return best_gain, best_b

    def _class_balance_passes(self) -> None:
        """Extra passes that move boundary nodes by the class-distribution gain only."""
        for lo_pass_idx in range(self.max_passes):
            self.label_only_passes += 1
            pass_offset = self.max_passes + lo_pass_idx
            structural = self.structural_candidates()
            positive = []
            for v in structural:
                best_gain, _ = self._best_label_move(v, self.node_to_shard[v])
                if best_gain is not None and best_gain > REFINE_MIN_GAIN:
                    positive.append((best_gain, v))
            positive.sort(key=lambda item: (-item[0], item[1]))
            max_candidates = int(np.ceil(len(structural) * self.candidate_max_ratio))
            candidates = [v for _, v in positive[:max_candidates]]

            improved = False
            for v in candidates:
                a = self.node_to_shard[v]
                if self.tabu.get((v, a), -1) > pass_offset:
                    continue
                best_gain, best_b = None, None
                for b in self.dest_candidates(v):
                    if self.shard_node_counts[b] + 1 > self.cap:
                        continue
                    g = self.lambda_l * self.label_move_gain(v, a, b)
                    if g > REFINE_MIN_GAIN and (best_gain is None or g > best_gain):
                        best_gain, best_b = g, b
                if best_b is None:
                    continue
                self._apply_single_move(v, a, best_b, pass_offset + REFINE_TABU_TENURE)
                self.label_only_single_moves += 1
                self.gain_sum += best_gain
                self.label_only_gain_sum += best_gain
                self.label_gain_sum += best_gain / self.lambda_l
                self.label_improving_moves += 1
                improved = True
            if not improved:
                break

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> Tuple[Dict[int, int], Dict[str, Any]]:
        stats: Dict[str, Any] = self._structural_passes()
        if self.use_label_distribution and self.class_balance_passes:
            self._class_balance_passes()

        counts_before, counts_after = self.counts_before, self.shard_node_counts
        mean_before = float(counts_before.mean()) if counts_before.size else 0.0
        mean_after = float(counts_after.mean()) if counts_after.size else 0.0
        stats.update({
            "single_moves": self.single_moves,
            "swap_moves": self.swap_moves,
            "gain_sum": self.gain_sum,
            "label_distribution_enabled": self.use_label_distribution,
            "lambda_l": self.lambda_l,
            "label_distribution_loss_before": self.label_loss_before,
            "label_distribution_loss_after": self.label_distribution_loss(),
            "label_distribution_gain_sum": self.label_gain_sum,
            "label_distribution_improving_moves": self.label_improving_moves,
            "label_only_pass_enabled": self.class_balance_passes,
            "label_only_passes": self.label_only_passes,
            "label_only_single_moves": self.label_only_single_moves,
            "label_only_gain_sum": self.label_only_gain_sum,
            "size_cap": self.cap,
            "node_count_min_before": int(counts_before.min()) if counts_before.size else 0,
            "node_count_min_after": int(counts_after.min()) if counts_after.size else 0,
            "node_count_max_before": int(counts_before.max()) if counts_before.size else 0,
            "node_count_max_after": int(counts_after.max()) if counts_after.size else 0,
            "node_count_cv_before": (
                float(counts_before.std() / mean_before) if mean_before > 0 else 0.0
            ),
            "node_count_cv_after": (
                float(counts_after.std() / mean_after) if mean_after > 0 else 0.0
            ),
        })
        return self.node_to_shard, stats
