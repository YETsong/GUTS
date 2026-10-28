"""Common shard construction: one PyG subgraph per shard from a partition.

A sample is any graph container with `x` [num_nodes, in_dim], `in_dim` and
`edge_index` (yjmob TemporalSample, static-graph CoraSample)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from guts.method.partition.partitioner import GUTSPartition


class ShardBuilder:
    """
    Build a PyG Data object (owned nodes + intra-shard edges) for each shard.
    """

    # ── Main build method ─────────────────────────────────────────────────────

    def build(
        self,
        partition: GUTSPartition,
        sample,
    ) -> Dict[int, Data]:
        """
        Build a Dict[shard_id, Data] from a GUTSPartition + the partitioned sample.

        Args:
            partition: GUTSPartition from GUTSPartitioner.fit().
            sample   : Sample whose x is being split.

        Returns:
            Dict mapping shard_id → PyG Data (intra-shard, no ghost nodes).
        """
        shards: Dict[int, Data] = {}
        for shard_id, global_nodes in partition.shards_nodes.items():
            shards[shard_id] = self._build_one_shard(
                global_nodes=global_nodes,
                shard_edge_index=partition.shards_edges[shard_id],
                sample=sample,
            )
        return shards

    # ── Per-shard construction ────────────────────────────────────────────────

    def _build_one_shard(
        self,
        global_nodes: List[int],
        shard_edge_index: torch.Tensor,
        sample,
    ) -> Data:
        """
        Build the PyG Data object for a single shard.

        Steps:
            1. Extract node features for global_nodes → x_local [n_local, in_dim].
            2. train_mask = all True.
            3. Store uids = tensor of global node IDs for back-projection.

        Edge case — empty shard:
            If global_nodes is empty, return a minimal Data with no nodes or edges.

        Args:
            global_nodes     : Sorted list of global node IDs in this shard.
            shard_edge_index : Intra-shard edge_index (LOCAL IDs from GUTSPartitioner).
            sample           : Source sample.

        Returns:
            PyG Data object.
        """
        if not global_nodes:
            return Data(
                x=torch.zeros((0, sample.in_dim), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                train_mask=torch.zeros(0, dtype=torch.bool),
                uids=torch.zeros(0, dtype=torch.long),
            )

        # global_nodes is already sorted (from _build_shards_nodes)
        node_idx = torch.tensor(global_nodes, dtype=torch.long)
        return Data(
            x=sample.x[node_idx],                        # [n_local, in_dim]
            edge_index=shard_edge_index,
            train_mask=torch.ones(len(global_nodes), dtype=torch.bool),
            uids=node_idx.clone(),                       # [n_local] global IDs
        )


def build_shard_edges(
    edge_index: torch.Tensor,
    shards_nodes: Dict[int, List[int]],
    num_nodes: int,
) -> Dict[int, torch.Tensor]:
    """shard id -> edges between the shard's nodes, relabelled to local ids (the
    position in the sorted shards_nodes list). Also used by unlearning to rebuild
    shard edges from a cleaned graph."""
    from torch_geometric.utils import subgraph
    shards_edges: Dict[int, torch.Tensor] = {}
    for shard_id, global_nodes in shards_nodes.items():
        if not global_nodes:
            shards_edges[shard_id] = torch.zeros((2, 0), dtype=torch.long)
            continue
        node_tensor = torch.tensor(global_nodes, dtype=torch.long)
        sub_ei, _ = subgraph(node_tensor, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        shards_edges[shard_id] = sub_ei
    return shards_edges
