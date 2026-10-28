"""STEP 1 of guts/yjmob/pipeline.py: run the GUTS partitioner over the training days.

The partitioning algorithm itself is guts/method/partition/partitioner.py; this file runs
it for yjmob: loads each day graph, fits the days in parallel processes, applies
temporal subgraph assignment in day order, saves/reuses partition/day{N}.pkl
(reused only when every day exists), builds the shard subgraphs and writes the
refinement diagnostics.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.multiprocessing as mp

from guts.method.partition.partitioner import GUTSPartitioner
from guts.method.shard import ShardBuilder
from guts.method.utils import compact_json, ensure_dir, load_pickle, relpath_for_log, save_pickle
from guts.yjmob.data.train_trajs import TrajectoryExtractor, load_grid_info


@dataclass
class TemporalSample:
    """
    Typed container for one day graph sample.

    Field shapes:
        x           : FloatTensor [num_nodes, in_dim]
        edge_index  : LongTensor  [2, num_edges]
        edge_weight : FloatTensor [num_edges]   (ones if absent)
    """
    day: int
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def in_dim(self) -> int:
        return int(self.x.size(1))


def make_partitioner(cfg) -> GUTSPartitioner:
    """GUTS partitioner for the yjmob days: KaHyPar seed training.seed, temporal subgraph
    assignment over temporal_graph.window_size days on the grid of data.nextgrid_processed_dir."""
    return GUTSPartitioner(
        cfg.partition,
        seed=int(cfg.training.seed),
        window_size=cfg.temporal_graph.window_size,
        grid_info=load_grid_info(cfg.data.nextgrid_processed_dir),
    )


def load_day_sample(source_processed_dir: str, day_id: int) -> TemporalSample:
    """Build a partitioner-compatible sample from GraphSAGE next-grid daily graphs."""
    graph_path = os.path.join(
        source_processed_dir,
        "daily_graphs",
        f"day_{int(day_id):04d}.pt",
    )
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Next-grid daily graph not found: {relpath_for_log(graph_path)}")
    graph = torch.load(graph_path, map_location="cpu")
    x = graph["x"].float()
    edge_index = graph["edge_index"].long()
    edge_weight = graph.get("edge_weight")
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    else:
        edge_weight = edge_weight.float()
    return TemporalSample(
        day=int(day_id),
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


def _partition_fit_worker(args):
    cfg, day_id, nextgrid_source_dir = args
    partitioner = make_partitioner(cfg)
    sample = load_day_sample(nextgrid_source_dir, day_id)
    tc = TrajectoryExtractor(cfg.data.nextgrid_processed_dir).extract(day_id)
    import contextlib
    import io
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        partition, tc = partitioner.fit_unassigned(sample, tc)
    for _line in _buf.getvalue().splitlines():
        if _line.strip():
            print(f"[day {day_id}] {_line}", flush=True)
    return day_id, partition, tc, sample


def source_days(source_processed_dir: str):
    meta_path = os.path.join(source_processed_dir, "next_location_meta.pt")
    meta = torch.load(meta_path, map_location="cpu")
    if "days" in meta:
        return sorted(int(day) for day in meta["days"])
    day_dir = os.path.join(source_processed_dir, "daily_graphs")
    out = []
    for name in os.listdir(day_dir):
        if name.startswith("day_") and name.endswith(".pt"):
            out.append(int(name[len("day_"):-len(".pt")]))
    return sorted(out)


def step_build_day(partition, sample):
    """Build shard subgraphs for one day (no ghost nodes); returns them and the time in ms."""
    t0 = time.time()
    shards = ShardBuilder().build(partition, sample)
    return shards, round((time.time() - t0) * 1000)


def _save_refinement_metrics(sharding_dir: str, partitions: dict, cfg) -> dict:
    """Write human-readable GUTS refinement diagnostics for one run."""
    daily = {}
    for day, partition in sorted(partitions.items()):
        metrics = partition.quality_metrics
        values = {
            key.removeprefix("refine_"): (
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in metrics.items()
            if key.startswith("refine_")
        }
        if values:
            daily[str(day)] = values

    config = {
        "candidate_max_ratio": float(cfg.partition.refine_candidate_max_ratio),
        "lambda_di": float(cfg.partition.lambda_di),
        "lambda_c": float(cfg.partition.lambda_c),
        "lambda_l": float(cfg.partition.lambda_l),
        "max_passes": int(cfg.partition.refine_max_passes),
    }

    def values_for(key):
        return [
            float(day_metrics[key])
            for day_metrics in daily.values()
            if key in day_metrics
        ]

    aggregate = {"days": len(daily)}
    for key in (
        "structural_candidates_initial",
        "positive_candidates_initial",
        "selected_candidates_initial",
        "candidate_evaluations",
        "single_moves",
        "swap_moves",
        "gain_sum",
        "wall_time_s",
    ):
        values = values_for(key)
        if values:
            aggregate[f"{key}_total"] = sum(values)
            aggregate[f"{key}_mean"] = sum(values) / len(values)
    for key in (
        "shards_per_traj_before",
        "shards_per_traj_after",
        "cstr_before",
        "cstr_after",
        "pre_refine_max_shard_size",
        "node_count_min_before",
        "node_count_min_after",
        "node_count_max_before",
        "node_count_max_after",
        "node_count_cv_before",
        "node_count_cv_after",
    ):
        values = values_for(key)
        if values:
            aggregate[f"{key}_mean"] = sum(values) / len(values)

    payload = {"config": config, "daily": daily, "aggregate": aggregate}
    path = os.path.join(sharding_dir, "refinement_metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(compact_json(payload))
    return {
        "metrics_file": "Sharding/refinement_metrics.json",
        "config": config,
        "aggregate": aggregate,
    }



def run_partition_stage(cfg, train_days: List[int], source_dir: str, sharding_dir: str, external: bool, log) -> dict:
    """Partition every training day (or reuse a complete saved set) and build the shard subgraphs.

    Returns ``by_day`` ({day: {"shards", "sample"}}, the input of STEP 2), the partitions,
    the partition/build times and the refinement summary.
    """
    t0_stage = time.time()
    partitioner = make_partitioner(cfg)

    partition_dir = os.path.join(sharding_dir, "partition")
    processed_dir = os.path.join(sharding_dir, "processed")

    # Partitions are reused only when every training day was saved. Temporal subgraph
    # assignment processes the days in order on one partitioner, so an incomplete set
    # (an interrupted run) is never resumed: all days are partitioned again.
    partition_paths = {d: os.path.join(partition_dir, f"day{d}.pkl") for d in train_days}
    has_processed  = os.path.isdir(processed_dir)
    skip_partition = all(os.path.exists(path) for path in partition_paths.values())
    skip_materialize = has_processed
    if not skip_partition and has_processed:
        raise RuntimeError(
            f"Partitions in {relpath_for_log(partition_dir)} are incomplete but "
            f"{relpath_for_log(processed_dir)} already exists. Remove "
            f"{relpath_for_log(sharding_dir)} and run again."
        )

    if external:
        skip_labels = []
        if skip_partition:   skip_labels.append("STEP 1(Partition)")
        if skip_materialize: skip_labels.append("STEP 2(Materialize)")
        if skip_labels:
            log(f"  [sharded-dir] using existing data in {relpath_for_log(sharding_dir)} -> skipping {', '.join(skip_labels)}")
        else:
            log(f"  [sharded-dir] {relpath_for_log(sharding_dir)} given — running from the first missing step")

    # ── Per-day Partition + Build ─────────────────────────────────────
    by_day: dict = {}
    all_partitions: dict = {}

    if skip_partition:
        log(f"STEP 1/3 partition: reusing the saved partitions of {len(train_days)} days")
        ensure_dir(partition_dir)
        t0_skip = time.time()
        partition_ms_total = 0
        build_ms_total = 0
        for day_id in train_days:
            sample = load_day_sample(source_dir, day_id)
            partition = load_pickle(partition_paths[day_id])
            shards, day_build_ms = step_build_day(partition, sample)
            build_ms_total += day_build_ms
            by_day[day_id]       = {"shards": shards, "sample": sample}
            all_partitions[day_id] = partition
        log(f"  shard subgraphs rebuilt  {time.time()-t0_skip:.0f}s")
    else:
        log(f"STEP 1/3 partition ({len(train_days)} days, K={cfg.partition.num_shards})")
        ensure_dir(partition_dir)
        partition_ms_total = 0
        build_ms_total = 0

        n_stale = sum(os.path.exists(path) for path in partition_paths.values())
        if n_stale:
            log(f"  [WARN] incomplete partitions ({n_stale}/{len(train_days)} days) — "
                f"partitioning all days again")

        fit_results: Dict[int, tuple] = {}
        num_workers = min(len(train_days), 8, max(1, (os.cpu_count() or 4) - 2))
        log(f"  partition fit: {len(train_days)} days on {num_workers} worker processes")
        t0_parallel = time.time()
        worker_args = [(cfg, d, source_dir) for d in train_days]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            for day_id, partition, tc, sample in executor.map(_partition_fit_worker, worker_args):
                fit_results[day_id] = (partition, tc, sample)
        partition_ms_total += round((time.time() - t0_parallel) * 1000)
        log(f"  partition fit done  {time.time()-t0_parallel:.0f}s")

        # Temporal subgraph assignment must see the days in order.
        t0_assign = time.time()
        for day_id in train_days:
            partition, tc, sample = fit_results[day_id]
            partition = partitioner.subgraph_assign(partition, tc)
            save_pickle(partition, partition_paths[day_id])
            shards, day_build_ms = step_build_day(partition, sample)
            build_ms_total += day_build_ms
            by_day[day_id]       = {"shards": shards, "sample": sample}
            all_partitions[day_id] = partition
        log(f"  subgraph assignment done  {len(train_days)}/{len(train_days)} days  {time.time() - t0_assign:.0f}s")

    refinement_summary = _save_refinement_metrics(
        sharding_dir, all_partitions, cfg
    )
    agg = refinement_summary["aggregate"]
    if "shards_per_traj_after_mean" in agg:
        log(f"  partition done  mean over {agg['days']} days: shards/traj {agg['shards_per_traj_after_mean']:.3f}"
            f"  cstr {agg['cstr_after_mean']:.3f}  {time.time() - t0_stage:.0f}s")


    return {
        "by_day": by_day,
        "partitions": all_partitions,
        "partition_ms": partition_ms_total,
        "build_ms": build_ms_total,
        "refinement": refinement_summary,
    }
