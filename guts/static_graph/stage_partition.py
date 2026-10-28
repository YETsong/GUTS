"""STEP 1 of guts/static_graph/pipeline.py: partition the train graph with GUTS and build the shards.

The partitioning algorithm itself is guts/method/partition/partitioner.py; this file
turns a static graph into its input (one size-2 traj per edge), runs it once, saves
partition.pkl and builds the shard subgraphs.
"""

from __future__ import annotations

import os
import random
import time

from guts.method.partition.partitioner import GUTSPartitioner
from guts.method.partition.trajectories import Trajectories
from guts.method.shard import ShardBuilder
from guts.method.utils import save_pickle


def edges_to_trajectories(sample) -> Trajectories:
    """One traj (a size-2 hyperedge) per directed edge of `sample`; static graphs carry no users.

    The edges are visited in a random.sample order and deduplicated through a set, so the
    traj order follows the global `random` state (the run seed)."""
    src, dst = sample.edge_index.tolist()
    seen = set()
    for i in random.sample(range(len(src)), len(src)):
        seen.add((src[i], dst[i]))
    trajs = list(seen)
    return Trajectories(trajs=trajs, traj_uids=[None] * len(trajs))


def make_partitioner(cfg) -> GUTSPartitioner:
    """GUTS partitioner for one static graph: KaHyPar seed training.seed (the run seed, already
    resolved), no grid; a single fit, so §4.4 keeps the shard ids."""
    return GUTSPartitioner(cfg.partition, seed=int(cfg.training.seed))


def run_partition_stage(cfg, train_sample, run_dir: str, log) -> dict:
    """Partition the train graph (labels drive the label-distribution refinement), save
    <run_dir>/partition.pkl and build the shard subgraphs.

    Returns the partition, the shards and the partition/build times (ms)."""
    t0_partition = time.perf_counter()
    t0 = time.time()
    partition = make_partitioner(cfg).fit(train_sample, edges_to_trajectories(train_sample),
                                          node_labels=train_sample.y)
    qm = partition.quality_metrics
    log(f"  partition done  balance {qm['balance']:.4f}  fairness {qm['fairness']:.4f}  {time.time()-t0:.1f}s")
    partition_ms = round((time.perf_counter() - t0_partition) * 1000)
    save_pickle(partition, os.path.join(run_dir, "partition.pkl"))

    log("STEP 2/4 shard subgraphs")
    t0_build = time.perf_counter()
    shards = ShardBuilder().build(partition, train_sample)
    build_ms = round((time.perf_counter() - t0_build) * 1000)
    sizes = [int(s.uids.numel()) for s in shards.values()]
    log(f"  {len(shards)} shards built  nodes {min(sizes)}-{max(sizes)}")
    return {"partition": partition, "shards": shards, "partition_ms": partition_ms, "build_ms": build_ms}
