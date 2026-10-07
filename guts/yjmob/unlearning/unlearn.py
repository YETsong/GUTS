"""Unlearn a forget set from a trained GUTS yjmob run by retraining only the affected shards.

The affected shards (from the unlearn index over the source run's train sessions)
are rebuilt from the clean data (preprocessed without the forget set) with the
source partitions, retrained from scratch, and the run is evaluated with the
shard routing evaluation. Untouched shards keep their data (linked) and models
(best.pt linked as shard_run_best.pt). Output: shard-unlearns/<dataset>/<run>/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple

import torch
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.utils import ensure_dir, append_log, timestamp_id, relpath_for_log, estimate_shard_weight, balanced_lanes
from guts.config import load_config
from guts.yjmob.unlearning.unlearn_index import GraphSAGEUnlearnIndex

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unlearn a forget set from a trained GUTS yjmob run (retrain affected shards).")
    p.add_argument("--state_dir", required=True, help="source shard-runs directory")
    p.add_argument("--sharded-dir", dest="sharded_dir", default="", help="Sharding/ root (default: <state_dir>/Sharding)")
    p.add_argument("--forget_type", required=True, choices=["uid", "node", "edge", "path"])
    p.add_argument("--forget_set_path", required=True, help="forget set JSON")
    p.add_argument("--clean_source_dir", required=True,
                   help="preprocessed data rebuilt without the forget set (graphs of affected days are rebuilt from it)")
    p.add_argument("--index_path", default="", help="unlearn index cache (default: <sharding_dir>/graphsage_unlearn_index.pkl)")
    p.add_argument("--rebuild_index", action="store_true", help="rebuild the unlearn index")
    p.add_argument("--out_root", default="", help="output root (default: shard-unlearns/ next to state_dir)")
    p.add_argument("--run_tag", default="", help="suffix of the output directory name")
    p.add_argument("--num_gpus", type=int, default=1, help="GPUs for retraining (one shard per GPU at a time) and evaluation")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--eval_data_dir", default="", help="test data (default: the source run's data)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_sharding_dir(state_dir: str, sharded_dir_arg: str) -> str:
    if sharded_dir_arg:
        return os.path.abspath(sharded_dir_arg)
    return os.path.join(state_dir, "Sharding")


def _get_or_build_index(state_dir: str, sharding_dir: str, index_path: str, rebuild: bool, log) -> GraphSAGEUnlearnIndex:
    if not index_path:
        index_path = os.path.join(sharding_dir, "graphsage_unlearn_index.pkl")
    if os.path.exists(index_path) and not rebuild:
        log(f"Loading index: {relpath_for_log(index_path)}")
        return GraphSAGEUnlearnIndex.load(index_path)
    log(f"Building index from: {relpath_for_log(sharding_dir)}")
    idx = GraphSAGEUnlearnIndex.build(state_dir, sharding_dir=sharding_dir, log_fn=log)
    idx.save(index_path)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Forget set
# ─────────────────────────────────────────────────────────────────────────────


def _load_forget_set(forget_type: str, forget_set_path: str, idx: GraphSAGEUnlearnIndex, log) -> Tuple[List[int], str]:
    """Affected shards and a short description of a forget set JSON."""
    log(f"Loading forget set: {relpath_for_log(forget_set_path)}")
    with open(forget_set_path) as f:
        records = json.load(f)["records"]
    if forget_type == "uid":
        shards = idx.affected_shards_for_uids([int(r["uid"]) for r in records])
    elif forget_type == "node":
        shards = idx.affected_shards_for_nodes([int(r["nid"]) for r in records])
    elif forget_type == "edge":
        shards = idx.affected_shards_for_edges([(int(r["src"]), int(r["dst"])) for r in records])
    elif forget_type == "path":
        shards = idx.affected_shards_for_path_keys({
            (int(r["uid"]), int(r["day"]), int(r.get("start_time", -1)), int(r.get("end_time", -1))) for r in records
        })
    else:
        raise ValueError(f"Unknown forget_type: {forget_type}")
    log(f"forget {forget_type}: {len(records)} record(s)")
    return sorted(set(shards)), f"{forget_type}_count={len(records)}"


# ─────────────────────────────────────────────────────────────────────────────
# Shard data of the unlearned run
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_partition_edges_from_clean(partition, clean_sample) -> None:
    from guts.shard import build_shard_edges

    new_edges = build_shard_edges(
        clean_sample.edge_index, partition.shards_nodes, clean_sample.num_nodes
    )
    partition.shards_edges = {
        shard_id: new_edges.get(shard_id, torch.zeros((2, 0), dtype=torch.long))
        for shard_id in partition.shards_edges
    }


def _rebuild_target_shards_from_clean(
    cfg,
    sharding_dir: str,
    clean_source_dir: str,
    dst_processed: str,
    target_shard_ids: List[int],
    log,
) -> Dict[int, dict]:
    """Build the retrained shards' processed dirs from the clean data with the source partitions.

    Every training day's partition (partition/day{N}.pkl, not recomputed) is applied
    to the clean day graph, and the target shards are written exactly as the pipeline
    writes shard data: day graphs, train/val/test sessions, meta, rolling graphs. The
    retrained shards thus see only clean data. Returns their session counts per split.
    """
    from guts.partition.partitioner import make_partitioner
    from guts.shard import ShardBuilder
    from guts.yjmob.data.rolling_graphs import update_shard_meta_with_rolling_graphs, write_shard_rolling_graphs
    from guts.yjmob.data.shard_data import materialize_graphsage_shards
    from guts.yjmob.stage_partition import load_day_sample

    partition_dir = os.path.join(sharding_dir, "partition")
    days = sorted(
        int(name[len("day"):-len(".pkl")]) for name in os.listdir(partition_dir)
        if name.startswith("day") and name.endswith(".pkl")
    )
    if not days:
        raise FileNotFoundError(f"no partition/day*.pkl under {sharding_dir}")
    partitioner = make_partitioner(cfg)
    builder = ShardBuilder()
    by_day = {}
    for day_id in days:
        partition = partitioner.load(os.path.join(partition_dir, f"day{day_id}.pkl"))
        sample = load_day_sample(clean_source_dir, day_id)
        _refresh_partition_edges_from_clean(partition, sample)
        by_day[day_id] = {"shards": builder.build(partition, sample), "sample": sample}

    log(f"  [clean-source] building {len(target_shard_ids)} shard(s) from {relpath_for_log(clean_source_dir)} "
        f"with the source partitions of days {days[0]}-{days[-1]}")
    summary = materialize_graphsage_shards(
        source_processed_dir=clean_source_dir,
        out_dir=dst_processed,
        days=days,
        by_day=by_day,
        splits=["train", "val", "test"],
        log_fn=log,
        shard_ids=target_shard_ids,
    )

    window_size = int(cfg.temporal_graph.window_size)
    target_days = [d for d in days if d >= window_size]
    for shard_id in target_shard_ids:
        processed_dir = os.path.join(dst_processed, f"shard_{shard_id}")
        num_nodes = int(torch.load(os.path.join(processed_dir, "next_location_meta.pt"), map_location="cpu")["num_nodes"])
        entries = write_shard_rolling_graphs(processed_dir, target_days, window_size, num_nodes)
        update_shard_meta_with_rolling_graphs(processed_dir, window_size, entries)
    log(f"  [clean-source] rolling graphs written for target days {target_days} (window_size={window_size})")
    return {int(k): v["sessions"] for k, v in summary["shards"].items()}


def _link_untouched_shards(src_processed: str, dst_processed: str, shard_ids: List[int]) -> None:
    """Untouched shards keep the source data: link their processed dirs."""
    os.makedirs(dst_processed, exist_ok=True)
    for shard_id in shard_ids:
        src_shard = os.path.join(src_processed, f"shard_{shard_id}")
        if not os.path.isdir(src_shard):
            raise FileNotFoundError(f"processed dir not found: {src_shard}")
        os.symlink(src_shard, os.path.join(dst_processed, f"shard_{shard_id}"))


def _copy_partition_metadata(sharding_dir: str, dst_sharding_dir: str, log) -> None:
    """Copy Sharding/partition/ (unchanged by unlearning); the evaluation routes by it."""
    src_partition_dir = os.path.join(sharding_dir, "partition")
    dst_partition_dir = os.path.join(dst_sharding_dir, "partition")
    if not os.path.isdir(src_partition_dir):
        raise FileNotFoundError(f"partition dir not found: {src_partition_dir}")
    if not os.path.exists(dst_partition_dir):
        shutil.copytree(src_partition_dir, dst_partition_dir)
        log(f"  Copying Sharding/partition/ (day-level, unmodified) → {relpath_for_log(dst_partition_dir)} ...")


# ─────────────────────────────────────────────────────────────────────────────
# Selective copy — graphsage_runs/
# ─────────────────────────────────────────────────────────────────────────────

def _copy_graphsage_runs_selective(
    src_graphsage: str,
    dst_graphsage: str,
    affected_shard_ids: Set[int],
    log,
) -> None:
    os.makedirs(dst_graphsage, exist_ok=True)
    if not os.path.isdir(src_graphsage):
        raise FileNotFoundError(f"graphsage_runs not found: {src_graphsage}")

    shard_dirs = sorted(d for d in os.listdir(src_graphsage) if d.startswith("shard_"))
    for shard_dir in shard_dirs:
        shard_idx = int(shard_dir.split("_", 1)[1])
        src_shard = os.path.join(src_graphsage, shard_dir)
        dst_shard = os.path.join(dst_graphsage, shard_dir)
        os.makedirs(dst_shard, exist_ok=True)

        src_args = os.path.join(src_shard, "args.json")
        if os.path.exists(src_args):
            shutil.copy2(src_args, os.path.join(dst_shard, "args.json"))

        if shard_idx in affected_shard_ids:
            continue

        src_results = os.path.join(src_shard, "results.json")
        if os.path.exists(src_results):
            shutil.copy2(src_results, os.path.join(dst_shard, "results.json"))

        src_best = os.path.join(src_shard, "checkpoints", "best.pt")
        if not os.path.exists(src_best):
            raise FileNotFoundError(f"{shard_dir}: best.pt not found ({src_best})")
        dst_ckpt_dir = os.path.join(dst_shard, "checkpoints")
        os.makedirs(dst_ckpt_dir, exist_ok=True)
        dst_link = os.path.join(dst_ckpt_dir, "shard_run_best.pt")
        if os.path.lexists(dst_link):
            os.remove(dst_link)
        os.symlink(os.path.abspath(src_best), dst_link)


# ─────────────────────────────────────────────────────────────────────────────
# Retrain
# ─────────────────────────────────────────────────────────────────────────────

def retrain_shard(
    shard_run_dir: str,
    processed_dir: str,
    cuda_device: int,
    log,
) -> bool:
    args_path = os.path.join(shard_run_dir, "args.json")
    if not os.path.exists(args_path):
        log(f"  [ERROR] args.json not found: {relpath_for_log(args_path)}")
        return False

    with open(args_path) as f:
        train_args = json.load(f)

    train_args["processed_dir"] = processed_dir
    train_args["fixed_run_dir"] = shard_run_dir

    with open(args_path, "w") as f:
        json.dump(train_args, f, indent=2)

    train_py = os.path.join(_REPO_ROOT, "guts", "yjmob", "model", "train_single.py")
    skip_keys = {"out_root", "resume"}

    train_flags = []
    for key, val in train_args.items():
        if key in skip_keys:
            continue
        if isinstance(val, list):
            train_flags += [f"--{key}"] + [str(v) for v in val]
        else:
            train_flags += [f"--{key}", str(val)]

    cmd = [sys.executable, train_py] + train_flags

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    log(f"  Retraining shard @ {relpath_for_log(shard_run_dir)} (cuda:{cuda_device}) ...")
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, env=env)
    ok = result.returncode == 0
    if not ok:
        log(f"  → FAILED (exit {result.returncode})")
        for line in (result.stderr or "").strip().splitlines()[-20:]:
            log(f"  [STDERR] {line}")
    else:
        log("  → OK")
    return ok


def retrain_shards_parallel(
    dst_graphsage: str,
    dst_processed: str,
    shard_ids: List[int],
    num_gpus: int,
    log,
) -> Dict[str, bool]:
    num_gpus = max(1, num_gpus)
    shard_weights = {
        shard_idx: estimate_shard_weight(os.path.join(dst_processed, f"shard_{shard_idx}"))
        for shard_idx in shard_ids
    }
    lanes: List[List[int]] = balanced_lanes(list(shard_weights.items()), num_gpus)

    lane_desc = ", ".join(
        f"cuda:{gpu_id}→shard{lane} (load={sum(shard_weights[s] for s in lane)})"
        for gpu_id, lane in enumerate(lanes) if lane
    )
    log(f"  {num_gpus} GPUs for {len(shard_ids)} shards (load-balanced): {lane_desc}")

    results: Dict[str, bool] = {}
    log_lock = threading.Lock()

    def _locked_log(msg: str) -> None:
        with log_lock:
            log(msg)

    def _worker(gpu_id: int, my_shards: List[int]) -> None:
        for shard_idx in my_shards:
            key = f"shard_{shard_idx}"
            shard_run_dir = os.path.join(dst_graphsage, key)
            shard_processed_dir = os.path.join(dst_processed, key)
            ok = retrain_shard(shard_run_dir, shard_processed_dir, gpu_id, _locked_log)
            with log_lock:
                results[key] = ok

    threads = [
        threading.Thread(target=_worker, args=(gpu_id, lane))
        for gpu_id, lane in enumerate(lanes) if lane
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_eval_data_dir(state_dir: str, override: str = "") -> str:
    """Resolve the original processed dataset actually used by the source run."""
    candidate = override
    if not candidate:
        results_path = os.path.join(state_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                source_results = json.load(f)
            candidate = str(
                source_results.get("graphsage_preprocess", {}).get(
                    "source_processed_dir", ""
                )
            )
    if not candidate:
        config_path = os.path.join(state_dir, "config_used.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                source_config = yaml.safe_load(f) or {}
            candidate = str(
                source_config.get("data", {}).get("nextgrid_processed_dir", "")
            )
    if not candidate:
        raise ValueError(
            "Evaluation data path not found in the source run; set --eval_data_dir."
        )

    candidate = (
        os.path.abspath(candidate)
        if os.path.isabs(candidate)
        else os.path.abspath(os.path.join(_REPO_ROOT, candidate))
    )
    meta_path = os.path.join(candidate, "next_location_meta.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"evaluation metadata not found: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu")
    if not meta.get("sequence_index", {}).get("test", {}).get("parts"):
        raise ValueError(f"test sequence parts are not registered: {meta_path}")
    return candidate


def _mean_std(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return None
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1) if n > 1 else 0.0  # sample std (ddof=1)
    return {"mean": round(m, 6), "std": round(var ** 0.5, 6), "n": n}


def run_eval(out_dir: str, full_data_dir: str, num_gpus: int, log, cfg, sharding_dir: str) -> dict:
    """Shard routing evaluation of the unlearned run (logs to <out_dir>/logs/eval_log.txt)."""
    from guts.yjmob.stage_eval import run_shard_routing_eval

    eval_log_path = os.path.join(ensure_dir(os.path.join(out_dir, "logs")), "eval_log.txt")
    with open(eval_log_path, "w", encoding="utf-8") as _:
        pass

    def eval_log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        append_log(eval_log_path, line)

    log(f"Running eval ... (rolling, window_size={cfg.temporal_graph.window_size})")
    return run_shard_routing_eval(
        run_dir=out_dir,
        full_data_dir=full_data_dir,
        eval_log_path=eval_log_path,
        log=eval_log,
        window_size=cfg.temporal_graph.window_size,
        num_gpus_eval=max(1, num_gpus),
        sharding_dir=sharding_dir,
    )


def _detect_method(state_dir: str) -> str:
    """Partition strategy of the source run, for the output directory name."""
    return str(load_config(os.path.join(state_dir, "config_used.yaml")).partition.strategy).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    wall_start = time.time()

    available_gpus = torch.cuda.device_count()
    if available_gpus > 0 and args.num_gpus > available_gpus:
        print(f"[WARN] --num_gpus {args.num_gpus} > available GPUs {available_gpus} — using {available_gpus}", flush=True)
        args.num_gpus = available_gpus

    state_dir = os.path.abspath(args.state_dir)

    if args.out_root:
        out_root = args.out_root
    else:
        # state_dir = {results_root}/shard-runs/{dataset}/{run_name}
        dataset_dir_name = os.path.basename(os.path.dirname(state_dir))
        shard_runs_dir = os.path.dirname(os.path.dirname(state_dir))
        results_root = os.path.dirname(shard_runs_dir)
        out_root = os.path.join(results_root, "shard-unlearns", dataset_dir_name)

    method = _detect_method(state_dir)
    run_name = f"{timestamp_id()}_{method}_{args.forget_type}_w{max(1, args.num_gpus)}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    out_dir = ensure_dir(os.path.join(out_root, run_name))

    log_path = os.path.join(ensure_dir(os.path.join(out_dir, "logs")), "unlearn_log.txt")

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        append_log(log_path, line)

    sharding_dir = _resolve_sharding_dir(state_dir, args.sharded_dir)

    cfg = load_config(os.path.join(state_dir, "config_used.yaml"))
    rolling_window_size = int(cfg.temporal_graph.window_size)

    log("=" * 60)
    log(f"GraphSAGE UNLEARNING  type={args.forget_type}")
    log(f"  state_dir    : {relpath_for_log(state_dir)}")
    log(f"  sharding_dir : {relpath_for_log(sharding_dir)}")
    log(f"  temporal_mode: rolling (window_size={rolling_window_size})")
    log(f"  out_dir      : {relpath_for_log(out_dir)}")
    log("=" * 60)

    eval_data_dir = ""
    if not args.skip_eval:
        eval_data_dir = _resolve_eval_data_dir(state_dir, args.eval_data_dir)
        log(f"  eval_data    : {relpath_for_log(eval_data_dir)}")

    idx = _get_or_build_index(state_dir, sharding_dir, args.index_path, args.rebuild_index, log)
    log(f"Index: {idx.summary()}")

    shard_keys, desc = _load_forget_set(args.forget_type, args.forget_set_path, idx, log)

    log(f"Forget targets: {desc}")

    all_shard_ids = list(range(idx.num_shards))
    untouched_shard_ids = [s for s in all_shard_ids if s not in set(shard_keys)]
    log(f"Retrain target shards: {len(shard_keys)}/{idx.num_shards} -> {shard_keys}")
    log(f"Untouched shards (original checkpoints kept): {len(untouched_shard_ids)}/{idx.num_shards} -> {untouched_shard_ids}")

    if not shard_keys:
        log("[WARN] No affected shards. Nothing to unlearn.")
        sys.exit(0)

    log(f"Copying graphsage_runs/ → {relpath_for_log(out_dir)}/graphsage_runs/ (selective) ...")
    src_graphsage = os.path.join(state_dir, "graphsage_runs")
    dst_graphsage = os.path.join(out_dir, "graphsage_runs")
    _copy_graphsage_runs_selective(src_graphsage, dst_graphsage, set(shard_keys), log)

    src_processed = os.path.join(sharding_dir, "processed")
    dst_processed = os.path.join(out_dir, "Sharding", "processed")
    log(f"Linking {len(untouched_shard_ids)} untouched shard(s) → {relpath_for_log(dst_processed)} ...")
    _link_untouched_shards(src_processed, dst_processed, untouched_shard_ids)

    _copy_partition_metadata(sharding_dir, os.path.join(out_dir, "Sharding"), log)

    retrain_results: Dict[str, bool] = {}
    for shard_idx in shard_keys:
        shard_run_dir = os.path.join(dst_graphsage, f"shard_{shard_idx}")
        if not os.path.isdir(shard_run_dir):
            raise FileNotFoundError(f"[shard_{shard_idx}] graphsage_runs dir not found: {shard_run_dir}")

    log(f"\nRebuilding the {len(shard_keys)} retrain shard(s) from the clean source ...")
    new_sessions = _rebuild_target_shards_from_clean(
        cfg, sharding_dir, args.clean_source_dir, dst_processed, list(shard_keys), log,
    )
    for shard_idx in shard_keys:
        old_meta = torch.load(os.path.join(src_processed, f"shard_{shard_idx}", "next_location_meta.pt"), map_location="cpu")
        before = {split: int(old_meta["sequence_index"][split]["sessions"]) for split in ("train", "val", "test")}
        log(f"  [shard_{shard_idx}] sessions train/val/test: "
            + "  ".join(f"{before[sp]}→{new_sessions[shard_idx][sp]}" for sp in ("train", "val", "test")))
    ready_shard_ids: List[int] = list(shard_keys)

    if ready_shard_ids:
        log(f"\nRetraining {len(ready_shard_ids)} shard(s) in parallel across {args.num_gpus} GPU(s) ...")
        retrain_results.update(
            retrain_shards_parallel(
                dst_graphsage, dst_processed, ready_shard_ids, args.num_gpus, log,
            )
        )

    failed = [k for k, ok in retrain_results.items() if not ok]
    if failed:
        log(f"[ERROR] Retraining failed for: {failed}")
        sys.exit(1)

    eval_results = None
    agg_eval_ms = None
    if not args.skip_eval:
        t0_eval = time.perf_counter()
        eval_results = run_eval(
            out_dir, eval_data_dir, args.num_gpus, log,
            cfg=cfg, sharding_dir=os.path.join(out_dir, "Sharding"),
        )
        agg_eval_ms = round((time.perf_counter() - t0_eval) * 1000)

    train_ms_values, predict_ms_values = [], []
    for shard_key in retrain_results:
        shard_result_path = os.path.join(dst_graphsage, shard_key, "results.json")
        if not os.path.exists(shard_result_path):
            continue
        with open(shard_result_path) as f:
            shard_result = json.load(f)
        shard_timing = shard_result.get("timing_ms") or {}
        if shard_timing.get("train") is not None:
            train_ms_values.append(shard_timing["train"] + (shard_timing.get("val_eval") or 0))
        if shard_timing.get("predict") is not None:
            predict_ms_values.append(shard_timing["predict"])

    retrain_wall_s = (
        round(sum(train_ms_values) / 1000 / max(1, args.num_gpus), 1)
        if train_ms_values else None
    )
    if retrain_wall_s is not None:
        log(f"  GraphSAGE training done  elapsed={retrain_wall_s:.1f}s "
            f"(sum(per-shard train)/{args.num_gpus}, excluding scheduling imbalance)")

    wall_time = round(time.time() - wall_start, 1)
    results = {
        "experiment_type": f"graphsage_{args.forget_type}_unlearn",
        "source_state_dir": state_dir,
        "source_sharding_dir": sharding_dir,
        "out_dir": out_dir,
        "wall_time_s": wall_time,
        "eval_data_dir": eval_data_dir or None,
        "forget_request": {
            "forget_type": args.forget_type,
            "description": desc,
            "affected_shards": list(shard_keys),
        },
        "retrain_results": retrain_results,
        "eval": eval_results,
        "timing_ms": {
            "per_shard_train": _mean_std(train_ms_values) if train_ms_values else None,
            "per_shard_predict": _mean_std(predict_ms_values) if predict_ms_values else None,
            "agg_eval_test": agg_eval_ms,
            "retrain_wall_s": retrain_wall_s,
        },
    }

    results_path = os.path.join(out_dir, "results_unlearn.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log("=" * 60)
    log(f"UNLEARN COMPLETE  wall={wall_time:.1f}s")
    log(f"  forget_type      : {args.forget_type}")
    log(f"  affected shards  : {shard_keys}")
    if eval_results:
        r = eval_results.get("routable", {})
        acc1 = r.get("acc@1", "?")
        mrr = r.get("mrr", "?")
        log(f"  [routable] ACC@1={acc1}  MRR={mrr}")
    log(f"  results → {relpath_for_log(results_path)}")
    log("=" * 60)


if __name__ == "__main__":
    main()
