"""Unlearn a forget set from a trained GUTS yjmob run by retraining only the affected shards.

The affected shards (from the unlearn index over the source run's train sessions)
are rebuilt from the clean data (preprocessed without the forget set) with the
source partitions, retrained from scratch, and the run is evaluated with the
shard routing evaluation. Untouched shards keep their data (linked) and models
(best.pt linked as shard_run_best.pt). Output: shard-unlearns/<dataset>/<run>/.

Console and <out>/run_log.txt: progress, one line per retrained shard of the GPU 0
lane, and the test result; the evaluation details go to <out>/eval_log.txt.

--affected_only only counts the affected shards (no clean data, no retraining) and
writes <state_dir>/affected_shards/<forget set file name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.method.reporting import mean_std
from guts.method.utils import (ensure_dir, timestamp_id, timestamp_line, relpath_for_log, repo_relpath, load_pickle,
                               compact_json, append_log_synced)
from guts.yjmob.data.shard_data import estimate_shard_weight
from guts.yjmob.model.trainer import resumable_checkpoint, stored_args
from guts.yjmob.config import load_yjmob_config
from guts.yjmob.stage_train import format_shard_line
from guts.yjmob.unlearning.affected_shards import AffectedShardIndex, affected_shards, load_forget_records

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unlearn a forget set from a trained GUTS yjmob run (retrain affected shards).")
    p.add_argument("--state_dir", required=True, help="source shard-runs directory")
    p.add_argument("--sharded-dir", dest="sharded_dir", default="", help="Sharding/ root (default: <state_dir>/Sharding)")
    p.add_argument("--forget_type", required=True, choices=["user", "node", "edge", "traj"])
    p.add_argument("--forget_set_path", required=True, help="forget set JSON")
    p.add_argument("--clean_source_dir", default="",
                   help="preprocessed data rebuilt without the forget set (the affected shards are rebuilt from it); "
                        "required unless --affected_only")
    p.add_argument("--out_root", default="", help="output root (default: shard-unlearns/ next to state_dir)")
    p.add_argument("--run_name", default="", help="output directory name (default: <time>_<method>_<forget_type>_w<G>)")
    p.add_argument("--run_tag", default="", help="suffix of the auto-generated output directory name")
    p.add_argument("--num_gpus", type=int, default=1, help="GPUs for retraining (one shard per GPU at a time) and evaluation")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--eval_data_dir", default="", help="test data (default: the source run's data)")
    p.add_argument("--affected_only", action="store_true",
                   help="only count the affected shards and write <state_dir>/affected_shards/<forget set>.json")
    args = p.parse_args(argv)
    if not args.affected_only and not args.clean_source_dir:
        p.error("--clean_source_dir is required unless --affected_only")
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_sharding_dir(state_dir: str, sharded_dir_arg: str) -> str:
    if sharded_dir_arg:
        return os.path.abspath(sharded_dir_arg)
    return os.path.join(state_dir, "Sharding")


def _build_index(sharding_dir: str, log) -> AffectedShardIndex:
    return AffectedShardIndex.build(sharding_dir, log_fn=log)


# ─────────────────────────────────────────────────────────────────────────────
# Forget set
# ─────────────────────────────────────────────────────────────────────────────


def _load_forget_set(forget_type: str, forget_set_path: str, idx: AffectedShardIndex, log) -> Tuple[List[int], str]:
    """Affected shards and a short description of a forget set JSON."""
    records = load_forget_records(forget_set_path)
    shards = idx.affected_shards_for_records(forget_type, records)
    return shards, f"{forget_type}_count={len(records)}"


def _write_affected_only(state_dir: str, args) -> None:
    """--affected_only: affected shards of the forget set for the source run; nothing is retrained."""
    sharding_dir = _resolve_sharding_dir(state_dir, args.sharded_dir)
    result = affected_shards(sharding_dir, args.forget_type, args.forget_set_path)
    name = os.path.splitext(os.path.basename(args.forget_set_path))[0] + ".json"
    out_path = os.path.join(ensure_dir(os.path.join(state_dir, "affected_shards")), name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(compact_json(result))
    print(f"affected shards: {result['num_affected']}/{result['num_shards']} "
          f"({args.forget_type}, {result['num_records']} record(s)) → {relpath_for_log(out_path)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shard data of the unlearned run
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_partition_edges_from_clean(partition, clean_sample) -> None:
    from guts.method.shard import build_shard_edges

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
    from guts.method.shard import ShardBuilder
    from guts.yjmob.data.rolling_graphs import update_shard_meta_with_rolling_graphs, write_shard_rolling_graphs
    from guts.yjmob.data.shard_data import materialize_graphsage_shards
    from guts.yjmob.stage_partition import load_day_sample

    partition_dir = os.path.join(sharding_dir, "partition")
    days = sorted(
        int(name[len("day"):-len(".pkl")]) for name in os.listdir(partition_dir)
        if name.startswith("day") and name.endswith(".pkl")
    )
    if not days:
        raise FileNotFoundError(f"no partition/day*.pkl under {relpath_for_log(sharding_dir)}")
    builder = ShardBuilder()
    by_day = {}
    for day_id in days:
        partition = load_pickle(os.path.join(partition_dir, f"day{day_id}.pkl"))
        sample = load_day_sample(clean_source_dir, day_id)
        _refresh_partition_edges_from_clean(partition, sample)
        by_day[day_id] = {"shards": builder.build(partition, sample), "sample": sample}

    t0 = time.time()
    log(f"  rebuilding {len(target_shard_ids)} shards from the clean data (source partitions of days {days[0]}-{days[-1]})")
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
    log(f"  rebuild done (shard data + rolling graphs)  {time.time() - t0:.0f}s")
    return {int(k): v["sessions"] for k, v in summary["shards"].items()}


def _link_untouched_shards(src_processed: str, dst_processed: str, shard_ids: List[int]) -> None:
    """Untouched shards keep the source data: link their processed dirs."""
    os.makedirs(dst_processed, exist_ok=True)
    for shard_id in shard_ids:
        src_shard = os.path.join(src_processed, f"shard_{shard_id}")
        if not os.path.isdir(src_shard):
            raise FileNotFoundError(f"processed dir not found: {relpath_for_log(src_shard)}")
        os.symlink(src_shard, os.path.join(dst_processed, f"shard_{shard_id}"))


def _copy_partition_metadata(sharding_dir: str, dst_sharding_dir: str) -> None:
    """Copy Sharding/partition/ (unchanged by unlearning); the evaluation routes by it."""
    src_partition_dir = os.path.join(sharding_dir, "partition")
    dst_partition_dir = os.path.join(dst_sharding_dir, "partition")
    if not os.path.isdir(src_partition_dir):
        raise FileNotFoundError(f"partition dir not found: {relpath_for_log(src_partition_dir)}")
    if not os.path.exists(dst_partition_dir):
        shutil.copytree(src_partition_dir, dst_partition_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Selective copy — graphsage_runs/
# ─────────────────────────────────────────────────────────────────────────────

def _copy_graphsage_runs_selective(
    src_graphsage: str,
    dst_graphsage: str,
    affected_shard_ids: Set[int],
) -> None:
    os.makedirs(dst_graphsage, exist_ok=True)
    if not os.path.isdir(src_graphsage):
        raise FileNotFoundError(f"graphsage_runs not found: {relpath_for_log(src_graphsage)}")

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
            raise FileNotFoundError(f"{relpath_for_log(shard_dir)}: best.pt not found ({src_best})")
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
    resume: bool = False,
) -> bool:
    """Retrain one shard with its args.json; with resume, continue from checkpoints/last.pt when
    trainer.resumable_checkpoint allows it (else from scratch)."""
    args_path = os.path.join(shard_run_dir, "args.json")
    if not os.path.exists(args_path):
        log(f"  [ERROR] args.json not found: {relpath_for_log(args_path)}")
        return False

    with open(args_path) as f:
        train_args = json.load(f)

    train_args["processed_dir"] = processed_dir
    train_args["fixed_run_dir"] = shard_run_dir

    with open(args_path, "w") as f:
        json.dump(stored_args(train_args), f, indent=2)

    train_py = os.path.join(_REPO_ROOT, "guts", "yjmob", "model", "train_single.py")
    skip_keys = {"out_root", "resume"}

    # args.json holds the total epoch count; with --resume, --epochs counts the remaining epochs.
    flag_args = dict(train_args)
    resume_ckpt = (resumable_checkpoint(shard_run_dir) if resume else None) or ""
    if resume_ckpt:
        done_epochs = int(torch.load(resume_ckpt, map_location="cpu")["epoch"])
        flag_args["epochs"] = int(train_args["epochs"]) - done_epochs

    train_flags = []
    for key, val in flag_args.items():
        if key in skip_keys:
            continue
        if isinstance(val, list):
            train_flags += [f"--{key}"] + [str(v) for v in val]
        else:
            train_flags += [f"--{key}", str(val)]
    if resume_ckpt:
        train_flags += ["--resume", resume_ckpt]

    cmd = [sys.executable, train_py] + train_flags

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    # The trainer's own lines are in its train_log.txt; stdout (a summary line) is dropped.
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    ok = result.returncode == 0
    if not ok:
        log(f"  [ERROR] {os.path.basename(shard_run_dir)} retraining failed (exit {result.returncode}, cuda:{cuda_device})")
        for line in (result.stderr or "").strip().splitlines()[-20:]:
            log(f"  [STDERR] {line}")
    return ok


def retrain_shards_parallel(
    dst_graphsage: str,
    dst_processed: str,
    shard_ids: List[int],
    num_gpus: int,
    log,
    resume: bool = False,
) -> Dict[str, bool]:
    """Retrain the shards on num_gpus GPUs: one thread per GPU takes the next shard from a
    shared queue (largest first) when its GPU is free."""
    num_gpus = max(1, num_gpus)
    pending: "queue.Queue[int]" = queue.Queue()
    for shard_idx in sorted(shard_ids, reverse=True,
                            key=lambda k: estimate_shard_weight(os.path.join(dst_processed, f"shard_{k}"))):
        pending.put(shard_idx)

    log(f"  retraining {len(shard_ids)} shards on {num_gpus} GPU(s)"
        + ("; shards go to the next free GPU, largest first; lines below are the shards trained on GPU 0"
           if num_gpus > 1 else ""))

    results: Dict[str, bool] = {}
    log_lock = threading.Lock()
    t0 = time.time()

    def _locked_log(msg: str) -> None:
        with log_lock:
            log(msg)

    def _worker(gpu_id: int) -> None:
        done = 0
        while True:
            try:
                shard_idx = pending.get_nowait()
            except queue.Empty:
                return
            done += 1
            key = f"shard_{shard_idx}"
            shard_run_dir = os.path.join(dst_graphsage, key)
            shard_processed_dir = os.path.join(dst_processed, key)
            ok = retrain_shard(shard_run_dir, shard_processed_dir, gpu_id, _locked_log, resume=resume)
            with log_lock:
                results[key] = ok
                if ok and gpu_id == 0:
                    with open(os.path.join(shard_run_dir, "results.json")) as f:
                        result = json.load(f)
                    lane = f"GPU0 #{done}, " if num_gpus > 1 else f"{done}/{len(shard_ids)}, "
                    log(f"  {format_shard_line(shard_idx, result, f'{lane}all {len(results)}/{len(shard_ids)}')}")

    threads = [threading.Thread(target=_worker, args=(gpu_id,)) for gpu_id in range(min(num_gpus, len(shard_ids)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"  retraining done  {sum(results.values())}/{len(shard_ids)} shards  {time.time() - t0:.0f}s")
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
        raise FileNotFoundError(f"evaluation metadata not found: {relpath_for_log(meta_path)}")
    meta = torch.load(meta_path, map_location="cpu")
    if not meta.get("sequence_index", {}).get("test", {}).get("parts"):
        raise ValueError(f"test sequence parts are not registered: {relpath_for_log(meta_path)}")
    return candidate


def run_eval(out_dir: str, full_data_dir: str, num_gpus: int, log, cfg, sharding_dir: str) -> dict:
    """Shard routing evaluation of the unlearned run (details in <out_dir>/eval_log.txt); logs the test result."""
    from guts.yjmob.stage_eval import result_line, run_shard_routing_eval

    eval_log_path = os.path.join(out_dir, "eval_log.txt")
    with open(eval_log_path, "w", encoding="utf-8") as _:
        pass

    log("shard routing evaluation (test)")
    t0 = time.time()
    result = run_shard_routing_eval(
        run_dir=out_dir,
        full_data_dir=full_data_dir,
        eval_log_path=eval_log_path,
        window_size=cfg.temporal_graph.window_size,
        num_gpus_eval=max(1, num_gpus),
        sharding_dir=sharding_dir,
    )
    log(f"  test  {result_line(result)}  {time.time() - t0:.0f}s")
    return result


def _detect_method(state_dir: str) -> str:
    """Partition strategy of the source run, for the output directory name."""
    return str(load_yjmob_config(os.path.join(state_dir, "config_used.yaml")).partition.strategy).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Request and results files (shared with unlearn_batch_retry.py)
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_FILENAME = "unlearn_request.json"


def _deferred_run_logger():
    """log(msg) like run_logger, but lines stay in memory until log.open(run_log_path) starts the run log,
    so a run that stops before choosing the shards leaves no output directory."""
    state = {"path": None, "pending": []}

    def log(message: str) -> None:
        line = timestamp_line(message)
        print(line, flush=True)
        if state["path"] is None:
            state["pending"].append(line)
        else:
            append_log_synced(state["path"], line)

    def open_log(run_log_path: str) -> None:
        state["path"] = run_log_path
        for line in state["pending"]:
            append_log_synced(run_log_path, line)
        state["pending"].clear()

    log.open = open_log
    return log


def save_unlearn_request(out_dir: str, request: dict) -> None:
    with open(os.path.join(out_dir, REQUEST_FILENAME), "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)


def load_unlearn_request(out_dir: str) -> dict:
    path = os.path.join(out_dir, REQUEST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{REQUEST_FILENAME} not found in {relpath_for_log(out_dir)} (not an unlearn_batch.py output directory)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_unlearn_results(
    out_dir: str,
    request: dict,
    retrain_results: Dict[str, bool],
    eval_results,
    agg_eval_ms,
    wall_time,
    extra: dict = None,
) -> str:
    """Write results_unlearn.json; per-shard timings come from the retrained shards' results.json."""
    dst_graphsage = os.path.join(out_dir, "graphsage_runs")
    num_gpus = max(1, int(request["num_gpus"]))
    train_ms_values = []
    for shard_key in retrain_results:
        shard_result_path = os.path.join(dst_graphsage, shard_key, "results.json")
        if not os.path.exists(shard_result_path):
            continue
        with open(shard_result_path) as f:
            shard_result = json.load(f)
        shard_timing = shard_result.get("timing_ms") or {}
        if shard_timing.get("train") is not None:
            train_ms_values.append(shard_timing["train"] + (shard_timing.get("val_eval") or 0))

    retrain_wall_s = (
        round(sum(train_ms_values) / 1000 / num_gpus, 1)
        if train_ms_values else None
    )

    results = {
        "experiment_type": f"graphsage_{request['forget_type']}_unlearn",
        "source_state_dir": request["source_state_dir"],
        "source_sharding_dir": request["source_sharding_dir"],
        "wall_time_s": wall_time,
        "eval_data_dir": request["eval_data_dir"],
        "forget_request": {
            "forget_type": request["forget_type"],
            "description": request["description"],
            "affected_shards": request["affected_shards"],
        },
        # train/val/test sessions of each retrained shard before and after the rebuild
        "shard_sessions": request.get("shard_sessions"),
        "retrain_results": retrain_results,
        "eval": eval_results,
        "timing_ms": {
            "per_shard_train": mean_std(train_ms_values) if train_ms_values else None,
            "agg_eval_test": agg_eval_ms,
            "retrain_wall_s": retrain_wall_s,
        },
    }
    if extra:
        results.update(extra)

    results_path = os.path.join(out_dir, "results_unlearn.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)
    wall_start = time.time()

    available_gpus = torch.cuda.device_count()
    if available_gpus > 0 and args.num_gpus > available_gpus:
        print(f"[WARN] --num_gpus {args.num_gpus} > available GPUs {available_gpus} — using {available_gpus}", flush=True)
        args.num_gpus = available_gpus

    state_dir = os.path.abspath(args.state_dir)
    if args.affected_only:
        _write_affected_only(state_dir, args)
        return

    if args.out_root:
        out_root = args.out_root
    else:
        # state_dir = {results_root}/shard-runs/{dataset}/{run_name}
        dataset_dir_name = os.path.basename(os.path.dirname(state_dir))
        shard_runs_dir = os.path.dirname(os.path.dirname(state_dir))
        results_root = os.path.dirname(shard_runs_dir)
        out_root = os.path.join(results_root, "shard-unlearns", dataset_dir_name)

    run_name = args.run_name
    if not run_name:
        run_name = f"{timestamp_id()}_{_detect_method(state_dir)}_{args.forget_type}_w{max(1, args.num_gpus)}"
        if args.run_tag:
            run_name += f"_{args.run_tag}"
    out_dir = os.path.join(out_root, run_name)
    if os.path.exists(os.path.join(out_dir, REQUEST_FILENAME)):
        raise FileExistsError(f"{relpath_for_log(out_dir)} is an earlier unlearn run; finish it with: python -m "
                              f"guts.yjmob.unlearning.unlearn_batch_retry --out_dir {relpath_for_log(out_dir)}")
    if os.path.exists(out_dir):
        raise FileExistsError(f"{relpath_for_log(out_dir)} exists without {REQUEST_FILENAME}; remove it first")

    # The output directory is created only once the affected shards are known (see below).
    log = _deferred_run_logger()

    sharding_dir = _resolve_sharding_dir(state_dir, args.sharded_dir)
    cfg = load_yjmob_config(os.path.join(state_dir, "config_used.yaml"))

    log(f"GUTS unlearn ({args.forget_type})  {time.strftime('%Y-%m-%d')}  source={relpath_for_log(state_dir)}  out={relpath_for_log(out_dir)}"
        f"  GPUs={args.num_gpus}")

    eval_data_dir = ""
    if not args.skip_eval:
        eval_data_dir = _resolve_eval_data_dir(state_dir, args.eval_data_dir)

    idx = _build_index(sharding_dir, log)
    shard_keys, desc = _load_forget_set(args.forget_type, args.forget_set_path, idx, log)
    untouched_shard_ids = [s for s in range(idx.num_shards) if s not in set(shard_keys)]
    log(f"  affected shards {len(shard_keys)}/{idx.num_shards}  ({desc.replace('_count=', ': ')} records)")

    if not shard_keys:
        log("[WARN] No affected shards. Nothing to unlearn.")
        return

    out_dir = ensure_dir(out_dir)
    log.open(os.path.join(out_dir, "run_log.txt"))

    # Everything unlearn_batch_retry.py needs to finish this run if a shard retrain fails.
    request = {
        "source_state_dir": repo_relpath(state_dir),
        "source_sharding_dir": repo_relpath(sharding_dir),
        "forget_type": args.forget_type,
        "forget_set_path": repo_relpath(args.forget_set_path),
        "clean_source_dir": repo_relpath(args.clean_source_dir),
        "eval_data_dir": repo_relpath(eval_data_dir) or None,
        "num_gpus": args.num_gpus,
        "description": desc,
        "affected_shards": list(shard_keys),
    }
    save_unlearn_request(out_dir, request)

    src_graphsage = os.path.join(state_dir, "graphsage_runs")
    dst_graphsage = os.path.join(out_dir, "graphsage_runs")
    _copy_graphsage_runs_selective(src_graphsage, dst_graphsage, set(shard_keys))

    src_processed = os.path.join(sharding_dir, "processed")
    dst_processed = os.path.join(out_dir, "Sharding", "processed")
    _link_untouched_shards(src_processed, dst_processed, untouched_shard_ids)

    _copy_partition_metadata(sharding_dir, os.path.join(out_dir, "Sharding"))

    retrain_results: Dict[str, bool] = {}
    for shard_idx in shard_keys:
        shard_run_dir = os.path.join(dst_graphsage, f"shard_{shard_idx}")
        if not os.path.isdir(shard_run_dir):
            raise FileNotFoundError(f"[shard_{shard_idx}] graphsage_runs dir not found: {relpath_for_log(shard_run_dir)}")

    new_sessions = _rebuild_target_shards_from_clean(
        cfg, sharding_dir, args.clean_source_dir, dst_processed, list(shard_keys), log,
    )
    shard_sessions = {}
    for shard_idx in shard_keys:
        old_meta = torch.load(os.path.join(src_processed, f"shard_{shard_idx}", "next_location_meta.pt"), map_location="cpu")
        shard_sessions[str(shard_idx)] = {
            "before": {split: int(old_meta["sequence_index"][split]["sessions"]) for split in ("train", "val", "test")},
            "after": {split: int(new_sessions[shard_idx][split]) for split in ("train", "val", "test")},
        }
    request["shard_sessions"] = shard_sessions
    save_unlearn_request(out_dir, request)
    ready_shard_ids: List[int] = list(shard_keys)

    if ready_shard_ids:
        retrain_results.update(
            retrain_shards_parallel(
                dst_graphsage, dst_processed, ready_shard_ids, args.num_gpus, log,
            )
        )

    failed = [k for k, ok in retrain_results.items() if not ok]
    if failed:
        log(f"[ERROR] Retraining failed for: {failed}")
        log(f"  finish this run with: python -m guts.yjmob.unlearning.unlearn_batch_retry --out_dir {relpath_for_log(out_dir)}")
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

    wall_time = round(time.time() - wall_start, 1)
    results_path = write_unlearn_results(
        out_dir, request, retrain_results, eval_results, agg_eval_ms, wall_time,
    )
    log(f"done  {wall_time:.0f}s  -> {relpath_for_log(results_path)}")


if __name__ == "__main__":
    main()
