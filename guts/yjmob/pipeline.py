"""
End-to-end GUTS shard pipeline for yjmob next-grid prediction (multi-day + multi-GPU):
    STEP 1   per-day partition (GUTS) + shard build          guts/yjmob/stage_partition.py
    STEP 2   shard datasets + one GraphSAGE model per shard  guts/yjmob/stage_train.py
    STEP 3   shard routing evaluation (rolling graphs)       guts/yjmob/stage_eval.py
This file only parses the arguments, runs the three stages and writes results.json.

Usage:
    python guts/yjmob/pipeline.py --config configs/yjmob_14d.yaml --num-gpus 4
    python guts/yjmob/pipeline.py --config ... --num-gpus 4 --days 0 1 2 3 --override training.epochs=3

Output ({output.root}/{run_name}/):
    Sharding/            partition/, processed/shard_{k}/, refinement_metrics.json
    graphsage_runs/      shard_{k}/ checkpoints + per-shard results.json
    logs/                pipeline.txt, eval_log.txt
    results.json         final metrics
    config_used.yaml     exact config used
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

import torch
import yaml

from guts.config import align_retrain_seed, config_to_dict, load_config, resolve_output_dirs
from guts.utils import append_log_synced, compact_json, ensure_dir, relpath_for_log, timestamp_id
from guts.yjmob.stage_eval import run_shard_routing_eval
from guts.yjmob.stage_partition import run_partition_stage, source_days
from guts.yjmob.stage_train import run_train_stage


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-day GUTS pipeline for yjmob next-grid prediction.")
    p.add_argument("--config", required=True, help="YAML config (e.g. configs/yjmob_14d.yaml)")
    p.add_argument("--days", type=int, nargs="*", default=None, help="training days (default: all days in the data)")
    p.add_argument("--out", default="", help="output root (default: cfg.output.root)")
    p.add_argument("--device", default="auto", help='device when --num-gpus 1: "auto", "cpu", "cuda" or "cuda:N"')
    p.add_argument("--num-gpus", type=int, default=1, dest="num_gpus", help="GPUs for shard training and evaluation (0 = all)")
    p.add_argument("--graphsage-source-processed-dir", default="",
                   help="preprocessed data dir (default: cfg.data.nextgrid_processed_dir)")
    p.add_argument("--sharded-dir", default="", help="reuse an existing Sharding/ dir")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. training.epochs=3")
    p.add_argument("--resume-dir", default="", dest="resume_dir", help="continue an interrupted run directory")
    return p.parse_args()


def _parse_overrides(override_list) -> dict:
    result = {}
    for item in (override_list or []):
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, raw_val = item.split("=", 1)
        for cast in (int, float):
            try:
                result[key] = cast(raw_val)
                break
            except ValueError:
                pass
        else:
            if raw_val.lower() in ("true", "false"):
                result[key] = raw_val.lower() == "true"
            else:
                result[key] = raw_val
    return result


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_devices(num_gpus: int, device_str: str):
    def _cuda_count() -> int:
        try:
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            return 0

    if num_gpus == 0:
        n = _cuda_count()
        if n == 0:
            return [torch.device("cpu")]
        return [torch.device(f"cuda:{i}") for i in range(n)]
    if num_gpus == 1:
        return [_resolve_device(device_str)]
    n = _cuda_count()
    if n < num_gpus:
        raise RuntimeError(
            f"Requested --num-gpus {num_gpus}, but PyTorch sees only {n} CUDA device(s). "
            "Check the NVIDIA driver / PyTorch CUDA build, or run with --num-gpus 1 --device cpu."
        )
    return [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def _mean_std_ms(values: list) -> dict:
    """Mean and sample standard deviation (ddof=1; 0 for a single value)."""
    n = len(values)
    if n == 0:
        return None
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {"mean": round(m, 3), "std": round(var ** 0.5, 3), "min": min(values), "max": max(values), "n": n}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    overrides = _parse_overrides(args.override)
    cfg = load_config(args.config, overrides=overrides)
    # Retrain on a per-seed clean dataset: train with the forget-set seed.
    seed_msg = align_retrain_seed(cfg, args.graphsage_source_processed_dir or cfg.data.nextgrid_processed_dir, overrides)

    if args.out:
        cfg.output.root = args.out

    devices  = _resolve_devices(args.num_gpus, args.device)
    num_gpus = len(devices)

    if args.resume_dir:
        run_dir = os.path.abspath(args.resume_dir)
    else:
        if not cfg.output.run_name:
            _label = str(cfg.partition.strategy).lower()
            cfg.output.run_name = f"{timestamp_id()}_{_label}_w{num_gpus}"
        cfg     = resolve_output_dirs(cfg)
        run_dir = os.path.join(cfg.output.root, cfg.output.run_name)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir       = ensure_dir(os.path.join(run_dir, "logs"))
    log_path      = os.path.join(log_dir, "pipeline.txt")
    eval_log_path = os.path.join(log_dir, "eval_log.txt")

    def log(msg: str) -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        append_log_synced(log_path, line)

    # ── Dataset + train days ──────────────────────────────────────────────────
    log("=" * 60)
    _strategy_label = str(cfg.partition.strategy).upper()
    log(f"GRAPH UNLEARNING PIPELINE  strategy={_strategy_label}  (multi-day)")
    log(f"  run_dir : {relpath_for_log(run_dir)}")
    log(f"  config  : {relpath_for_log(args.config)}")
    log(f"  gpus    : {num_gpus}  {[str(d) for d in devices]}")
    log(f"  K shards: {cfg.partition.num_shards}")
    log("  model   : graphsage")
    log(f"  seed    : {cfg.training.seed}" + (f"  [{seed_msg}]" if seed_msg else ""))
    log("=" * 60)

    nextgrid_source_dir = args.graphsage_source_processed_dir or cfg.data.nextgrid_processed_dir
    # The partitioner reads the train paths and grid of this same directory.
    cfg.data.nextgrid_processed_dir = nextgrid_source_dir

    if args.days:
        train_days = sorted(args.days)
        log(f"Using specified days: {train_days}")
    else:
        train_days = source_days(nextgrid_source_dir)
        log(f"Using all {len(train_days)} next-grid days: {train_days[:5]}...{train_days[-5:]}")

    _config_used_path = os.path.join(run_dir, "config_used.yaml")
    if os.path.exists(_config_used_path):
        log("  config_used.yaml already exists — keeping the first run's values (not rewritten on resume)")
    else:
        _save_config_yaml(cfg, _config_used_path)

    wall_start = time.time()

    log("=" * 60)
    _ext_sharding = args.sharded_dir
    sharding_dir  = os.path.abspath(_ext_sharding) if _ext_sharding \
                    else ensure_dir(os.path.join(run_dir, "Sharding"))
    part = run_partition_stage(cfg, train_days, nextgrid_source_dir, sharding_dir, bool(_ext_sharding), log)
    by_day = part["by_day"]
    partition_ms_total, build_ms_total = part["partition_ms"], part["build_ms"]
    refinement_summary = part["refinement"]

    log("=" * 60)
    log("STEP 2/3  Build/reuse GraphSAGE data + train shard models")
    preprocess_summary, shard_results = run_train_stage(
        cfg=cfg,
        args=args,
        run_dir=run_dir,
        train_days=train_days,
        by_day=by_day,
        devices=devices,
        eval_log_path=eval_log_path,
        log=log,
    )

    log("=" * 60)
    log("STEP 3/3  Shard routing evaluation (all shards)")
    t0_agg_eval = time.time()
    agg_eval_results = run_shard_routing_eval(
        run_dir=run_dir,
        full_data_dir=nextgrid_source_dir,
        eval_log_path=eval_log_path,
        log=log,
        window_size=cfg.temporal_graph.window_size,
        num_gpus_eval=num_gpus,
        sharding_dir=sharding_dir,
    )

    agg_eval_ms = round((time.time() - t0_agg_eval) * 1000)
    shard_train_ms = [
        r.get("timing_ms", {}).get("train") for r in shard_results.values()
        if r.get("timing_ms", {}).get("train") is not None
    ]
    shard_val_eval_ms = [
        r.get("timing_ms", {}).get("val_eval") for r in shard_results.values()
        if r.get("timing_ms", {}).get("val_eval") is not None
    ]
    timing_ms = {
        "partition": partition_ms_total,
        "build": build_ms_total,
        "per_shard_train": _mean_std_ms(shard_train_ms),
        "per_shard_val_eval": _mean_std_ms(shard_val_eval_ms),
        "agg_eval_test": agg_eval_ms,
    }

    log("=" * 60)
    results = {
        "experiment_type": "guts_graphsage_pipeline",
        "run_dir": run_dir,
        "train_days": train_days,
        "num_gpus": num_gpus,
        "devices": [str(d) for d in devices],
        "wall_time_s": round(time.time() - wall_start, 1),
        "partition": {
            "num_shards": cfg.partition.num_shards,
            "method": cfg.partition.strategy,
            "per_day": True,
            "num_train_days": len(train_days),
            "refinement": refinement_summary,
        },
        "graphsage_preprocess": preprocess_summary,
        "graphsage_shard_results": shard_results,
        "graphsage_agg_eval": agg_eval_results,
        "timing_ms": timing_ms,
    }
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(compact_json(results))
    log("=" * 60)
    log(f"PIPELINE COMPLETE  strategy={_strategy_label}  total_wall={results['wall_time_s']:.1f}s")
    log(f"  results  → {relpath_for_log(results_path)}")
    log(f"  eval_log → {relpath_for_log(eval_log_path)}")
    log("=" * 60)
    return


def _save_config_yaml(cfg, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_to_dict(cfg), f, default_flow_style=False)



if __name__ == "__main__":
    main()
