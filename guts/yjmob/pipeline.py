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
    graphsage_runs/      shard_{k}/: checkpoints, results.json, train_log.txt
    run_log.txt          the lines printed to the console (progress and results)
    eval_log.txt         shard routing evaluation details
    results.json         final metrics
    config_used.yaml     exact config used
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

import torch
import yaml

from guts.method.config import config_to_dict, parse_overrides
from guts.method.reporting import mean_std
from guts.yjmob.config import load_yjmob_config, resolve_output_dirs
from guts.method.utils import (compact_json, ensure_dir, relpath_for_log, repo_relpath, resolve_seed, run_logger,
                               seed_mismatch_warning, timestamp_id)
from guts.yjmob.stage_eval import result_line, run_shard_routing_eval
from guts.yjmob.stage_partition import run_partition_stage, source_days
from guts.yjmob.stage_train import run_train_stage


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-day GUTS pipeline for yjmob next-grid prediction.")
    p.add_argument("--config", required=True, help="YAML config (e.g. configs/yjmob_14d.yaml)")
    p.add_argument("--days", type=int, nargs="*", default=None, help="training days (default: all days in the data)")
    p.add_argument("--out", default="", help="output root (default: cfg.output.root)")
    p.add_argument("--device", default="auto", help='device when --num-gpus 1: "auto", "cpu", "cuda" or "cuda:N"')
    p.add_argument("--num-gpus", type=int, default=1, dest="num_gpus", help="GPUs for shard training and evaluation (0 = all)")
    p.add_argument("--graphsage-source-processed-dir", default="",
                   help="preprocessed data dir (default: cfg.data.nextgrid_processed_dir)")
    p.add_argument("--sharded-dir", default="", help="reuse an existing Sharding/ dir")
    p.add_argument("--seed", type=int, default=None,
                   help="run seed: partition and shard training; default: training.seed in the YAML, else random")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. training.epochs=3")
    p.add_argument("--resume-dir", default="", dest="resume_dir", help="continue an interrupted run directory")
    return p.parse_args(argv)


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


def resolve_run_seed(cfg, args, overrides: dict, run_dir: str) -> str:
    """Set cfg.training.seed: --seed, else training.seed (YAML/override), else a random seed.
    A resumed run keeps its saved seed; a different explicit seed is an error. config_used.yaml
    records the result. Returns the note of the header line ("given", "random" or "")."""
    resumed_cfg_path = os.path.join(run_dir, "config_used.yaml")
    if args.resume_dir and os.path.exists(resumed_cfg_path):
        saved_seed = load_yjmob_config(resumed_cfg_path).training.seed
        # Only an explicitly given seed is checked; the YAML default is replaced by the saved one.
        given = args.seed if args.seed is not None else (
            cfg.training.seed if "training.seed" in overrides else None)
        if given is not None and given != saved_seed:
            raise ValueError(f"seed {given} differs from the resumed run's seed {saved_seed}")
        cfg.training.seed = saved_seed
        return ""
    if args.seed is not None:
        cfg.training.seed = resolve_seed(args.seed)
        return "given"
    seed_msg = "random" if cfg.training.seed is None else ""
    cfg.training.seed = resolve_seed(cfg.training.seed)
    return seed_msg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)

    overrides = parse_overrides(args.override)
    cfg = load_yjmob_config(args.config, overrides=overrides)

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

    seed_msg = resolve_run_seed(cfg, args, overrides, run_dir)

    # ── Logging: console + run_log.txt; details go to eval_log.txt and the shard logs ──
    ensure_dir(run_dir)
    run_log_path  = os.path.join(run_dir, "run_log.txt")
    eval_log_path = os.path.join(run_dir, "eval_log.txt")
    log = run_logger(run_log_path)

    # ── Dataset + train days ──────────────────────────────────────────────────
    log(f"GUTS pipeline{' (resumed)' if args.resume_dir else ''}  {time.strftime('%Y-%m-%d')}  run={relpath_for_log(run_dir)}"
        f"  seed={cfg.training.seed}" + (f" ({seed_msg})" if seed_msg else "")
        + f"  K={cfg.partition.num_shards}  GPUs={num_gpus}")

    nextgrid_source_dir = args.graphsage_source_processed_dir or cfg.data.nextgrid_processed_dir
    seed_warning = seed_mismatch_warning(nextgrid_source_dir, cfg.training.seed)
    if seed_warning:
        log(seed_warning)
    # The partitioner reads the train trajs and grid of this same directory.
    cfg.data.nextgrid_processed_dir = nextgrid_source_dir

    train_days = sorted(args.days) if args.days else source_days(nextgrid_source_dir)
    log(f"  data={relpath_for_log(nextgrid_source_dir)}  days {train_days[0]}-{train_days[-1]} ({len(train_days)})")

    # A resumed run keeps the config_used.yaml of its first start.
    _config_used_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(_config_used_path):
        _save_config_yaml(cfg, _config_used_path)

    wall_start = time.time()

    _ext_sharding = args.sharded_dir
    sharding_dir  = os.path.abspath(_ext_sharding) if _ext_sharding \
                    else ensure_dir(os.path.join(run_dir, "Sharding"))
    part = run_partition_stage(cfg, train_days, nextgrid_source_dir, sharding_dir, bool(_ext_sharding), log)
    by_day = part["by_day"]
    partition_ms_total, build_ms_total = part["partition_ms"], part["build_ms"]
    refinement_summary = part["refinement"]

    log("STEP 2/3 shard data + shard training")
    preprocess_summary, shard_results = run_train_stage(
        cfg=cfg,
        args=args,
        run_dir=run_dir,
        train_days=train_days,
        by_day=by_day,
        devices=devices,
        log=log,
        run_log_path=run_log_path,
    )

    log("STEP 3/3 shard routing evaluation (test)")
    t0_agg_eval = time.time()
    agg_eval_results = run_shard_routing_eval(
        run_dir=run_dir,
        full_data_dir=nextgrid_source_dir,
        eval_log_path=eval_log_path,
        window_size=cfg.temporal_graph.window_size,
        num_gpus_eval=num_gpus,
        sharding_dir=sharding_dir,
    )

    agg_eval_ms = round((time.time() - t0_agg_eval) * 1000)
    log(f"  test  {result_line(agg_eval_results)}  {agg_eval_ms / 1000:.0f}s")
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
        "per_shard_train": mean_std(shard_train_ms, digits=3, extremes=True),
        "per_shard_val_eval": mean_std(shard_val_eval_ms, digits=3, extremes=True),
        "agg_eval_test": agg_eval_ms,
    }

    results = {
        "experiment_type": "guts_graphsage_pipeline",
        "run_dir": repo_relpath(run_dir),
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
    log(f"done  {results['wall_time_s']:.0f}s  -> {relpath_for_log(results_path)}")


def _save_config_yaml(cfg, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = config_to_dict(cfg)
    # Paths are stored relative to the repository root (see repo_relpath).
    data["data"]["nextgrid_processed_dir"] = repo_relpath(data["data"]["nextgrid_processed_dir"])
    data["output"]["root"] = repo_relpath(data["output"]["root"])
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)



if __name__ == "__main__":
    main()
