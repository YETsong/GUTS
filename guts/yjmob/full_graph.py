"""
Full-Graph baseline for yjmob next-grid prediction: one GraphSAGE + GRU model trained on
the whole graph, without sharding. With clean data (--graphsage-source-processed-dir)
it is the Full-Retrain baseline.

The run has the layout of a guts/yjmob/pipeline.py run with a single shard that holds
every cell, so training and evaluation use the same code as the GUTS runs:
    STEP 1   data          Sharding/processed/shard_0/: the preprocessed day graphs in the
                           trainer format (one row of x per active cell), the sessions as
                           they are, and the rolling graphs; Sharding/partition/day{N}.pkl
                           puts every active cell of day N in shard 0
    STEP 2   training      one model: torchrun data parallel (guts/yjmob/model/train_multi_gpu.py)
                           with several GPUs, guts/yjmob/model/train_single.py with one
    STEP 3   evaluation    shard routing evaluation (guts/yjmob/stage_eval.py) of that one
                           shard: the same test transitions, routable cells (seen on a
                           training day before the test day), exact rank and zero graph
                           embedding outside the rolling graph as the GUTS runs

Usage:
    python guts/yjmob/full_graph.py --config configs/yjmob_14d.yaml --num-gpus 4
    python guts/yjmob/full_graph.py --config ... --num-gpus 4 --graphsage-source-processed-dir <clean dir>

Output ({--out}/{run_name}/, default Results/runs/yjmob_14d/<time>_full-graph_w<GPUs>):
    Sharding/            partition/, processed/shard_0/
    graphsage_runs/      shard_0/: checkpoints, results.json, train_log.txt
    run_log.txt          the lines printed to the console (progress and results)
    eval_log.txt         shard routing evaluation details
    results.json         final metrics
    config_used.yaml     exact config used (training, data, temporal_graph and output sections)
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
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
from guts.method.partition.partitioner import GUTSPartition
from guts.method.reporting import format_metrics
from guts.method.utils import (compact_json, ensure_dir, relpath_for_log, repo_relpath, run_logger, save_pickle,
                               seed_mismatch_warning, timestamp_id)
from guts.yjmob.config import load_yjmob_config
from guts.yjmob.data.rolling_graphs import update_shard_meta_with_rolling_graphs, write_shard_rolling_graphs
from guts.yjmob.pipeline import _resolve_devices, resolve_run_seed
from guts.yjmob.stage_eval import result_line, run_shard_routing_eval
from guts.yjmob.stage_partition import source_days
from guts.yjmob.stage_train import (_collect_graphsage_results, _cuda_visible_devices_for, run_trainer_subprocess,
                                    trainer_cli_args)

DEFAULT_OUT_ROOT = os.path.join(_REPO_ROOT, "Results", "runs", "yjmob_14d")
SHARD_ID = 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-Graph baseline (one model, no sharding) for yjmob next-grid prediction.")
    p.add_argument("--config", required=True, help="YAML config (e.g. configs/yjmob_14d.yaml); the partition section is not used")
    p.add_argument("--out", default="", help="output root (default: Results/runs/yjmob_14d)")
    p.add_argument("--device", default="auto", help='device when --num-gpus 1: "auto", "cpu", "cuda" or "cuda:N"')
    p.add_argument("--num-gpus", type=int, default=1, dest="num_gpus",
                   help="GPUs for data-parallel training and evaluation (0 = all)")
    p.add_argument("--graphsage-source-processed-dir", default="",
                   help="preprocessed data dir (default: cfg.data.nextgrid_processed_dir); a clean dir gives Full-Retrain")
    p.add_argument("--seed", type=int, default=None,
                   help="training seed; default: training.seed in the YAML, else random")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. training.epochs=3")
    p.add_argument("--resume-dir", default="", dest="resume_dir", help="continue an interrupted run directory")
    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: data
# ──────────────────────────────────────────────────────────────────────────────

def _data_ready(shard_dir: str, partition_dir: str, days, window_size: int) -> bool:
    meta_path = os.path.join(shard_dir, "next_location_meta.pt")
    if not os.path.exists(meta_path):
        return False
    if not torch.load(meta_path, map_location="cpu").get("rolling_graphs", {}).get(int(window_size)):
        return False
    return all(os.path.exists(os.path.join(partition_dir, f"day{d}.pkl")) for d in days)


def build_full_graph_data(source_dir: str, sharding_dir: str, days, window_size: int, log) -> str:
    """Write the single-shard data and partitions of STEP 1 (see the module docstring); returns the shard dir.
    An incomplete shard dir (no rolling graphs in its meta yet) is built again."""
    shard_dir = os.path.join(sharding_dir, "processed", f"shard_{SHARD_ID}")
    partition_dir = os.path.join(sharding_dir, "partition")
    if _data_ready(shard_dir, partition_dir, days, window_size):
        log(f"  data: reusing {relpath_for_log(shard_dir)}")
        return shard_dir
    if os.path.isdir(shard_dir):
        shutil.rmtree(shard_dir)
    t0 = time.time()
    daily_dir = ensure_dir(os.path.join(shard_dir, "daily_graphs"))
    source_meta = torch.load(os.path.join(source_dir, "next_location_meta.pt"), map_location="cpu")
    num_nodes = int(source_meta["num_nodes"])

    daily_entries = []
    n_edges = []
    for entry in source_meta["daily_graphs"]:
        day = int(entry["day"])
        if day not in days:
            continue
        g = torch.load(os.path.join(source_dir, entry["path"]), map_location="cpu")
        active = torch.sort(g["active_nodes"].long()).values
        edge_index = g["edge_index"].long()
        torch.save({"day": day, "x": g["x"][active].float(), "edge_index": edge_index.to(torch.int32),
                    "edge_weight": g["edge_weight"].float(), "active_nodes": active.to(torch.int32)},
                   os.path.join(daily_dir, f"day_{day:04d}.pt"))
        daily_entries.append({"day": day, "path": f"daily_graphs/day_{day:04d}.pt",
                              "num_edges": int(edge_index.size(1)), "num_active_nodes": int(active.numel())})
        nodes = active.tolist()
        save_pickle(GUTSPartition(shards_nodes={SHARD_ID: nodes},
                                  shards_edges={SHARD_ID: torch.searchsorted(active, edge_index)},
                                  node_to_shard={n: SHARD_ID for n in nodes}),
                    os.path.join(partition_dir, f"day{day}.pkl"))
        n_edges.append(int(edge_index.size(1)))
    missing = sorted(set(days) - {e["day"] for e in daily_entries})
    if missing:
        raise FileNotFoundError(f"no day graph for days {missing} in {relpath_for_log(source_dir)}")
    shutil.copytree(os.path.join(source_dir, "sequences"), os.path.join(shard_dir, "sequences"))

    meta = copy.deepcopy(source_meta)
    meta["task"] = "full_graph_next_grid_prediction"
    meta["node_feat_dim"] = len(meta["feature_names"])
    meta["source_processed_dir"] = repo_relpath(source_dir)
    meta["daily_graphs"] = daily_entries
    torch.save(meta, os.path.join(shard_dir, "next_location_meta.pt"))

    target_days = [d for d in sorted(days) if d >= window_size]
    entries = write_shard_rolling_graphs(shard_dir, target_days, window_size, num_nodes)
    update_shard_meta_with_rolling_graphs(shard_dir, window_size, entries)
    log(f"  data done  {len(daily_entries)} day graphs ({min(n_edges):,}-{max(n_edges):,} edges)"
        f"  rolling graphs (window {window_size}, target days {target_days[0]}-{target_days[-1]})"
        f"  {time.time() - t0:.0f}s")
    return shard_dir


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: training
# ──────────────────────────────────────────────────────────────────────────────

def train_full_graph(cfg, run_dir: str, shard_dir: str, devices, log) -> dict:
    """Train the single model (resumed from its last.pt when interrupted); returns its
    entry in the format of _collect_graphsage_results."""
    model_dir = ensure_dir(os.path.join(run_dir, "graphsage_runs", f"shard_{SHARD_ID}"))
    if os.path.exists(os.path.join(model_dir, "results.json")):
        log("  training: model already trained")
    else:
        args = trainer_cli_args(cfg, shard_dir, model_dir)
        resumed = " (resumed from last.pt)" if "--resume" in args else ""
        model_py = os.path.join(_REPO_ROOT, "guts", "yjmob", "model")
        t0 = time.time()
        if len(devices) > 1:
            log(f"  training 1 model on {len(devices)} GPUs (torchrun data parallel){resumed}")
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(devices)}",
                   os.path.join(model_py, "train_multi_gpu.py"), *args]
            run_trainer_subprocess(cmd, ",".join(str(d.index) for d in devices))
        else:
            log(f"  training 1 model on {devices[0]}{resumed}")
            run_trainer_subprocess([sys.executable, os.path.join(model_py, "train_single.py"), *args],
                                   _cuda_visible_devices_for(str(devices[0])))
        log(f"  training done  {time.time() - t0:.0f}s")
    result = _collect_graphsage_results(run_dir, [SHARD_ID])[str(SHARD_ID)]
    log(f"  best E{int(result['best_epoch']):02d}/stopped E{int(result['stopped_epoch']):02d}"
        f"  train time {result['timing_ms']['train'] / 1000:.0f}s")
    log(f"    train  {format_metrics(result['best_train_metrics'], sep='  ')}")
    log(f"    val    {format_metrics(result['best_val_metrics'], sep='  ')}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _save_config_yaml(cfg, path: str) -> None:
    data = config_to_dict(cfg)
    data.pop("partition")
    # Paths are stored relative to the repository root (see repo_relpath).
    data["data"]["nextgrid_processed_dir"] = repo_relpath(data["data"]["nextgrid_processed_dir"])
    data["output"]["root"] = repo_relpath(data["output"]["root"])
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def main(argv=None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)
    overrides = parse_overrides(args.override)
    cfg = load_yjmob_config(args.config, overrides=overrides)
    cfg.output.root = args.out or DEFAULT_OUT_ROOT

    devices = _resolve_devices(args.num_gpus, args.device)
    num_gpus = len(devices)

    if args.resume_dir:
        run_dir = os.path.abspath(args.resume_dir)
        cfg.output.root, cfg.output.run_name = os.path.split(run_dir)
    else:
        if not cfg.output.run_name:
            cfg.output.run_name = f"{timestamp_id()}_full-graph_w{num_gpus}"
        run_dir = os.path.join(cfg.output.root, cfg.output.run_name)
    seed_msg = resolve_run_seed(cfg, args, overrides, run_dir)

    ensure_dir(run_dir)
    run_log_path = os.path.join(run_dir, "run_log.txt")
    eval_log_path = os.path.join(run_dir, "eval_log.txt")
    log = run_logger(run_log_path)

    log(f"Full-Graph baseline{' (resumed)' if args.resume_dir else ''}  {time.strftime('%Y-%m-%d')}"
        f"  run={relpath_for_log(run_dir)}  seed={cfg.training.seed}" + (f" ({seed_msg})" if seed_msg else "")
        + f"  GPUs={num_gpus}")
    source_dir = args.graphsage_source_processed_dir or cfg.data.nextgrid_processed_dir
    seed_warning = seed_mismatch_warning(source_dir, cfg.training.seed)
    if seed_warning:
        log(seed_warning)
    cfg.data.nextgrid_processed_dir = source_dir
    train_days = source_days(source_dir)
    window_size = int(cfg.temporal_graph.window_size)
    log(f"  data={relpath_for_log(source_dir)}  days {train_days[0]}-{train_days[-1]} ({len(train_days)})")

    config_used_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(config_used_path):
        _save_config_yaml(cfg, config_used_path)

    wall_start = time.time()
    sharding_dir = ensure_dir(os.path.join(run_dir, "Sharding"))
    log("STEP 1/3 full-graph data (one shard with every cell)")
    t0 = time.time()
    shard_dir = build_full_graph_data(source_dir, sharding_dir, train_days, window_size, log)
    build_ms = round((time.time() - t0) * 1000)

    log("STEP 2/3 training")
    model_result = train_full_graph(cfg, run_dir, shard_dir, devices, log)

    log("STEP 3/3 shard routing evaluation (test)")
    t0 = time.time()
    agg_eval_results = run_shard_routing_eval(
        run_dir=run_dir,
        full_data_dir=source_dir,
        eval_log_path=eval_log_path,
        window_size=window_size,
        num_gpus_eval=num_gpus,
        sharding_dir=sharding_dir,
    )
    agg_eval_ms = round((time.time() - t0) * 1000)
    log(f"  test  {result_line(agg_eval_results)}  {agg_eval_ms / 1000:.0f}s")

    results = {
        "experiment_type": "guts_full_graph",
        "run_dir": repo_relpath(run_dir),
        "source_processed_dir": repo_relpath(source_dir),
        "train_days": train_days,
        "num_gpus": num_gpus,
        "devices": [str(d) for d in devices],
        "wall_time_s": round(time.time() - wall_start, 1),
        "graphsage_shard_results": {str(SHARD_ID): model_result},
        "graphsage_agg_eval": agg_eval_results,
        "timing_ms": {
            "build": build_ms,
            "train": model_result["timing_ms"].get("train"),
            "val_eval": model_result["timing_ms"].get("val_eval"),
            "agg_eval_test": agg_eval_ms,
        },
    }
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(compact_json(results))
    log(f"done  {results['wall_time_s']:.0f}s  -> {relpath_for_log(results_path)}")


if __name__ == "__main__":
    main()
