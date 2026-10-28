"""Static-graph GUTS pipeline: split file -> train subgraph -> partition (STEP 1) -> shard
subgraphs (STEP 2) -> shard training (STEP 3) -> aggregated evaluation on the full graph (STEP 4).

Output (<output.root>/<run_name>/): run_log.txt (the lines printed to the console),
config_used.yaml, args.json, partition.pkl, train_orig_ids.pt, checkpoints/shard_<k>.pt,
results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import yaml

from guts.method.config import config_to_dict, parse_overrides
from guts.method.reporting import mean_std
from guts.method.utils import (ensure_dir, random_seed, relpath_for_log, repo_relpath, run_logger,
                               seed_mismatch_warning, timestamp_id)
from guts.static_graph.config import load_static_config
from guts.static_graph.cora_data import DATASET, induce_train_subgraph, load_cora_sample
from guts.static_graph.model import resolve_model_hp
from guts.static_graph.stage_eval import evaluate_aggregated
from guts.static_graph.stage_partition import run_partition_stage
from guts.static_graph.stage_train import run_train_stage


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static-graph sharding pipeline (GUTS)")
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", required=True,
                   help="graph with its split: a split file or, for M_retrain, a clean graph (guts/static_graph/"
                   "cora_data.py split/clean)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--model_type", default="graphsage", choices=["graphsage", "gcn", "gat", "gin"],
                   help="defaults are per model (MODEL_DEFAULTS); --lr etc. override them only when given")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None,
                   help="run seed: partition and training; default: training.seed in the YAML, else random")
    p.add_argument("--run_name", default="", help="auto-generated if empty")
    return p.parse_args(argv)


def _output_root(cfg) -> str:
    return cfg.output.root or os.path.join(_REPO_ROOT, "Results", "shard-runs", DATASET)


def resolve_run_seed(args: argparse.Namespace, cfg) -> Tuple[int, str]:
    """Run seed: --seed, else training.seed (YAML/override), else random."""
    if args.seed is not None:
        return int(args.seed), "given"
    if cfg.training.seed is not None:
        return int(cfg.training.seed), "config"
    return random_seed(), "random"


def set_run_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_run_args(run_dir: str, args: argparse.Namespace, hp: dict) -> None:
    """args.json: what the unlearning and the notebooks read back (paths repository-relative)."""
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump({
            "model_type": args.model_type,
            "seed": args.seed, "data_path": repo_relpath(args.data_path),
            "epochs": args.epochs, "hp": hp,
        }, f, indent=2)


def test_line(ev: dict, num_models: int) -> str:
    return (f"test  ACC {ev['test_acc']:.4f}  F1 {ev['test_f1_macro']:.4f}"
            f"  ({ev['test_total']} test nodes, {num_models} model{'s' if num_models != 1 else ''})")


def run(args: argparse.Namespace, cfg, run_dir: str) -> dict:
    t_wall0 = time.perf_counter()
    seed, seed_source = resolve_run_seed(args, cfg)
    args.seed = seed
    cfg.training.seed = seed
    set_run_seed(seed)
    run_dir = ensure_dir(run_dir)
    log = run_logger(os.path.join(run_dir, "run_log.txt"))

    log(f"GUTS static-graph pipeline  {time.strftime('%Y-%m-%d')}  run={relpath_for_log(run_dir)}"
        f"  seed={seed} ({seed_source})  model={args.model_type}  K={cfg.partition.num_shards}")
    seed_warning = seed_mismatch_warning(args.data_path, seed)
    if seed_warning:
        log(seed_warning)

    cfg_dict = config_to_dict(cfg)
    cfg_dict["output"]["root"] = repo_relpath(cfg_dict["output"]["root"])
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    hp = resolve_model_hp(
        args.model_type, hidden_dim=args.hidden_dim, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout,
    )
    save_run_args(run_dir, args, hp)

    sample = load_cora_sample(args.data_path)
    log(f"  data={relpath_for_log(args.data_path)}  nodes {sample.num_nodes}  edges {sample.num_edges}"
        f"  classes {sample.num_classes}  train {int(sample.train_mask.sum())}  test {int(sample.test_mask.sum())}")
    train_sample, train_orig_ids = induce_train_subgraph(sample)
    log(f"  G_train  nodes {train_sample.num_nodes}  edges {train_sample.num_edges}")
    torch.save(train_orig_ids, os.path.join(run_dir, "train_orig_ids.pt"))

    log(f"STEP 1/4 partition (K={cfg.partition.num_shards})")
    part = run_partition_stage(cfg, train_sample, run_dir, log)
    partition, shards = part["partition"], part["shards"]

    log(f"STEP 3/4 shard training ({args.model_type}, {args.epochs} epochs, final-epoch models)")
    t0 = time.perf_counter()
    shard_models, shard_summaries = run_train_stage(
        shards, train_sample, args.model_type, hp, args.epochs, args.device,
        ensure_dir(os.path.join(run_dir, "checkpoints")),
    )
    log(f"  {len(shard_models)}/{len(shards)} shards trained  {time.perf_counter() - t0:.0f}s")

    log("STEP 4/4 aggregated evaluation (test, G_full, mean softmax over the shard models)")
    t0_agg_eval = time.perf_counter()
    ev = evaluate_aggregated(shard_models, sample, args.device)
    agg_eval_ms = round((time.perf_counter() - t0_agg_eval) * 1000)
    log(f"  {test_line(ev, len(shard_models))}  {agg_eval_ms / 1000:.1f}s")

    train_ms_values = [s["train_ms"] for s in shard_summaries.values() if s["train_ms"] is not None]
    epoch0_train_ms_values = [s["epoch0_train_ms"] for s in shard_summaries.values() if s["epoch0_train_ms"] is not None]
    timing_ms = {
        "partition": part["partition_ms"],
        "build": part["build_ms"],
        "epoch0_train_ms": mean_std(epoch0_train_ms_values),
        "per_shard_train": mean_std(train_ms_values, extremes=True),
        "agg_eval_test": agg_eval_ms,
    }

    results = {
        "experiment_type": "static_graph_sharding",
        "dataset": DATASET,
        "model_type": args.model_type,
        "strategy": cfg.partition.strategy,
        "num_shards": cfg.partition.num_shards,
        "partition_balance": partition.quality_metrics["balance"],
        "partition_fairness": partition.quality_metrics["fairness"],
        "shard_summaries": shard_summaries,
        "aggregation": "uniform_average",
        "aggregation_weights": {str(k): round(v, 6) for k, v in ev["weights"].items()},
        "test_acc": ev["test_acc"],
        "test_f1_macro": ev["test_f1_macro"],
        "test_scored": ev["test_total"],
        "test_total": ev["test_total"],
        "timing_ms": timing_ms,
    }
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"done  {time.perf_counter() - t_wall0:.0f}s  -> {relpath_for_log(results_path)}")
    return results


def main(argv: Optional[List[str]] = None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)
    cfg = load_static_config(args.config, overrides=parse_overrides(args.override))
    method_tag = str(cfg.partition.strategy).lower()
    run_name = args.run_name or f"{timestamp_id()}_{method_tag}_{args.model_type}"
    run(args, cfg, os.path.join(_output_root(cfg), run_name))


if __name__ == "__main__":
    main()
