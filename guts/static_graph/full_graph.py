"""Full-Graph baseline for static graphs: one model trained on the whole train subgraph,
without sharding, and evaluated on the full graph like the sharded runs (STEP 4 of
guts/static_graph/pipeline.py with a single model). With a clean graph as --data_path
it is the Full-Retrain baseline.

Training is the shard training of the pipeline (guts/static_graph/stage_train.py) on
G_train: full batch, --epochs epochs, the final-epoch model. The run seed (--seed, else
training.seed of --config, else random) drives the training.

Output (<out_root>/<run_name>/, default Results/runs/Cora/<time>_full-graph_<model>):
run_log.txt (the lines printed to the console), args.json, checkpoints/model.pt, results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from guts.method.config import parse_overrides
from guts.method.utils import ensure_dir, relpath_for_log, run_logger, seed_mismatch_warning, timestamp_id
from guts.static_graph.config import load_static_config
from guts.static_graph.cora_data import DATASET, induce_train_subgraph, load_cora_sample
from guts.static_graph.model import resolve_model_hp
from guts.static_graph.pipeline import resolve_run_seed, save_run_args, set_run_seed, test_line
from guts.static_graph.stage_eval import evaluate_aggregated
from guts.static_graph.stage_train import train_seconds, train_shard

DEFAULT_OUT_ROOT = os.path.join(_REPO_ROOT, "Results", "runs", DATASET)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static-graph Full-Graph baseline (one model, no sharding)")
    p.add_argument("--config", required=True, help="YAML whose training.seed is the default run seed (other sections unused)")
    p.add_argument("--data_path", required=True,
                   help="graph with its split: a split file or, for Full-Retrain, a clean graph (guts/static_graph/"
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
                   help="run seed (training); default: training.seed in the YAML, else random")
    p.add_argument("--out_root", default="", help=f"default: Results/runs/{DATASET}")
    p.add_argument("--run_name", default="", help="auto-generated if empty")
    return p.parse_args(argv)


def run(args: argparse.Namespace, cfg, run_dir: str) -> dict:
    t_wall0 = time.perf_counter()
    seed, seed_source = resolve_run_seed(args, cfg)
    args.seed = seed
    set_run_seed(seed)
    run_dir = ensure_dir(run_dir)
    log = run_logger(os.path.join(run_dir, "run_log.txt"))

    log(f"Full-Graph baseline (static graph)  {time.strftime('%Y-%m-%d')}  run={relpath_for_log(run_dir)}"
        f"  seed={seed} ({seed_source})  model={args.model_type}")
    seed_warning = seed_mismatch_warning(args.data_path, seed)
    if seed_warning:
        log(seed_warning)
    hp = resolve_model_hp(
        args.model_type, hidden_dim=args.hidden_dim, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout,
    )
    save_run_args(run_dir, args, hp)

    sample = load_cora_sample(args.data_path)
    log(f"  data={relpath_for_log(args.data_path)}  nodes {sample.num_nodes}  edges {sample.num_edges}"
        f"  classes {sample.num_classes}  train {int(sample.train_mask.sum())}  test {int(sample.test_mask.sum())}")
    train_sample, _ = induce_train_subgraph(sample)
    log(f"  G_train  nodes {train_sample.num_nodes}  edges {train_sample.num_edges}")

    log(f"STEP 1/2 training ({args.model_type}, {args.epochs} epochs, final-epoch model)")
    model, summary = train_shard(args.model_type, hp, args.epochs, train_sample.x, train_sample.edge_index,
                                 train_sample.y, train_sample.train_mask, sample.num_classes, args.device)
    if model is None:
        raise ValueError(f"{relpath_for_log(args.data_path)} has no train node")
    torch.save(model.state_dict(), os.path.join(ensure_dir(os.path.join(run_dir, "checkpoints")), "model.pt"))
    log(f"  training done  train nodes {summary['train_nodes']}  {train_seconds(summary):.1f}s")

    log("STEP 2/2 evaluation (test, G_full)")
    t0 = time.perf_counter()
    ev = evaluate_aggregated({0: model}, sample, args.device)
    eval_ms = round((time.perf_counter() - t0) * 1000)
    log(f"  {test_line(ev, 1)}  {eval_ms / 1000:.1f}s")

    results = {
        "experiment_type": "static_graph_full_graph",
        "dataset": DATASET,
        "model_type": args.model_type,
        "train_nodes": summary["train_nodes"],
        "test_acc": ev["test_acc"],
        "test_f1_macro": ev["test_f1_macro"],
        "test_total": ev["test_total"],
        "timing_ms": {
            "epoch0_train_ms": summary["epoch0_train_ms"],
            "train": summary["train_ms"],
            "eval_test": eval_ms,
        },
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
    run_name = args.run_name or f"{timestamp_id()}_full-graph_{args.model_type}"
    run(args, cfg, os.path.join(args.out_root or DEFAULT_OUT_ROOT, run_name))


if __name__ == "__main__":
    main()
