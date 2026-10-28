"""Static-graph shard unlearning: delete a forget set from the source run's shards, retrain
only the affected shards (shrink + remap their subgraphs) and evaluate the aggregate on
the clean graph.

Output (<out_root>/<run_name>/): run_log.txt (the lines printed to the console),
checkpoints/shard_<k>.pt (retrained or copied from the source run), results_unlearn.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch_geometric.utils import subgraph

from guts.method.reporting import mean_std
from guts.method.shard import ShardBuilder
from guts.method.utils import (ensure_dir, load_pickle, relpath_for_log, repo_relpath, resolve_repo_path, run_logger,
                               timestamp_id)
from guts.static_graph.cora_data import DATASET, induce_train_subgraph, load_cora_sample, make_clean, source_mismatch_warning
from guts.static_graph.model import build_model
from guts.static_graph.pipeline import test_line
from guts.static_graph.stage_eval import evaluate_aggregated
from guts.static_graph.stage_train import train_shard


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static-graph M_unlearn (shrink + remap the affected shards)")
    p.add_argument("--run_dir", required=True, help="source run directory")
    p.add_argument("--forget_set", required=True, help="forget set JSON")
    p.add_argument("--clean_data_path", default="",
                   help="clean G_full for evaluation; built in memory from the source run's data if empty")
    p.add_argument("--out_root", default="", help=f"default: Results/shard-unlearns/{DATASET}")
    p.add_argument("--run_name", default="", help="auto-generated if empty")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def _rebuild_for_node_deletion(data, deleted_local: torch.Tensor):
    keep_mask = ~torch.isin(data.uids, deleted_local)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    new_edge_index, _ = subgraph(keep_idx, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0))
    return data.x[keep_idx], new_edge_index, data.train_mask[keep_idx], data.uids[keep_idx]


def _rebuild_for_edge_deletion(data, deleted_pairs: set):
    u_local = data.uids[data.edge_index[0]]
    v_local = data.uids[data.edge_index[1]]
    keep_edge = torch.tensor(
        [(int(u), int(v)) not in deleted_pairs for u, v in zip(u_local.tolist(), v_local.tolist())],
        dtype=torch.bool,
    )
    return data.x, data.edge_index[:, keep_edge], data.train_mask, data.uids


def run(args: argparse.Namespace) -> dict:
    t_wall0 = time.perf_counter()

    with open(args.forget_set) as f:
        forget = json.load(f)
    forget_type = forget["forget_type"]
    if forget_type not in ("node", "edge"):
        raise ValueError(f"Unsupported forget_type: {forget_type}")
    run_dir = args.run_dir
    with open(os.path.join(run_dir, "args.json")) as f:
        run_args = json.load(f)
    model_type = run_args["model_type"]
    data_path = resolve_repo_path(run_args["data_path"])
    epochs = run_args["epochs"]
    hp = run_args["hp"]

    torch.manual_seed(int(run_args["seed"]))

    out_root = args.out_root or os.path.join(_REPO_ROOT, "Results", "shard-unlearns", DATASET)
    run_name = args.run_name or f"{timestamp_id()}_guts_{model_type}"
    out_dir = ensure_dir(os.path.join(out_root, run_name))
    log = run_logger(os.path.join(out_dir, "run_log.txt"))

    log(f"GUTS static-graph unlearn ({forget_type})  {time.strftime('%Y-%m-%d')}  source={relpath_for_log(run_dir)}"
        f"  out={relpath_for_log(out_dir)}  model={model_type}")
    log(f"  forget set {relpath_for_log(args.forget_set)}  {len(forget['records'])} targets")
    warning = source_mismatch_warning(forget, data_path)
    if warning:
        log(warning)

    sample = load_cora_sample(data_path)
    # Checks the forget set against the source split (raises on a mismatch) and gives the
    # clean graph used for evaluation unless --clean_data_path is given.
    clean_sample, clean_info = make_clean(sample, forget)
    if args.clean_data_path:
        clean_sample = load_cora_sample(args.clean_data_path)
        log(f"  clean graph {relpath_for_log(args.clean_data_path)}")
    else:
        log(f"  clean graph built in memory ({clean_info})")

    train_sample, train_orig_ids = induce_train_subgraph(sample)
    orig_to_local = {int(o): i for i, o in enumerate(train_orig_ids.tolist())}
    if forget_type == "node":
        deleted_local = torch.tensor(sorted(orig_to_local[int(r["nid"])] for r in forget["records"]),
                                     dtype=torch.long)
        deleted_pairs = None
    else:
        deleted_pairs = set()
        for r in forget["records"]:
            s, d = orig_to_local[int(r["src"])], orig_to_local[int(r["dst"])]
            deleted_pairs.add((s, d))
            deleted_pairs.add((d, s))
        deleted_local = None

    partition = load_pickle(os.path.join(run_dir, "partition.pkl"))
    shards = ShardBuilder().build(partition, train_sample)

    owned_sets = {sid: set(int(n) for n in nodes) for sid, nodes in partition.shards_nodes.items()}
    affected = []
    for shard_id, owned in owned_sets.items():
        if forget_type == "node":
            hit = bool(owned & set(deleted_local.tolist()))
        else:
            hit = any(u in owned and v in owned for u, v in deleted_pairs)
        if hit:
            affected.append(shard_id)
    log(f"  affected shards {len(affected)}/{len(shards)}")

    ckpt_dir_src = os.path.join(run_dir, "checkpoints")
    ckpt_dir_dst = ensure_dir(os.path.join(out_dir, "checkpoints"))

    shard_models = {}
    retrain_ms = {}
    for shard_id, data in shards.items():
        src_ckpt = os.path.join(ckpt_dir_src, f"shard_{shard_id}.pt")
        dst_ckpt = os.path.join(ckpt_dir_dst, f"shard_{shard_id}.pt")

        if shard_id not in affected:
            if not os.path.exists(src_ckpt):
                continue  # the source shard had no train node, hence no model
            shutil.copy(src_ckpt, dst_ckpt)
            model = build_model(model_type, train_sample.in_dim, sample.num_classes, hp)
            model.load_state_dict(torch.load(src_ckpt, map_location="cpu"))
            shard_models[shard_id] = model.to(args.device)
            continue

        if forget_type == "node":
            x_new, edge_index_new, train_mask_new, uids_new = _rebuild_for_node_deletion(data, deleted_local)
        else:
            x_new, edge_index_new, train_mask_new, uids_new = _rebuild_for_edge_deletion(data, deleted_pairs)

        t0 = time.perf_counter()
        model, _ = train_shard(model_type, hp, epochs, x_new, edge_index_new, train_sample.y[uids_new],
                               train_mask_new, sample.num_classes, args.device)
        retrain_ms[shard_id] = round((time.perf_counter() - t0) * 1000)
        if model is None:
            continue  # no train node left after deletion, no model
        torch.save(model.state_dict(), dst_ckpt)
        shard_models[shard_id] = model
    log(f"  {sum(s in shard_models for s in affected)}/{len(affected)} affected shards retrained  {sum(retrain_ms.values()) / 1000:.0f}s")

    t0_eval = time.perf_counter()
    ev = evaluate_aggregated(shard_models, clean_sample, args.device)
    agg_eval_ms = round((time.perf_counter() - t0_eval) * 1000)
    log(f"  {test_line(ev, len(shard_models))}  {agg_eval_ms / 1000:.1f}s")

    results = {
        "experiment_type": "static_graph_unlearn",
        "dataset": DATASET,
        "source_run_dir": repo_relpath(run_dir),
        "forget_set": repo_relpath(args.forget_set),
        "forget_type": forget_type,
        "model_type": model_type,
        "seed": int(run_args["seed"]),
        "num_shards": len(shards),
        "affected_shards": sorted(affected),
        "num_affected": len(affected),
        "aggregation": "uniform_average",
        "test_acc": ev["test_acc"],
        "test_f1_macro": ev["test_f1_macro"],
        "test_total": ev["test_total"],
        "timing_ms": {
            "per_shard_retrain_ms": {str(k): v for k, v in retrain_ms.items()},
            "per_shard_retrain": mean_std(retrain_ms.values(), extremes=True),
            "retrain_total": sum(retrain_ms.values()),
            "agg_eval": agg_eval_ms,
            "total": round((time.perf_counter() - t_wall0) * 1000),
        },
    }
    results_path = os.path.join(out_dir, "results_unlearn.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"done  {results['timing_ms']['total'] / 1000:.0f}s  -> {relpath_for_log(results_path)}")
    return results


def main(argv: Optional[List[str]] = None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()
