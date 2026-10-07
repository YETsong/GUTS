from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
import yaml

from guts.config import load_config, config_to_dict
from guts.cora.partitioner import make_cora_partitioner
from guts.shard import ShardBuilder
from guts.utils import ensure_dir, timestamp_id, save_pickle
from guts.cora.data import load_cora, load_cora_clean, induce_train_subgraph, DEFAULT_CORA_ROOT
from guts.cora.model import build_cora_model, resolve_model_hp
from guts.cora.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cora static-graph sharding pipeline (GUTS)")
    p.add_argument("--config", required=True)
    p.add_argument("--cora_root", default=DEFAULT_CORA_ROOT)
    p.add_argument("--clean_data_path", default="",
                   help="cora_clean.pt from guts/cora/clean_data.py; trains M_retrain on it "
                   "('{seed}' is filled per seed in multi-seed mode)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--model_type", default="graphsage", choices=["graphsage", "gcn", "gat", "gin"],
                   help="defaults are per model (MODEL_DEFAULTS); --lr etc. override them only when given")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026, help="seed (base seed in multi-seed mode)")
    p.add_argument("--run_name", default="", help="auto-generated if empty")
    p.add_argument("--num_gpus", type=int, default=0,
                   help="0 (default): single run; >0: multi-seed orchestrator spreading --num_seeds runs over GPUs")
    p.add_argument("--num_seeds", type=int, default=0,
                   help="number of runs in multi-seed mode (default: num_gpus)")
    return p.parse_args()


def _parse_overrides(pairs: list) -> dict:
    out = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def _run_tag(args: argparse.Namespace) -> str:
    return args.model_type


def train_one_shard(
    shard_data,
    sample,
    device: str,
    in_dim: int,
    num_classes: int,
    args: argparse.Namespace,
    log,
    shard_id: int,
):
    global_ids = shard_data.uids
    y_local = sample.y[global_ids]
    train_local = sample.train_mask[global_ids]
    test_local = sample.test_mask[global_ids]

    if train_local.sum() == 0:
        log(f"  [shard {shard_id}] no train nodes owned by this shard — skip")
        return None, {"train_nodes": 0, "train_ms": None, "epoch0_train_ms": None}

    hp = resolve_model_hp(
        args.model_type, hidden_dim=args.hidden_dim, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout, num_layers=args.num_layers,
    )
    model = build_cora_model(args.model_type, in_dim, num_classes,
                              hidden_dim=hp["hidden_dim"], dropout=hp["dropout"],
                              num_layers=hp["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    x = shard_data.x.to(device)
    edge_index = shard_data.edge_index.to(device)
    y = y_local.to(device)
    train_mask = train_local.to(device)

    epoch0_train_ms = 0.0
    train_ms_acc = 0.0
    for epoch in range(args.epochs):
        t0_epoch = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        dt = (time.perf_counter() - t0_epoch) * 1000
        if epoch == 0:
            epoch0_train_ms += dt
        else:
            train_ms_acc += dt

    log(f"  [shard {shard_id}] train_nodes={int(train_local.sum())} "
        f"test_nodes={int(test_local.sum())} (final epoch model, no val)")
    return model, {
        "train_nodes": int(train_local.sum()),
        "test_nodes": int(test_local.sum()),
        "train_ms": round(train_ms_acc),
        "epoch0_train_ms": round(epoch0_train_ms),
    }


def run_single(args: argparse.Namespace, run_dir: str) -> dict:
    import random as _random
    import numpy as _np
    torch.manual_seed(args.seed)
    _np.random.seed(args.seed)
    _random.seed(args.seed)

    cfg = load_config(args.config, overrides=_parse_overrides(args.override))
    dataset_name = "Cora"
    run_dir = ensure_dir(run_dir)
    log_path = os.path.join(run_dir, "log.txt")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("=" * 60)
    log(f"Cora Static Sharding Pipeline  strategy={cfg.partition.strategy}  K={cfg.partition.num_shards}  "
        f"model_type={args.model_type}")
    log(f"  run_dir: {run_dir}")
    log("=" * 60)

    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config_to_dict(cfg), f, default_flow_style=False)
    resolved_hp = resolve_model_hp(
        args.model_type, hidden_dim=args.hidden_dim, lr=args.lr,
        weight_decay=args.weight_decay, dropout=args.dropout, num_layers=args.num_layers,
    )
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump({
            "model_type": args.model_type,
            "seed": args.seed, "split_seed": args.seed,
            "clean_data_path": args.clean_data_path, "cora_root": args.cora_root,
            "epochs": args.epochs, "hp": resolved_hp,
        }, f, indent=2)

    if args.clean_data_path:
        sample = load_cora_clean(args.clean_data_path)
        log(f"Cora loaded (CLEAN/M_retrain): {args.clean_data_path}")
    else:
        sample = load_cora(args.cora_root, split_seed=args.seed)
    log(f"Cora loaded: nodes={sample.num_nodes} edges={sample.num_edges} "
        f"classes={sample.num_classes} split_seed={args.seed} "
        f"train={int(sample.train_mask.sum())} test={int(sample.test_mask.sum())}")

    train_sample, train_orig_ids = induce_train_subgraph(sample)
    log(f"G_train induced: nodes={train_sample.num_nodes} edges={train_sample.num_edges} "
        f"(partitioning/training on {train_sample.num_nodes}/{sample.num_nodes} nodes of G_full)")
    torch.save(train_orig_ids, os.path.join(run_dir, "train_orig_ids.pt"))

    partitioner = make_cora_partitioner(cfg)
    t0_partition = time.perf_counter()
    t0 = time.time()
    partition_labels = train_sample.y
    partition = partitioner.fit(train_sample, node_labels=partition_labels)
    qm = partition.quality_metrics
    log(f"STEP 1  partition done  K={cfg.partition.num_shards}  "
        f"balance={qm['balance']:.4f}  fairness={qm['fairness']:.4f}  {time.time()-t0:.1f}s")
    partition_ms = round((time.perf_counter() - t0_partition) * 1000)
    save_pickle(partition, os.path.join(run_dir, "partition.pkl"))

    t0_build = time.perf_counter()
    shards = ShardBuilder().build(partition, train_sample)
    build_ms = round((time.perf_counter() - t0_build) * 1000)
    log(f"STEP 2  shards built: {len(shards)}")

    shard_models: dict = {}
    shard_summaries: dict = {}
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
    for shard_id in sorted(shards.keys()):
        model, summary = train_one_shard(
            shards[shard_id], train_sample, args.device, train_sample.in_dim, train_sample.num_classes,
            args, log, shard_id,
        )
        summary["val_eval_ms"] = None
        shard_summaries[shard_id] = summary
        if model is not None:
            shard_models[shard_id] = model
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"shard_{shard_id}.pt"))

    # ── STEP 4: Aggregation eval (G_full, inductive) ───────────────────────
    t0_agg_eval = time.perf_counter()
    x_full = sample.x.to(args.device)
    edge_index_full = sample.edge_index.to(args.device)
    shard_weights = {sid: 1.0 / max(len(shard_models), 1) for sid in shard_models}
    log("STEP 4  aggregation eval on G_full (average softmax over all shard models)")
    prob_sum = torch.zeros(sample.num_nodes, sample.num_classes, device=args.device)
    n_models = 0
    for shard_id, model in shard_models.items():
        model.eval()
        with torch.no_grad():
            logits = model(x_full, edge_index_full)
            prob_sum += shard_weights.get(shard_id, 0.0) * F.softmax(logits, dim=-1)
        n_models += 1

    test_idx = sample.test_mask.nonzero(as_tuple=True)[0]
    if n_models > 0 and test_idx.numel() > 0:
        pred = prob_sum[test_idx].argmax(dim=-1).cpu()
        true = sample.y[test_idx]
        metrics = compute_metrics(pred, true)
        total = int(test_idx.numel())
    else:
        metrics = {"test_acc": 0.0, "test_f1_macro": 0.0}
        total = int(test_idx.numel())

    agg_eval_ms = round((time.perf_counter() - t0_agg_eval) * 1000)

    test_acc = metrics["test_acc"]
    log(f"STEP 4  test_acc={test_acc:.4f}  test_f1_macro={metrics['test_f1_macro']:.4f}  "
        f"(scored={total}/{int(sample.test_mask.sum())}, shard_models={n_models})")

    train_ms_values = [s["train_ms"] for s in shard_summaries.values() if s.get("train_ms") is not None]
    val_eval_ms_values = [s["val_eval_ms"] for s in shard_summaries.values() if s.get("val_eval_ms") is not None]
    epoch0_train_ms_values = [
        s["epoch0_train_ms"] for s in shard_summaries.values() if s.get("epoch0_train_ms") is not None
    ]
    timing_ms = {
        "partition": partition_ms,
        "build": build_ms,
        "epoch0_train_ms": _mean_std(epoch0_train_ms_values) if epoch0_train_ms_values else None,
        "per_shard_train": _mean_std(train_ms_values) if train_ms_values else None,
        "per_shard_val_eval": _mean_std(val_eval_ms_values) if val_eval_ms_values else None,
        "agg_eval_test": agg_eval_ms,
    }
    if train_ms_values:
        timing_ms["per_shard_train"]["min"] = min(train_ms_values)
        timing_ms["per_shard_train"]["max"] = max(train_ms_values)

    results = {
        "experiment_type": "cora_static_sharding",
        "dataset": dataset_name,
        "model_type": args.model_type,
        "strategy": cfg.partition.strategy,
        "num_shards": cfg.partition.num_shards,
        "partition_balance": partition.quality_metrics["balance"],
        "partition_fairness": partition.quality_metrics["fairness"],
        "shard_summaries": shard_summaries,
        "aggregation": "uniform_average",
        "aggregation_weights": {str(k): round(v, 6) for k, v in shard_weights.items()},
        "test_acc": metrics["test_acc"],
        "test_f1_macro": metrics["test_f1_macro"],
        "test_scored": total,
        "test_total": int(test_idx.numel()),
        "timing_ms": timing_ms,
    }
    save_pickle(results, os.path.join(run_dir, "results.pkl"))
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Results saved -> {run_dir}")
    return results


def _mean_std(values: list) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return {"mean": round(m, 6), "std": round(var ** 0.5, 6), "n": n}


def run_orchestrator(args: argparse.Namespace) -> None:
    num_gpus = max(1, args.num_gpus)
    num_seeds = args.num_seeds or num_gpus
    seeds = [args.seed + i for i in range(num_seeds)]

    cfg_probe = load_config(args.config, overrides=_parse_overrides(args.override))
    out_root = cfg_probe.output.root or "Results/shard-runs"
    _method_tag = str(cfg_probe.partition.strategy).lower()
    run_name = args.run_name or f"{timestamp_id()}_{_method_tag}_{_run_tag(args)}_multiseed"
    root_dir = ensure_dir(os.path.join(out_root, "Cora", run_name))
    print(f"Orchestrator: {num_seeds} seed(s) {seeds} across {num_gpus} GPU(s) -> {root_dir}")

    lanes: list = [[] for _ in range(num_gpus)]
    for i, seed in enumerate(seeds):
        lanes[i % num_gpus].append(seed)

    base_cmd = [
        sys.executable, os.path.abspath(__file__),
        "--config", args.config,
        "--cora_root", args.cora_root,
        "--model_type", args.model_type,
        "--epochs", str(args.epochs),
        "--device", "cuda:0",
        "--num_gpus", "0",
    ]
    for flag, val in [("--lr", args.lr), ("--weight_decay", args.weight_decay),
                       ("--hidden_dim", args.hidden_dim), ("--num_layers", args.num_layers),
                       ("--dropout", args.dropout)]:
        if val is not None:
            base_cmd += [flag, str(val)]
    if args.override:
        base_cmd += ["--override"] + args.override

    max_rounds = max(len(lane) for lane in lanes)
    for round_idx in range(max_rounds):
        procs = []
        for gpu_id, lane in enumerate(lanes):
            if round_idx >= len(lane):
                continue
            seed = lane[round_idx]
            child_run_name = f"{run_name}/seed_{seed}"
            cmd = base_cmd + ["--seed", str(seed), "--run_name", child_run_name]
            if args.clean_data_path:
                cmd += ["--clean_data_path", args.clean_data_path.format(seed=seed)]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(cmd, env=env)
            procs.append((seed, proc))
            print(f"  round {round_idx}: gpu {gpu_id} -> seed {seed} (pid {proc.pid})")
        for seed, proc in procs:
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED(exit={proc.returncode})"
            print(f"  seed {seed}: {status}")

    test_accs, f1s, balances, fairnesses = [], [], [], []
    per_seed = {}
    for seed in seeds:
        result_path = os.path.join(root_dir, f"seed_{seed}", "results.json")
        if not os.path.exists(result_path):
            print(f"  [WARN] seed {seed}: no results.json — excluded from aggregation")
            continue
        with open(result_path) as f:
            r = json.load(f)
        per_seed[seed] = r
        test_accs.append(r["test_acc"])
        if "test_f1_macro" in r:
            f1s.append(r["test_f1_macro"])
        if "partition_balance" in r:
            balances.append(r["partition_balance"])
            fairnesses.append(r["partition_fairness"])

    summary = {
        "experiment_type": "cora_static_sharding_multiseed",
        "num_seeds": len(seeds),
        "num_gpus": num_gpus,
        "seeds": seeds,
        "test_acc": _mean_std(test_accs) if test_accs else None,
        "test_f1_macro": _mean_std(f1s) if f1s else None,
        "partition_balance": _mean_std(balances) if balances else None,
        "partition_fairness": _mean_std(fairnesses) if fairnesses else None,
        "per_seed": per_seed,
    }
    with open(os.path.join(root_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if test_accs:
        s = summary["test_acc"]
        print(f"FINAL (n={s['n']})  test_acc={s['mean']:.4f}±{s['std']:.4f}")
        if f1s:
            sf = summary["test_f1_macro"]
            print(f"       test_f1_macro={sf['mean']:.4f}±{sf['std']:.4f}")
    print(f"Summary saved -> {os.path.join(root_dir, 'summary.json')}")

    if len(per_seed) < len(seeds):
        raise RuntimeError(
            f"{len(seeds) - len(per_seed)}/{len(seeds)} seeds have no results.json "
            f"(see [WARN] above) — run_dir={root_dir}"
        )


def main() -> None:
    args = parse_args()
    if args.num_gpus and args.num_gpus > 0:
        run_orchestrator(args)
        return

    cfg_probe = load_config(args.config, overrides=_parse_overrides(args.override))
    dataset_name = "Cora"
    _method_tag = str(cfg_probe.partition.strategy).lower()
    run_name = args.run_name or f"{timestamp_id()}_{_method_tag}_{_run_tag(args)}"
    out_root = cfg_probe.output.root or "Results/shard-runs"
    run_dir = os.path.join(out_root, dataset_name, run_name)
    run_single(args, run_dir)


if __name__ == "__main__":
    main()
