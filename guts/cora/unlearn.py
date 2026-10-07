from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

from guts.config import load_config
from guts.shard import ShardBuilder
from guts.utils import ensure_dir, load_pickle, timestamp_id, find_latest_run
from guts.cora.data import load_cora, load_cora_clean, induce_train_subgraph, DEFAULT_CORA_ROOT
from guts.cora.model import build_cora_model
from guts.cora.metrics import compute_metrics

from guts.cora.clean_data import build_one as build_clean_one


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cora M_unlearn (shrink+remap)")
    p.add_argument("--run_dir", default="", help="source run seed directory (e.g. .../seed_2026); auto-detected if empty")
    p.add_argument("--source_run_dir", default="", help="source multi-seed parent directory (contains seed_N/); auto-detected if empty")
    p.add_argument("--config", default="", help="yaml of the source run (for auto-detection)")
    p.add_argument("--model_type", default="", choices=["", "graphsage", "gcn", "gat", "gin"], help="model type (for auto-detection)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides (for auto-detection), e.g. partition.num_shards=20")
    p.add_argument("--forget_set", default="", help="forget set JSON (single run)")
    p.add_argument("--forget_set_template", default="", help="forget set path with '{seed}' (multi-seed)")
    p.add_argument("--clean_data_path", default="", help="clean G_full for evaluation; built on the fly if empty")
    p.add_argument("--out_root", default="", help="default: shard-unlearns/Cora/ next to the source")
    p.add_argument("--run_name", default="", help="auto-generated if empty")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026, help="ignored in single runs (forget set seed is used); base seed in multi-seed mode")
    p.add_argument("--num_gpus", type=int, default=0,
                   help="0 (default): single run; >0: multi-seed orchestrator (as in cora_pipeline.py)")
    p.add_argument("--num_seeds", type=int, default=0, help="number of runs in multi-seed mode (default: num_gpus)")
    return p.parse_args()


def _parse_overrides(pairs: list) -> dict:
    out = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def _resolve_source_seed_dir(args: argparse.Namespace, seed: int) -> str:
    if args.run_dir:
        return args.run_dir
    parent = args.source_run_dir
    if not parent:
        if not (args.config and args.model_type):
            raise ValueError("--config and --model_type are required to auto-detect the source run "
                         "when --run_dir/--source_run_dir is not given.")
        cfg_probe = load_config(args.config, overrides=_parse_overrides(args.override))
        method = str(cfg_probe.partition.strategy).lower()
        pattern = f"*_{method}_{args.model_type}_multiseed"
        parent = find_latest_run(os.path.join(_REPO_ROOT, "Results", "shard-runs", "Cora"), pattern)
        if parent is None:
            raise FileNotFoundError(f"No matching source run: {pattern} (under Results/shard-runs/Cora/)")
    return os.path.join(parent, f"seed_{seed}")


def _rebuild_for_node_deletion(data, deleted_local: torch.Tensor):
    keep_mask = ~torch.isin(data.uids, deleted_local)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    new_edge_index, _, _ = subgraph(keep_idx, data.edge_index, relabel_nodes=True,
                                     num_nodes=data.x.size(0), return_edge_mask=True)
    return data.x[keep_idx], new_edge_index, data.train_mask[keep_idx], data.uids[keep_idx]


def _rebuild_for_edge_deletion(data, deleted_pairs: set):
    u_local = data.uids[data.edge_index[0]]
    v_local = data.uids[data.edge_index[1]]
    keep_edge = torch.tensor(
        [(int(u), int(v)) not in deleted_pairs for u, v in zip(u_local.tolist(), v_local.tolist())],
        dtype=torch.bool,
    )
    return data.x, data.edge_index[:, keep_edge], data.train_mask, data.uids


def run_single(args: argparse.Namespace) -> dict:
    t_wall0 = time.time()

    if not args.forget_set:
        raise ValueError("--forget_set is required (filled automatically in multi-seed mode)")
    with open(args.forget_set) as f:
        forget = json.load(f)
    forget_type = forget["forget_type"]
    if forget_type not in ("node", "edge"):
        raise ValueError(f"Unsupported forget_type: {forget_type}")
    seed = int(forget["seed"])

    run_dir = _resolve_source_seed_dir(args, seed)
    with open(os.path.join(run_dir, "args.json")) as f:
        run_args = json.load(f)
    model_type = run_args["model_type"]
    split_seed = run_args["split_seed"]
    epochs = run_args["epochs"]
    hp = run_args["hp"]

    torch.manual_seed(split_seed)

    cfg = load_config(os.path.join(run_dir, "config_used.yaml"))

    out_root = args.out_root or os.path.join(_REPO_ROOT, "Results", "shard-unlearns", "Cora")
    run_name = args.run_name or f"{timestamp_id()}_{str(cfg.partition.strategy).lower()}_{model_type}"
    out_dir = ensure_dir(os.path.join(out_root, run_name))
    log_path = os.path.join(out_dir, "log.txt")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"run_dir={run_dir}  model_type={model_type}  "
        f"forget_set={args.forget_set}  forget_type={forget_type}  targets={len(forget['records'])}")

    sample = load_cora(split_seed=split_seed)
    train_sample, train_orig_ids = induce_train_subgraph(sample)
    orig_to_local = {int(o): i for i, o in enumerate(train_orig_ids.tolist())}

    if forget_type == "node":
        forget_orig_ids = [int(r["nid"]) for r in forget["records"]]
        missing = [o for o in forget_orig_ids if o not in orig_to_local]
        if missing:
            raise ValueError(f"forget set contains non-train node ids: {missing[:5]}...")
        deleted_local = torch.tensor(sorted(orig_to_local[o] for o in forget_orig_ids), dtype=torch.long)
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
            deleted_set = set(deleted_local.tolist())
            hit = bool(owned & deleted_set)
        else:
            hit = any(u in owned and v in owned for u, v in deleted_pairs)
        if hit:
            affected.append(shard_id)
    log(f"Affected shards: {len(affected)}/{len(shards)}  {sorted(affected)}")

    if args.clean_data_path:
        clean_sample = load_cora_clean(args.clean_data_path)
    else:
        clean_path = build_clean_one(DEFAULT_CORA_ROOT, args.forget_set, os.path.join(_REPO_ROOT, "processed", "Cora_clean"))
        clean_sample = load_cora_clean(clean_path)

    ckpt_dir_src = os.path.join(run_dir, "checkpoints")
    ckpt_dir_dst = ensure_dir(os.path.join(out_dir, "checkpoints"))

    shard_models = {}
    retrain_seconds = {}
    for shard_id, data in shards.items():
        src_ckpt = os.path.join(ckpt_dir_src, f"shard_{shard_id}.pt")
        dst_ckpt = os.path.join(ckpt_dir_dst, f"shard_{shard_id}.pt")

        if shard_id not in affected:
            shutil.copy(src_ckpt, dst_ckpt)
            model = build_cora_model(model_type, train_sample.in_dim, sample.num_classes,
                                      hidden_dim=hp["hidden_dim"], dropout=hp["dropout"],
                                      num_layers=hp["num_layers"])
            model.load_state_dict(torch.load(src_ckpt, map_location="cpu"))
            shard_models[shard_id] = model.to(args.device)
            continue

        if forget_type == "node":
            x_new, edge_index_new, train_mask_new, uids_new = _rebuild_for_node_deletion(data, deleted_local)
        else:
            x_new, edge_index_new, train_mask_new, uids_new = _rebuild_for_edge_deletion(data, deleted_pairs)

        y_local = train_sample.y[uids_new]
        train_local = train_mask_new

        t0 = time.time()
        model = build_cora_model(model_type, train_sample.in_dim, sample.num_classes,
                                  hidden_dim=hp["hidden_dim"], dropout=hp["dropout"],
                                  num_layers=hp["num_layers"]).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

        x_dev = x_new.to(args.device)
        ei_dev = edge_index_new.to(args.device)
        y_dev = y_local.to(args.device)
        tm_dev = train_local.to(args.device)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x_dev, ei_dev)
            loss = F.cross_entropy(logits[tm_dev], y_dev[tm_dev])
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), dst_ckpt)
        shard_models[shard_id] = model
        retrain_seconds[shard_id] = round(time.time() - t0, 2)
        log(f"  [shard {shard_id}] RETRAINED  nodes {int(data.uids.numel())}->{int(uids_new.numel())}  "
            f"edges {int(data.edge_index.size(1))}->{int(edge_index_new.size(1))}  {retrain_seconds[shard_id]}s")

    t0_predict = time.time()
    shard_weights = {sid: 1.0 / max(len(shard_models), 1) for sid in shard_models}

    x_full = clean_sample.x.to(args.device)
    edge_index_full = clean_sample.edge_index.to(args.device)
    prob_sum = torch.zeros(clean_sample.num_nodes, clean_sample.num_classes, device=args.device)
    for shard_id, model in shard_models.items():
        model.eval()
        with torch.no_grad():
            logits = model(x_full, edge_index_full)
            prob_sum += shard_weights.get(shard_id, 0.0) * F.softmax(logits, dim=-1)

    test_idx = clean_sample.test_mask.nonzero(as_tuple=True)[0]
    pred = prob_sum[test_idx].argmax(dim=-1).cpu()
    true = clean_sample.y[test_idx]
    metrics = compute_metrics(pred, true)
    predict_seconds = round(time.time() - t0_predict, 2)

    results = {
        "experiment_type": "cora_unlearn",
        "source_run_dir": os.path.abspath(run_dir),
        "forget_set": args.forget_set,
        "forget_type": forget_type,
        "model_type": model_type,
        "seed": seed,
        "num_shards": len(shards),
        "affected_shards": sorted(affected),
        "num_affected": len(affected),
        "aggregation": "uniform_average",
        "retrain_seconds_per_shard": retrain_seconds,
        "predict_seconds": predict_seconds,
        "total_wall_seconds": round(time.time() - t_wall0, 2),
        "test_acc": metrics["test_acc"],
        "test_f1_macro": metrics["test_f1_macro"],
        "test_total": int(test_idx.numel()),
    }
    with open(os.path.join(out_dir, "results_unlearn.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"test_acc={metrics['test_acc']:.4f}  test_f1_macro={metrics['test_f1_macro']:.4f}  "
        f"affected={len(affected)}/{len(shards)}  wall={results['total_wall_seconds']}s")
    log(f"Results saved -> {out_dir}")
    return results


def _mean_std(values: list) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return {"mean": round(m, 6), "std": round(var ** 0.5, 6), "n": n}


def run_orchestrator(args: argparse.Namespace) -> None:
    if not args.forget_set_template or "{seed}" not in args.forget_set_template:
        raise ValueError("multi-seed mode requires a '{seed}' placeholder in --forget_set_template")
    if not (args.config and args.model_type):
        raise ValueError("multi-seed mode requires --config/--model_type (source auto-detection and run_name)")

    cfg_probe = load_config(args.config, overrides=_parse_overrides(args.override))
    method = str(cfg_probe.partition.strategy).lower()

    run_name = args.run_name or f"{timestamp_id()}_{method}_{args.model_type}_multiseed"
    out_root = args.out_root or os.path.join(_REPO_ROOT, "Results", "shard-unlearns", "Cora")
    root_dir = ensure_dir(os.path.join(out_root, run_name))

    num_gpus = max(1, args.num_gpus)
    num_seeds = args.num_seeds or num_gpus
    seeds = [args.seed + i for i in range(num_seeds)]
    print(f"Orchestrator: {num_seeds} seed(s) {seeds} across {num_gpus} GPU(s) -> {root_dir}")

    lanes: list = [[] for _ in range(num_gpus)]
    for i, seed in enumerate(seeds):
        lanes[i % num_gpus].append(seed)

    base_cmd = [
        sys.executable, os.path.abspath(__file__),
        "--config", args.config, "--model_type", args.model_type,
        "--device", "cuda:0", "--num_gpus", "0", "--out_root", out_root,
    ]
    if args.override:
        base_cmd += ["--override"] + args.override
    if args.source_run_dir:
        base_cmd += ["--source_run_dir", args.source_run_dir]
    if args.clean_data_path:
        base_cmd += ["--clean_data_path", args.clean_data_path]

    max_rounds = max(len(lane) for lane in lanes)
    for round_idx in range(max_rounds):
        procs = []
        for gpu_id, lane in enumerate(lanes):
            if round_idx >= len(lane):
                continue
            seed = lane[round_idx]
            forget_set = args.forget_set_template.format(seed=seed)
            child_run_name = f"{run_name}/seed_{seed}"
            cmd = base_cmd + ["--forget_set", forget_set, "--run_name", child_run_name]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(cmd, env=env, cwd=_REPO_ROOT)
            procs.append((seed, proc))
            print(f"  round {round_idx}: gpu {gpu_id} -> seed {seed} (pid {proc.pid})")
        for seed, proc in procs:
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED(exit={proc.returncode})"
            print(f"  seed {seed}: {status}")

    test_accs, f1s = [], []
    per_seed = {}
    for seed in seeds:
        result_path = os.path.join(root_dir, f"seed_{seed}", "results_unlearn.json")
        if not os.path.exists(result_path):
            print(f"  [WARN] seed {seed}: no results_unlearn.json — excluded from aggregation")
            continue
        with open(result_path) as f:
            r = json.load(f)
        per_seed[seed] = r
        test_accs.append(r["test_acc"])
        f1s.append(r["test_f1_macro"])

    summary = {
        "experiment_type": "cora_unlearn_multiseed",
        "num_seeds": len(seeds),
        "num_gpus": num_gpus,
        "seeds": seeds,
        "test_acc": _mean_std(test_accs) if test_accs else None,
        "test_f1_macro": _mean_std(f1s) if f1s else None,
        "per_seed": per_seed,
    }
    with open(os.path.join(root_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if test_accs:
        s = summary["test_acc"]
        print(f"FINAL (n={s['n']})  test_acc={s['mean']:.4f}±{s['std']:.4f}")
    print(f"Summary saved -> {os.path.join(root_dir, 'summary.json')}")

    if len(per_seed) < len(seeds):
        raise RuntimeError(
            f"{len(seeds) - len(per_seed)}/{len(seeds)} seeds have no results_unlearn.json "
            f"(see [WARN] above) — run_dir={root_dir}"
        )


def main() -> None:
    args = parse_args()
    if args.num_gpus and args.num_gpus > 0:
        run_orchestrator(args)
        return
    run_single(args)


if __name__ == "__main__":
    main()
