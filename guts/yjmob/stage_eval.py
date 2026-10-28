"""STEP 3 of guts/yjmob/pipeline.py: shard routing evaluation.

Each test step of day d is sent to one shard (majority vote over the partitions
of the training days before d) and predicted by that shard's model from its
rolling graph of d. Also used by the unlearning scripts to evaluate unlearned runs.
Writes its log to the given eval log file only; callers print the result.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import glob
import json
import pickle
import shutil
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from guts.method.reporting import format_metrics
from guts.method.utils import append_log_synced, relpath_for_log
from guts.yjmob.model.dataset import local_graph
from guts.yjmob.model.model import NextGridPredictor
from guts.yjmob.model.trainer import target_ranks


def _gs_load_shard(shard_run_dir: str, processed_dir: str, device: torch.device, window_size: int) -> Dict:
    """Load a shard model (best.pt, or shard_run_best.pt linked by unlearning for untouched shards)."""
    with open(os.path.join(shard_run_dir, "args.json")) as f:
        shard_args = json.load(f)
    meta = torch.load(os.path.join(processed_dir, "next_location_meta.pt"), map_location="cpu")
    num_nodes = int(meta["num_nodes"])

    model = NextGridPredictor(
        num_nodes=num_nodes,
        node_feat_dim=int(meta["node_feat_dim"]),
        id_embedding_dim=int(shard_args["id_embedding_dim"]),
        sage_hidden_dim=int(shard_args["sage_hidden_dim"]),
        sage_out_dim=int(shard_args["sage_out_dim"]),
        sage_layers=int(shard_args["sage_layers"]),
        sage_dropout=float(shard_args["sage_dropout"]),
        gru_hidden_dim=int(shard_args["gru_hidden_dim"]),
        gru_layers=int(shard_args["gru_layers"]),
        gru_dropout=float(shard_args["gru_dropout"]),
        sage_weighted=bool(int(shard_args["sage_weighted"])),
        sage_edge_weight_transform=str(shard_args["sage_edge_weight_transform"]),
        sage_edge_chunk_size=int(shard_args["sage_edge_chunk_size"]),
    )
    ckpt_dir = os.path.join(shard_run_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "shard_run_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no best.pt or shard_run_best.pt in {relpath_for_log(ckpt_dir)}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state"])
    model.to(device).eval()

    rolling_graph_dir = os.path.join(processed_dir, f"rolling_graphs_w{int(window_size)}")
    if not os.path.isdir(rolling_graph_dir):
        raise FileNotFoundError(f"rolling graphs not found: {relpath_for_log(rolling_graph_dir)}")
    return {
        "model": model,
        "device": device,
        "num_nodes": num_nodes,
        "rolling_graph_dir": rolling_graph_dir,
        "graph_z_cache": {},
    }


@torch.no_grad()
def _gs_get_rolling_graph_z(shard: Dict, target_day: int, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """GraphSAGE embeddings of a shard's rolling graph and its node_lookup (global id -> row, -1 outside).

    Reads the graph through the same ``local_graph`` as training, so cells absent
    from the rolling graph get a zero embedding in both.
    """
    cache_key = (int(target_day), int(window_size))
    if cache_key in shard["graph_z_cache"]:
        return shard["graph_z_cache"][cache_key]

    device = shard["device"]
    model = shard["model"]
    path = os.path.join(shard["rolling_graph_dir"], f"target_day_{int(target_day):04d}.pt")
    g = local_graph(torch.load(path, map_location="cpu"), shard["num_nodes"])
    edge_weight = g["edge_weight"]
    if edge_weight is not None:
        edge_weight = edge_weight.to(device, dtype=torch.float32)
    # The encoder ignores edge_weight unless the model was trained with --sage_weighted 1.
    z = model.encode_graph(g["x"].to(device), g["edge_index"].to(device), edge_weight=edge_weight)
    out = (z, g["node_lookup"].to(device))
    shard["graph_z_cache"][cache_key] = out
    return out


@torch.no_grad()
def _infer_target_ranks(
    model,
    graph_z: torch.Tensor,
    node_lookup: torch.Tensor,
    segments: List[List[int]],
    targets: np.ndarray,
    device: torch.device,
    infer_batch: int = 1024,
) -> np.ndarray:
    """Rank of each step's target over all cells (float32 logits, same rule as training)."""
    n = len(segments)
    ranks = np.empty(n, dtype=np.int64)

    for start in range(0, n, infer_batch):
        batch_segs = segments[start:start + infer_batch]
        B = len(batch_segs)
        lengths = torch.tensor([len(seg) for seg in batch_segs], dtype=torch.long)
        padded = torch.zeros(B, int(lengths.max()), dtype=torch.long, device=device)
        for bi, seg in enumerate(batch_segs):
            padded[bi, :len(seg)] = torch.tensor(seg, dtype=torch.long, device=device)

        packed = nn.utils.rnn.pack_padded_sequence(
            model.id_embedding(padded), lengths, batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(model.gru(packed)[0], batch_first=True)
        rows = torch.arange(B, device=device)
        last_idxs = (lengths - 1).to(device)
        last_cells = padded[rows, last_idxs]
        h = out[rows, last_idxs] + model.sage_beta * model.graph_embeddings_for_inputs(graph_z, last_cells, node_lookup)
        logits = model.logits(h).float()
        tgt = torch.as_tensor(targets[start:start + B], dtype=torch.long, device=device)
        ranks[start:start + B] = target_ranks(logits, tgt).long().cpu().numpy()

    return ranks


def _sequence_part_paths(processed_dir: str, split: str) -> Tuple[List[str], dict, int]:
    """Sequence parts registered in the processed dataset metadata, the split index, and num_nodes."""
    meta_path = os.path.join(os.path.abspath(processed_dir), "next_location_meta.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"next_location_meta.pt not found: {relpath_for_log(meta_path)}")
    meta = torch.load(meta_path, map_location="cpu")
    split_meta = meta.get("sequence_index", {}).get(split)
    if not split_meta or not split_meta.get("parts"):
        raise ValueError(f"sequence_index[{split!r}] is empty: {relpath_for_log(meta_path)}")

    paths = []
    for part in split_meta["parts"]:
        path = os.path.join(os.path.abspath(processed_dir), part["path"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"registered sequence part not found: {relpath_for_log(path)}")
        paths.append(path)
    return paths, split_meta, int(meta["num_nodes"])


# ──────────────────────────────────────────────────────────────────────────────
# Shard routing evaluation (rolling graphs)
#
# Every test step (cells[:t+1] -> cells[t+1]) of day d goes to one shard: the
# shard that held the source cell cells[t] most often in the partitions of the
# training days before d (ties: the most recent day). That shard's model
# predicts from its rolling graph of d. Only days before d are used, so no
# partition of day d or later informs the routing.
# ──────────────────────────────────────────────────────────────────────────────

def _build_shard_routing_by_day(sharding_dir: str, eval_days: List[int]) -> Dict[int, Dict[int, int]]:
    """{day d: {node: shard}} by majority vote over the partitions of the days before d."""
    day_pkls = {}
    for path in glob.glob(os.path.join(sharding_dir, "partition", "day*.pkl")):
        day_pkls[int(os.path.basename(path)[len("day"):-len(".pkl")])] = path
    if not day_pkls:
        raise FileNotFoundError(f"no partition/day*.pkl under {relpath_for_log(sharding_dir)}")

    votes: Dict[int, Dict[int, List[int]]] = defaultdict(dict)  # node -> shard -> [count, last day]
    routes: Dict[int, Dict[int, int]] = {}
    days_seen = sorted(day_pkls)
    next_pkl = 0
    for d in sorted(set(eval_days)):
        while next_pkl < len(days_seen) and days_seen[next_pkl] < d:
            day = days_seen[next_pkl]
            with open(day_pkls[day], "rb") as f:
                part_obj = pickle.load(f)
            for node, shard in dict(part_obj.node_to_shard).items():
                entry = votes[int(node)].setdefault(int(shard), [0, day])
                entry[0] += 1
                entry[1] = day
            next_pkl += 1
        routes[d] = {
            node: max(by_shard.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
            for node, by_shard in votes.items()
        }
    return routes


def _eval_worker(rank: int, world_size: int, worker_args: dict) -> None:
    cuda_dev = worker_args["cuda_devices"][rank]
    torch.cuda.set_device(cuda_dev)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{cuda_dev}")

    gs_dir          = os.path.join(worker_args["run_dir"], "graphsage_runs")
    processed_base  = worker_args["processed_base"]
    num_global      = worker_args["num_global"]
    split           = worker_args["split"]
    window_size     = int(worker_args["window_size"])
    routes          = worker_args["routes"]  # {day: {global_node_id: shard_id}}
    result_path     = worker_args["result_paths"][rank]
    infer_batch     = worker_args["infer_batch"]
    log_path        = worker_args["log_path"]

    tag = f"[GPU{cuda_dev}]"

    def log_w(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {tag} {msg}"
        if log_path:
            append_log_synced(log_path, line)

    log_w(f"worker {rank + 1}/{world_size}  split={split}  window={window_size}")

    all_sessions_flat: List[Tuple[List[int], int]] = []
    for part_path in worker_args["part_paths"]:
        part = torch.load(part_path, map_location="cpu")
        for item in part:
            cells = item["cells"]
            day = int(item["day"])
            if len(cells) >= 2 and day >= window_size:
                all_sessions_flat.append((cells, day))
    my_sessions = all_sessions_flat[rank::world_size]
    log_w(f"  sessions {len(my_sessions):,} (day >= {window_size})")

    all_steps: List[Tuple[List[int], int, int, int]] = []  # (cells, t, target, day)
    step_shard: List[Optional[int]] = []
    n_skipped = 0
    for cells, day in my_sessions:
        route = routes[day]
        for t in range(len(cells) - 1):
            target = int(cells[t + 1])
            if not (0 <= target < num_global):
                n_skipped += 1
                continue
            all_steps.append((cells, t, target, day))
            step_shard.append(route.get(int(cells[t])))

    N = len(all_steps)
    n_routable = sum(1 for k in step_shard if k is not None)
    n_unroutable = N - n_routable
    log_w(f"  steps {N:,}  routable {n_routable:,}  unroutable {n_unroutable:,}  skipped {n_skipped:,}")

    shard_day_to_steps: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, k in enumerate(step_shard):
        if k is not None:
            shard_day_to_steps[(k, all_steps[idx][3])].append(idx)

    cpu_device = torch.device("cpu")
    t_infer = time.time()
    shard_keys = sorted({k for (k, _d) in shard_day_to_steps})
    log_w(f"  inference  {len(shard_keys)} shards  {len(shard_day_to_steps)} (shard, day) groups")
    marks = {max(1, round(len(shard_keys) * q / 4)) for q in (1, 2, 3)}

    r_acc = {1: 0, 5: 0, 10: 0}; r_rr = 0.0; r_steps = 0
    n_out_of_graph = 0  # evaluated steps whose input cell is absent from the shard's rolling graph (z = 0)
    for shard_i, k in enumerate(shard_keys, start=1):
        sd = _gs_load_shard(
            os.path.join(gs_dir, f"shard_{k}"), os.path.join(processed_base, f"shard_{k}"),
            cpu_device, window_size,
        )
        sd["device"] = device
        sd["model"].to(device).eval()

        for d in sorted({d for (kk, d) in shard_day_to_steps if kk == k}):
            step_indices = shard_day_to_steps[(k, d)]
            graph_z, node_lookup = _gs_get_rolling_graph_z(sd, d, window_size)
            segments = [all_steps[idx][0][: all_steps[idx][1] + 1] for idx in step_indices]
            targets = [all_steps[idx][2] for idx in step_indices]
            last_cells = torch.tensor([seg[-1] for seg in segments], dtype=torch.long, device=device)
            n_out_of_graph += int((node_lookup[last_cells] < 0).sum().item())
            targets = np.array(targets, dtype=np.int64)
            rank = _infer_target_ranks(
                sd["model"], graph_z, node_lookup, segments, targets, device, infer_batch=infer_batch,
            )

            for kv in (1, 5, 10):
                r_acc[kv] += int((rank <= kv).sum())
            r_rr += float((1.0 / rank).sum())
            r_steps += len(step_indices)

        sd["model"].cpu()
        del sd
        torch.cuda.empty_cache()
        if shard_i in marks:
            log_w(f"  {shard_i}/{len(shard_keys)} shards  {r_steps:,} steps  {time.time() - t_infer:.0f}s")

    wall_s = time.time() - t_infer
    log_w(f"  done  {r_steps:,} steps  out of graph {n_out_of_graph:,} ({n_out_of_graph / max(r_steps, 1):.2%})"
          f"  {wall_s:.0f}s")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torch.save({
        "r_acc": r_acc, "r_rr": r_rr, "r_steps": r_steps, "n_out_of_graph": n_out_of_graph,
        "n_routable": n_routable, "n_unroutable": n_unroutable, "n_skipped": n_skipped,
        "wall_s": wall_s,
    }, result_path)


def result_line(result: dict) -> str:
    """'MRR ... ACC@10 ...  routable 99.97%' for a run_shard_routing_eval result."""
    return f"{format_metrics(result['routable'])}  routable {100 * result['coverage']['routable_ratio']:.2f}%"


def run_shard_routing_eval(
    run_dir: str,
    full_data_dir: str,
    eval_log_path: str,
    window_size: int = 3,
    split: str = "test",
    device: Optional[torch.device] = None,
    num_gpus_eval: int = 1,
    sharding_dir: Optional[str] = None,
) -> dict:
    """Shard routing evaluation of a trained run on ``split`` of ``full_data_dir``; see the section note above."""
    def log(msg: str) -> None:  # eval log only
        append_log_synced(eval_log_path, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    _sharding_dir  = sharding_dir or os.path.join(run_dir, "Sharding")
    processed_base = os.path.join(_sharding_dir, "processed")
    if not os.path.isdir(processed_base):
        raise FileNotFoundError(f"Sharding/processed/ not found: {relpath_for_log(processed_base)}")
    available_shards = sorted(
        int(d.split("_", 1)[1])
        for d in os.listdir(processed_base)
        if d.startswith("shard_") and os.path.isdir(os.path.join(processed_base, d))
    )
    if not available_shards:
        raise FileNotFoundError(f"no shard_* dirs in {relpath_for_log(processed_base)}")

    part_paths, split_meta, num_global = _sequence_part_paths(full_data_dir, split)
    total_sessions = 0
    total_steps = 0
    eval_days = set()
    for part_path in part_paths:
        part = torch.load(part_path, map_location="cpu")
        total_sessions += sum(1 for item in part if len(item.get("cells", [])) >= 2)
        total_steps += sum(max(0, len(item.get("cells", [])) - 1) for item in part)
        eval_days.update(int(item["day"]) for item in part if int(item["day"]) >= window_size)
    expected = (int(split_meta["sessions"]), int(split_meta["steps"]))
    if (total_sessions, total_steps) != expected:
        raise ValueError(
            f"{split} metadata mismatch: actual sessions/steps="
            f"{total_sessions}/{total_steps}, expected={expected[0]}/{expected[1]}"
        )

    routes = _build_shard_routing_by_day(_sharding_dir, sorted(eval_days))
    log(f"shard routing evaluation  run={relpath_for_log(run_dir)}  split={split}  window={window_size}"
        f"  {len(available_shards)} shards  {total_sessions:,} sessions")
    log(f"  routing: majority vote over the training days before each test day "
        f"({len(routes)} test days, up to {max((len(r) for r in routes.values()), default=0):,} routable nodes)")
    log(f"  rank: exact rank of the target over all {num_global:,} cells (float32 logits)")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    common_args = {
        "run_dir":            run_dir,
        "processed_base":     processed_base,
        "part_paths":         part_paths,
        "num_global":         num_global,
        "split":              split,
        "window_size":        int(window_size),
        "routes":             routes,
        "log_path":           eval_log_path,
        "infer_batch":        1024,
    }

    import uuid as _uuid
    uid = _uuid.uuid4().hex[:8]
    _scratch_root = os.path.join(run_dir, "_eval_tmp")
    os.makedirs(_scratch_root, exist_ok=True)

    if num_gpus_eval > 1:
        result_paths = [os.path.join(_scratch_root, f"eval_rank{i}_{uid}.pt") for i in range(num_gpus_eval)]
        worker_args = dict(common_args, cuda_devices=list(range(num_gpus_eval)), result_paths=result_paths)
        log(f"  {num_gpus_eval} GPU workers")
        mp.spawn(_eval_worker, args=(num_gpus_eval, worker_args), nprocs=num_gpus_eval, join=True)
    else:
        cuda_dev_sg = device.index if device.type == "cuda" and device.index is not None else 0
        result_paths = [os.path.join(_scratch_root, f"eval_rank0_{uid}.pt")]
        worker_args = dict(common_args, cuda_devices=[cuda_dev_sg], result_paths=result_paths)
        log("  1 GPU worker")
        _eval_worker(0, 1, worker_args)

    r_acc_m = {1: 0, 5: 0, 10: 0}; r_rr_m = 0.0; r_steps_m = 0
    n_routable_m = 0; n_unroutable_m = 0; n_skipped_m = 0; n_out_of_graph_m = 0
    max_wall = 0.0
    for rp in result_paths:
        res = torch.load(rp, map_location="cpu")
        for kv in [1, 5, 10]:
            r_acc_m[kv] += res["r_acc"][kv]
        r_rr_m += res["r_rr"]; r_steps_m += res["r_steps"]
        n_routable_m   += res["n_routable"]
        n_unroutable_m += res["n_unroutable"]
        n_skipped_m    += res["n_skipped"]
        n_out_of_graph_m += res["n_out_of_graph"]
        max_wall = max(max_wall, res["wall_s"])
    shutil.rmtree(_scratch_root, ignore_errors=True)

    def _metrics_m(acc_d, rr, steps):
        D = max(steps, 1)
        return {
            "acc@1":  round(acc_d[1] / D, 6),
            "acc@5":  round(acc_d[5] / D, 6),
            "acc@10": round(acc_d[10] / D, 6),
            "mrr":    round(rr / D, 6),
            "support": steps,
        }

    N_total_m = n_routable_m + n_unroutable_m
    result_m = {
        "approach": "shard_routing_rolling_graph",
        "routing": "majority_vote_over_previous_days",
        "window_size": int(window_size),
        "routable": _metrics_m(r_acc_m, r_rr_m, r_steps_m),
        "coverage": {
            "n_total":        N_total_m,
            "n_routable":     n_routable_m,
            "n_unroutable":   n_unroutable_m,
            "n_skipped":      n_skipped_m,
            "routable_ratio": round(n_routable_m / max(N_total_m, 1), 6),
            "n_out_of_graph": n_out_of_graph_m,
            "out_of_graph_ratio": round(n_out_of_graph_m / max(r_steps_m, 1), 6),
        },
        "wall_time_s": round(max_wall, 1),
        "split": split,
        "shards_used": available_shards,
    }
    log(f"  test  {result_line(result_m)}  out of graph {result_m['coverage']['out_of_graph_ratio']:.2%}"
        f"  {round(max_wall, 1)}s")
    return result_m
