"""Shared training loop for GraphSAGE + GRU next-grid prediction.

Entry points:
  guts/yjmob/model/train_single.py     single process (one GPU or CPU); used by guts/yjmob/pipeline.py
                                       and the unlearning retrain, one subprocess per shard
  guts/yjmob/model/train_multi_gpu.py  multi-GPU data parallel under torchrun

Both call ``run_training`` with a process context: ``SingleProcess`` here, or
``DDPContext`` in train_multi_gpu.py, which overrides the cross-process hooks
(gradient averaging, metric sums, broadcast, barrier). Everything else is the
same code path.

Each session of target day d is encoded with the rolling graph of d (the
previous --rolling_window_size days merged), so sessions with
d < --rolling_window_size are skipped. The best validation-MRR checkpoint is
saved as checkpoints/best.pt and evaluated on the test split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from guts.utils import append_log as _append_log, ensure_dir, set_seed, timestamp_id
from guts.yjmob.model.dataset import (
    DayBatchSampler,
    NextGridSessionDataset,
    build_day_index,
    collate_next_grid_sessions,
)
from guts.yjmob.model.metrics import finalize_metrics, reciprocal_rank_sum, topk_correct
from guts.yjmob.model.model import NextGridPredictor

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_LOG_LABEL: int = 0  # "[R<label>]" in log lines; only label 0 also prints to stdout


class SingleProcess:
    """Process context of a single-process run; the cross-process hooks are no-ops."""

    rank = 0
    world_size = 1

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @property
    def is_log_rank(self) -> bool:
        return self.rank == 0

    def sync_gradients(self, model: nn.Module, n_tokens: int) -> None:
        """Combine this step's gradients over processes (n_tokens = tokens behind them)."""

    def sum(self, values: List[float]) -> List[float]:
        return list(values)

    def broadcast(self, obj):
        return obj

    def barrier(self) -> None:
        pass


def append_log(path: str, message: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    _append_log(path, line)
    if _LOG_LABEL == 0:
        print(line, flush=True)


def _make_unique_run_dir(out_root: str, world_size: int, run_name: str) -> str:
    ts = timestamp_id()
    suffix = run_name.strip()
    base = f"{ts}_base_graphsage_w{world_size}" + (f"_{suffix}" if suffix else "")
    runs_root = ensure_dir(os.path.join(out_root, "runs", "yjmob"))
    cand = os.path.join(runs_root, base)
    if not os.path.exists(cand):
        return cand
    for i in range(1, 1000):
        cand_i = os.path.join(runs_root, f"{base}_{i:03d}")
        if not os.path.exists(cand_i):
            return cand_i
    raise RuntimeError("Could not create a unique run directory.")


def sample_graph_neighbors(
    edge_index: torch.Tensor,
    edge_weight,
    k: int,
) -> tuple:
    """Randomly keep at most k incoming edges per destination node."""
    if k <= 0:
        return edge_index, edge_weight
    src, dst = edge_index[0], edge_index[1]
    E = src.size(0)
    if E == 0:
        return edge_index, edge_weight

    # Randomly shuffle then stable-sort by dst → random order within each group
    perm = torch.randperm(E, device=src.device)
    _, order = torch.sort(dst[perm], stable=True)
    final = perm[order]
    dst_s = dst[final]

    # Compute rank within each dst group using cumsum trick
    same = torch.zeros(E, dtype=torch.long, device=dst.device)
    if E > 1:
        same[1:] = (dst_s[1:] == dst_s[:-1]).long()
    cumsum = torch.cumsum(same, dim=0)
    group_base = torch.where(same == 0, cumsum, torch.zeros_like(cumsum))
    rank = cumsum - torch.cummax(group_base, dim=0).values

    keep = final[rank < k]
    new_ew = edge_weight[keep] if edge_weight is not None else None
    return edge_index[:, keep], new_ew


def batch_hidden(
    model: NextGridPredictor,
    dataset: NextGridSessionDataset,
    batch: Dict,
    device: torch.device,
    window_size: int,
    num_neighbors: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Fused hidden states and targets of the valid steps of a same-day batch."""
    day = int(batch["day"][0].item())
    if bool((batch["day"] != day).any()):
        raise ValueError("a batch mixes days; use DayBatchSampler")
    graph = dataset.get_rolling_graph(day, window_size)
    x = graph["x"].to(device, non_blocking=True)
    edge_index = graph["edge_index"].to(device, dtype=torch.long, non_blocking=True)
    edge_weight = graph["edge_weight"]
    if edge_weight is not None:
        edge_weight = edge_weight.to(device, non_blocking=True)
    node_lookup = graph["node_lookup"].to(device, non_blocking=True)
    if num_neighbors > 0:
        edge_index, edge_weight = sample_graph_neighbors(edge_index, edge_weight, num_neighbors)

    input_cells = batch["input_cells"].to(device, non_blocking=True)
    hidden = model.sequence_features(
        input_cells=input_cells,
        lengths=batch["lengths"].to(device, non_blocking=True),
        graph_x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_lookup=node_lookup,
    )
    mask = batch["mask"].to(device, non_blocking=True)
    targets = batch["target_cells"].to(device, non_blocking=True)
    stats = {"graph_nodes": int(x.size(0)), "graph_edges": int(edge_index.size(1))}
    return hidden[mask], targets[mask], stats


def train_one_epoch(
    model: NextGridPredictor,
    dataset,
    loader,
    optimizer,
    ctx: SingleProcess,
    epoch: int,
    log_path: str,
    log_interval: int,
    window_size: int,
    num_neighbors: int = 0,
) -> Tuple[float, int, Dict[int, int], float]:
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss_sum = 0.0
    sample_count = 0
    correct = {1: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    total_steps = len(loader)
    t_start = time.time()
    logged_graph_info = False

    for step_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        n_used = 0  # tokens whose loss was backpropagated in this step
        if batch["lengths"].numel() == 0:
            # DDP filler batch (this rank has no sessions left for the day): no forward,
            # only the gradient all-reduce below.
            ctx.sync_gradients(model, 0)
            optimizer.step()
            continue
        hidden, target, stats = batch_hidden(
            model, dataset, batch, ctx.device, window_size, num_neighbors=num_neighbors,
        )
        logits = model.logits(hidden)
        n = int(target.numel())
        if n > 0:
            loss = criterion(logits, target)
            if torch.isfinite(loss):
                loss.backward()
                n_used = n
                loss_sum += float(loss.item()) * n
                sample_count += n
                batch_correct = topk_correct(logits.detach(), target.detach(), ks=(1, 5, 10))
                for k, v in batch_correct.items():
                    correct[k] += int(v)
                rr_sum += reciprocal_rank_sum(logits.detach(), target.detach())
            else:
                # A NaN/inf loss (possible on shards whose graph shrank sharply
                # after node/edge unlearning) would permanently corrupt later
                # weights if backpropagated — skip this step instead.
                append_log(log_path,
                    f"[R{_LOG_LABEL}] [WARN] non-finite loss at epoch {epoch} step {step_idx} "
                    f"day {int(batch['day'][0])} (n={n}) — skipping this step")
            del loss
        del logits, hidden, target
        ctx.sync_gradients(model, n_used)
        optimizer.step()
        if not logged_graph_info:
            append_log(log_path,
                f"[R{_LOG_LABEL}] GRAPH | nodes {stats['graph_nodes']} | edges {stats['graph_edges']}")
            logged_graph_info = True
        if log_interval > 0 and (step_idx == 1 or step_idx % log_interval == 0):
            elapsed = time.time() - t_start
            sec_per_step = elapsed / step_idx
            avg_loss = loss_sum / max(sample_count, 1)
            append_log(
                log_path,
                f"[R{_LOG_LABEL}] E {epoch:03d} step {step_idx}/{total_steps} ({100 * step_idx // total_steps}%) "
                f"| loss {avg_loss:.4f} | tokens {sample_count} "
                f"| {sec_per_step * 1000:.0f}ms/step | elapsed {elapsed:.0f}s",
            )
    return loss_sum, sample_count, correct, rr_sum


def _reduce_metrics(ctx: SingleProcess, loss_sum: float, count: int, correct: Dict[int, int], rr_sum: float) -> Dict[str, float]:
    """Sum the per-process totals over all processes and turn them into metrics."""
    loss_sum, count, c1, c5, c10, rr_sum = ctx.sum([loss_sum, count, correct[1], correct[5], correct[10], rr_sum])
    return finalize_metrics(int(count), loss_sum, {1: int(c1), 5: int(c5), 10: int(c10)}, rr_sum)


@torch.no_grad()
def evaluate(model: NextGridPredictor, dataset, loader, ctx: SingleProcess, window_size: int) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    sample_count = 0
    correct = {1: 0, 5: 0, 10: 0}
    rr_sum = 0.0
    for batch in loader:
        hidden, target, _ = batch_hidden(model, dataset, batch, ctx.device, window_size)
        logits = model.logits(hidden)
        loss_sum += float(criterion(logits, target).item())
        sample_count += int(target.numel())
        batch_correct = topk_correct(logits, target, ks=(1, 5, 10))
        for k, v in batch_correct.items():
            correct[k] += int(v)
        rr_sum += reciprocal_rank_sum(logits, target)
    return _reduce_metrics(ctx, loss_sum, sample_count, correct, rr_sum)


def save_checkpoint(path, epoch, model, optimizer, extra: dict):
    torch.save({
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra,
    }, path)


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--processed_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--id_embedding_dim", type=int, default=128)
    p.add_argument("--sage_hidden_dim", type=int, default=128)
    p.add_argument("--sage_out_dim", type=int, default=128)
    p.add_argument("--sage_layers", type=int, default=2)
    p.add_argument("--sage_dropout", type=float, default=0.1)
    p.add_argument("--gru_hidden_dim", type=int, default=128)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--gru_dropout", type=float, default=0.0)
    p.add_argument("--rolling_window_size", type=int, default=3,
                   help="number of previous days merged into the graph of each target day")
    p.add_argument("--sage_weighted", type=int, default=0, choices=[0, 1],
                   help="1 = edge-weighted mean aggregation (default 0: uniform mean)")
    p.add_argument("--sage_edge_weight_transform", type=str, default="log1p", choices=["raw", "log1p"],
                   help="edge weight transform when --sage_weighted 1")
    p.add_argument("--sage_num_neighbors", type=int, nargs="+", default=[10, 10],
                   help="max incoming edges kept per node during training (the smallest value is used); 0 = full graph")
    p.add_argument("--sage_edge_chunk_size", type=int, default=500_000,
                   help="edge chunk size of the aggregation (lower peak GPU memory); 0 = no chunking")
    p.add_argument("--seed", type=int, default=2026, help="-1 = random seed")
    p.add_argument("--resume", type=str, default="", help="checkpoint to resume from; --epochs then counts additional epochs")
    p.add_argument("--out_root", type=str, default=os.path.join(_REPO_ROOT, "Results"))
    p.add_argument("--fixed_run_dir", type=str, default="")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--part_cache_size", type=int, default=128)
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=1, help="validate every N epochs (0 = never)")
    p.add_argument("--early_stop_warmup", type=int, default=20,
                   help="epochs before early stopping can trigger (0 = none)")
    p.add_argument("--early_stop_patience", type=int, default=15,
                   help="stop after this many evals without val MRR improvement (0 = off)")
    return p


def run_training(args: argparse.Namespace, ctx: SingleProcess, log_label: int) -> None:
    """Train, keep the best validation-MRR checkpoint, and evaluate it on the test split."""
    global _LOG_LABEL
    _LOG_LABEL = int(log_label)
    device, rank, world_size, is_log_rank = ctx.device, ctx.rank, ctx.world_size, ctx.is_log_rank

    if args.seed < 0:
        args.seed = int(ctx.broadcast(int.from_bytes(os.urandom(4), byteorder="big", signed=False) if is_log_rank else 0))
    set_seed(args.seed)

    run_dir = ""
    if is_log_rank:
        run_dir = args.fixed_run_dir or _make_unique_run_dir(args.out_root, world_size, args.run_name)
    run_dir = ensure_dir(ctx.broadcast(run_dir))

    log_dir = ensure_dir(os.path.join(run_dir, "logs"))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))
    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
    train_log_path = os.path.join(log_dir, f"train_rank{rank}.txt")
    val_log_path = os.path.join(log_dir, "val_log.txt")
    test_log_path = os.path.join(log_dir, "test_log.txt")
    window_size = int(args.rolling_window_size)

    append_log(train_log_path, "START # GRAPHSAGE NEXT-GRID TRAIN")
    if is_log_rank:
        append_log(train_log_path, f"[INIT] run_dir={run_dir}")
        append_log(train_log_path, f"[INIT] device={device} world_size={world_size} seed={args.seed}")
        append_log(train_log_path, f"[INIT] sage_num_neighbors={args.sage_num_neighbors} sage_edge_chunk_size={args.sage_edge_chunk_size}")

        append_log(test_log_path, "==============================================")
        append_log(test_log_path, datetime.now().strftime("%Y/%m/%d - %H:%M:%S"))
        append_log(test_log_path, "==============================================")

    train_ds = NextGridSessionDataset(args.processed_dir, split="train", part_cache_size=args.part_cache_size)
    val_ds = NextGridSessionDataset(args.processed_dir, split="val", part_cache_size=args.part_cache_size)
    test_ds = NextGridSessionDataset(args.processed_dir, split="test", part_cache_size=args.part_cache_size)
    if not train_ds.has_rolling_graphs:
        raise ValueError(
            f"{args.processed_dir} has no rolling graphs "
            "(guts/yjmob/pipeline.py builds them before training)."
        )

    model = NextGridPredictor(
        num_nodes=train_ds.num_nodes,
        node_feat_dim=train_ds.node_feat_dim,
        id_embedding_dim=args.id_embedding_dim,
        sage_hidden_dim=args.sage_hidden_dim,
        sage_out_dim=args.sage_out_dim,
        sage_layers=args.sage_layers,
        sage_dropout=args.sage_dropout,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        sage_weighted=bool(args.sage_weighted),
        sage_edge_weight_transform=args.sage_edge_weight_transform,
        sage_edge_chunk_size=int(args.sage_edge_chunk_size),
    ).to(device)
    append_log(
        train_log_path,
        f"[R{_LOG_LABEL}] SAGE GRAPH | weighted {int(args.sage_weighted)} "
        f"| edge_weight_transform {args.sage_edge_weight_transform}",
    )

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_mrr = -1.0
    best_epoch = -1
    best_ckpt_path = ""
    epochs_since_improve = 0
    train_ms_acc = 0.0
    val_eval_ms_acc = 0.0
    finished = False
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        # --epochs N with --resume means N additional epochs, not absolute epoch number
        args.epochs = (start_epoch - 1) + args.epochs
        best_val_mrr = float(ckpt["extra"]["best_val_mrr"])
        best_epoch = int(ckpt["extra"]["best_epoch"])
        best_ckpt_path = str(ckpt["extra"]["best_ckpt_path"])
        if "rng" in ckpt["extra"]:
            # last.pt: continue exactly where the interrupted run stopped.
            epochs_since_improve = int(ckpt["extra"]["epochs_since_improve"])
            train_ms_acc = float(ckpt["extra"]["train_ms"])
            val_eval_ms_acc = float(ckpt["extra"]["val_eval_ms"])
            _set_rng_state(ckpt["extra"]["rng"])
            finished = bool(ckpt["extra"]["finished"])
        append_log(train_log_path, f"[RESUME] {args.resume} -> epoch {start_epoch} (train until epoch {args.epochs})")
    if is_log_rank:
        # Written after resolving --resume so "epochs" is the run's total: the unlearning
        # retrain trains a shard again from scratch with these arguments.
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    def _day_index(split: str, meta: dict) -> Dict[int, List[int]]:
        day_index = build_day_index(args.processed_dir, split, meta)
        return {d: idxs for d, idxs in day_index.items() if d >= window_size}

    append_log(
        train_log_path,
        f"[R{_LOG_LABEL}] ROLLING GRAPH | window_size={window_size} "
        f"| sessions with day < {window_size} are excluded from train/eval",
    )
    train_sampler = DayBatchSampler(_day_index("train", train_ds.meta), args.batch_size,
                                    rank=rank, world_size=world_size, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_next_grid_sessions,
        pin_memory=torch.cuda.is_available(),
    )
    eval_batch_size = min(args.batch_size, 128)

    def _eval_loader(ds, split):
        # No filler batches: evaluation combines per-rank totals only once at the end.
        sampler = DayBatchSampler(_day_index(split, ds.meta), eval_batch_size,
                                  rank=rank, world_size=world_size, shuffle=False, seed=0, equal_steps=False)
        return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collate_next_grid_sessions)

    val_loader = _eval_loader(val_ds, "val")
    test_loader = _eval_loader(test_ds, "test")

    append_log(
        train_log_path,
        f"[R{_LOG_LABEL}] SETUP | train_sessions {len(train_ds)} | val_sessions {len(val_ds)} "
        f"| test_sessions {len(test_ds)} | batch {args.batch_size} | part_cache_size {args.part_cache_size}",
    )
    append_log(
        train_log_path,
        f"[R{_LOG_LABEL}] TOKENS | train {train_ds.total_steps} | val {val_ds.total_steps} | test {test_ds.total_steps}",
    )
    append_log(train_log_path, f"[R{_LOG_LABEL}] EVAL | eval_every {args.eval_every}")
    append_log(train_log_path, f"[R{_LOG_LABEL}] TRAIN LOSS full softmax | classes {train_ds.num_nodes}")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, (start_epoch if finished else args.epochs + 1)):
        epoch_t0 = time.time()
        train_sampler.set_epoch(epoch)
        append_log(train_log_path, f"[R{_LOG_LABEL}] ===== Epoch {epoch:03d} ({datetime.now().strftime('%H:%M:%S')}) =====")
        loss_sum, sample_count, correct, rr_sum = train_one_epoch(
            model,
            train_ds,
            train_loader,
            optimizer,
            ctx,
            epoch,
            train_log_path,
            args.log_interval,
            window_size,
            num_neighbors=min(args.sage_num_neighbors) if args.sage_num_neighbors else 0,
        )
        train_ms_acc += (time.time() - epoch_t0) * 1000

        train_metrics = _reduce_metrics(ctx, loss_sum, sample_count, correct, rr_sum)
        _mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
        _mem_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
        append_log(
            train_log_path,
            (
                f"[R{_LOG_LABEL}] E {epoch:03d} = Loss {train_metrics['loss']:.4f} | "
                f"ACC@1 {train_metrics['acc@1']:.4f} | ACC@5 {train_metrics['acc@5']:.4f} | "
                f"ACC@10 {train_metrics['acc@10']:.4f} | MRR {train_metrics['mrr']:.4f} | "
                f"TimeSec {time.time() - epoch_t0:.2f} | "
                f"VRAM alloc={_mem_alloc_mb:.0f}MB reserved={_mem_reserved_mb:.0f}MB"
            ),
        )

        should_eval = args.eval_every > 0 and epoch % args.eval_every == 0
        if should_eval:
            t0_val_eval = time.time()
            torch.cuda.empty_cache()
            val_metrics = evaluate(model, val_ds, val_loader, ctx, window_size)
            val_eval_ms_acc += (time.time() - t0_val_eval) * 1000
            if is_log_rank:
                append_log(
                    val_log_path,
                    (
                        f"[VAL] E {epoch:03d} | loss {val_metrics['loss']:.4f} | "
                        f"ACC@1 {val_metrics['acc@1']:.4f} | ACC@5 {val_metrics['acc@5']:.4f} | "
                        f"ACC@10 {val_metrics['acc@10']:.4f} | MRR {val_metrics['mrr']:.4f} | "
                        f"support {val_metrics['support']}"
                    ),
                )
            cur_val_mrr = val_metrics["mrr"] if not np.isnan(val_metrics["mrr"]) else -1.0
            if cur_val_mrr > best_val_mrr:
                best_val_mrr = float(cur_val_mrr)
                best_epoch = int(epoch)
                epochs_since_improve = 0
                best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
                if is_log_rank:
                    save_checkpoint(
                        best_ckpt_path,
                        epoch,
                        model,
                        optimizer,
                        extra={
                            "best_val_mrr": float(best_val_mrr),
                            "best_epoch": int(best_epoch),
                            "best_ckpt_path": str(best_ckpt_path),
                        },
                    )
                    append_log(train_log_path, f"[CKPT] best.pt updated | best_val_mrr {best_val_mrr:.6f} | epoch {best_epoch:03d}")
            else:
                epochs_since_improve += 1

        early_stop = (
            should_eval
            and args.early_stop_patience > 0
            and epoch >= args.early_stop_warmup
            and epochs_since_improve >= args.early_stop_patience
        )
        if is_log_rank:
            # Exact resume point for an interrupted run (removed once results.json is written).
            save_checkpoint(last_ckpt_path, epoch, model, optimizer, extra={
                "best_val_mrr": float(best_val_mrr),
                "best_epoch": int(best_epoch),
                "best_ckpt_path": str(best_ckpt_path),
                "epochs_since_improve": int(epochs_since_improve),
                "train_ms": float(train_ms_acc),
                "val_eval_ms": float(val_eval_ms_acc),
                "rng": _rng_state(),
                "finished": bool(early_stop or epoch == args.epochs),
            })
        # Other ranks may read best.pt only after rank 0 has written it.
        ctx.barrier()

        if early_stop:
            if is_log_rank:
                append_log(
                    train_log_path,
                    f"[EARLY STOP] val MRR no improvement for {epochs_since_improve} epochs "
                    f"(warmup={args.early_stop_warmup}, patience={args.early_stop_patience}) "
                    f"-> stopping at epoch {epoch:03d} (best_epoch={best_epoch:03d})",
                )
            break

    if best_ckpt_path and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if is_log_rank:
            append_log(test_log_path, f"[LOAD BEST] {best_ckpt_path}")
    torch.cuda.empty_cache()
    t0_predict = time.time()
    test_metrics = evaluate(model, test_ds, test_loader, ctx, window_size)
    predict_ms = (time.time() - t0_predict) * 1000
    if is_log_rank:
        append_log(
            test_log_path,
            (
                f"[TEST FINAL] loss {test_metrics['loss']:.4f} | ACC@1 {test_metrics['acc@1']:.4f} | "
                f"ACC@5 {test_metrics['acc@5']:.4f} | ACC@10 {test_metrics['acc@10']:.4f} | "
                f"MRR {test_metrics['mrr']:.4f} | support {test_metrics['support']}"
            ),
        )
        append_log(test_log_path, f"[BEST] best_val_mrr: {best_val_mrr:.6f} | best_epoch: {best_epoch:03d}")
        results = {
            "experiment_type": "graphsage_next_grid",
            "run_dir": run_dir,
            "processed_dir": os.path.abspath(args.processed_dir),
            "device": str(device),
            "world_size": int(world_size),
            "epochs": int(args.epochs),
            "stopped_epoch": int(epoch),
            "early_stopped": bool(epoch < args.epochs),
            "early_stop_warmup": int(args.early_stop_warmup),
            "early_stop_patience": int(args.early_stop_patience),
            "best_epoch": int(best_epoch),
            "best_val_mrr": round(float(best_val_mrr), 6),
            "timing_ms": {
                "train": round(train_ms_acc),
                "val_eval": round(val_eval_ms_acc),
                "predict": round(predict_ms),
            },
            "metrics": {k: round(float(v), 6) if isinstance(v, float) else v for k, v in test_metrics.items()},
            "model": {
                "sage_weighted": bool(args.sage_weighted),
                "sage_edge_weight_transform": args.sage_edge_weight_transform,
                "id_embedding_dim": int(args.id_embedding_dim),
                "sage_hidden_dim": int(args.sage_hidden_dim),
                "sage_out_dim": int(args.sage_out_dim),
                "sage_layers": int(args.sage_layers),
                "gru_hidden_dim": int(args.gru_hidden_dim),
                "gru_layers": int(args.gru_layers),
                "rolling_window_size": window_size,
                "eval_every": int(args.eval_every),
            },
        }
        results_path = os.path.join(run_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        append_log(test_log_path, f"[RESULTS] saved -> {results_path}")
        if os.path.exists(last_ckpt_path):
            os.remove(last_ckpt_path)
