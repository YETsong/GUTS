"""STEP 2 of guts/yjmob/pipeline.py: shard datasets, rolling graphs and shard training.

Writes one processed directory per shard (guts/yjmob/data/shard_data.py), adds the
rolling graphs (guts/yjmob/data/rolling_graphs.py), and trains every shard with
guts/yjmob/model/train_single.py, one subprocess per shard. With several GPUs the
shards wait in one queue, largest first (estimate_shard_weight), and each GPU takes the
next shard when it is free. Shards with a results.json are skipped; an interrupted shard
resumes from its last.pt when trainer.resumable_checkpoint allows it, else starts over.

Console: one line per shard trained on GPU 0 (all shards with one GPU), with the number
of finished shards over all GPUs; the shards' own logs stay in their run directories
(train_log.txt).
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import glob
import json
import signal
import subprocess
import time
from typing import List, Tuple

import torch
import torch.multiprocessing as mp

from guts.method.reporting import format_metrics
from guts.method.utils import append_log_synced, ensure_dir, relpath_for_log, repo_relpath, timestamp_line
from guts.yjmob.data.shard_data import estimate_shard_weight
from guts.yjmob.model.trainer import resumable_checkpoint


_CURRENT_SHARD_PROC: list = [None]


def _kill_process_group(proc) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


def _kill_current_shard_subprocess(signum, frame):
    _kill_process_group(_CURRENT_SHARD_PROC[0])
    sys.exit(1)


def _num_trained_shards(run_dir: str) -> int:
    return len(glob.glob(os.path.join(run_dir, "graphsage_runs", "shard_*", "results.json")))


def _report_shard(run_dir: str, shard_id: int, lane_pos: str, num_shards: int, run_log_path: str) -> None:
    """Console + run log line of one finished shard (called on the GPU 0 lane only)."""
    result = _collect_graphsage_results(run_dir, [shard_id])[str(shard_id)]
    progress = f"{lane_pos}, all {_num_trained_shards(run_dir)}/{num_shards}"
    line = timestamp_line(f"  {format_shard_line(shard_id, result, progress)}")
    print(line, flush=True)
    append_log_synced(run_log_path, line)


def shard_queue(tasks, num_workers: int):
    """Queue of (shard_id, processed_dir) tasks, largest shard first, closed by one None per worker."""
    queue = mp.get_context("spawn").Queue()
    for task in sorted(tasks, key=lambda t: estimate_shard_weight(t[1]), reverse=True):
        queue.put(task)
    for _ in range(num_workers):
        queue.put(None)
    return queue


def _multi_graphsage_worker(rank: int, task_queue, cfg, run_dir: str, device_strs, num_shards: int, run_log_path: str):
    """One process per GPU: train the next shard from the shared queue until it is empty."""
    signal.signal(signal.SIGTERM, _kill_current_shard_subprocess)
    device_str = device_strs[rank]
    done = 0
    while True:
        task = task_queue.get()
        if task is None:
            break
        shard_id, processed_dir = task
        _run_graphsage_shard_train(cfg, shard_id, processed_dir, run_dir, device_str)
        done += 1
        if rank == 0:
            _report_shard(run_dir, shard_id, f"GPU0 #{done}", num_shards, run_log_path)


def _cuda_visible_devices_for(device_str: str) -> str | None:
    if not device_str.startswith("cuda"):
        return None
    if ":" not in device_str:
        return None
    return device_str.split(":", 1)[1]


def _run_graphsage_shard_train(
    cfg,
    shard_id: int,
    processed_dir: str,
    run_dir: str,
    device_str: str,
):
    """Run the GraphSAGE next-grid trainer (guts/yjmob/model/train_single.py) for one shard."""
    train_py = os.path.join(_REPO_ROOT, "guts", "yjmob", "model", "train_single.py")
    shard_run_dir = os.path.join(run_dir, "graphsage_runs", f"shard_{shard_id}")
    ensure_dir(shard_run_dir)
    cmd = [sys.executable, train_py, *trainer_cli_args(cfg, processed_dir, shard_run_dir)]
    run_trainer_subprocess(cmd, _cuda_visible_devices_for(device_str))


def trainer_cli_args(cfg, processed_dir: str, model_run_dir: str) -> List[str]:
    """Trainer arguments (guts/yjmob/model/trainer.py) of one model from cfg.training; an
    interrupted model resumes from its last.pt (exact continuation), else starts over."""
    t = cfg.training
    resume_ckpt = resumable_checkpoint(model_run_dir)
    target_epochs = int(t.epochs)
    if resume_ckpt:
        target_epochs = int(t.epochs) - int(torch.load(resume_ckpt, map_location="cpu")["epoch"])

    cmd = [
        "--processed_dir", processed_dir,
        "--fixed_run_dir", model_run_dir,
        "--epochs", str(target_epochs),
        "--batch_size", str(int(t.batch_size)),
        "--lr", str(float(t.lr)),
        "--weight_decay", str(float(t.weight_decay)),
        "--id_embedding_dim", str(int(t.id_embedding_dim)),
        "--sage_hidden_dim", str(int(t.sage_hidden_dim)),
        "--sage_out_dim", str(int(t.sage_out_dim)),
        "--sage_layers", str(int(t.sage_layers)),
        "--sage_dropout", str(float(t.sage_dropout)),
        "--gru_hidden_dim", str(int(t.gru_hidden_dim)),
        "--gru_layers", str(int(t.gru_layers)),
        "--gru_dropout", str(float(t.gru_dropout)),
        "--seed", str(int(t.seed)),
        "--sage_num_neighbors", *[str(n) for n in t.sage_num_neighbors],
        "--early_stop_warmup", str(int(t.early_stop_warmup)),
        "--early_stop_patience", str(int(t.early_stop_patience)),
    ]
    cmd += ["--rolling_window_size", str(int(cfg.temporal_graph.window_size))]
    if resume_ckpt:
        cmd += ["--resume", resume_ckpt]
    return cmd


def run_trainer_subprocess(cmd: List[str], visible: str | None) -> None:
    """Run a trainer command with CUDA_VISIBLE_DEVICES=visible; its output is printed only
    when it fails. The distributed variables of the parent are removed (torchrun sets its
    own). If the caller is interrupted, the trainer's process group is stopped too."""
    env = os.environ.copy()
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    env.pop("WORLD_SIZE", None)
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.pop("MASTER_ADDR", None)
    env.pop("MASTER_PORT", None)
    warning_filters = [
        item for item in env.get("PYTHONWARNINGS", "").split(",") if item
    ]
    warning_filters.append("ignore:pkg_resources is deprecated as an API:UserWarning")
    env["PYTHONWARNINGS"] = ",".join(warning_filters)
    proc = subprocess.Popen(
        cmd,
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _CURRENT_SHARD_PROC[0] = proc
    try:
        stdout, _ = proc.communicate()
    except BaseException:
        _kill_process_group(proc)
        raise
    finally:
        _CURRENT_SHARD_PROC[0] = None
    if proc.returncode != 0:
        if stdout:
            print(stdout, end="", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _collect_graphsage_results(run_dir: str, shard_ids: List[int]) -> dict:
    """Returns flat dict: {str(shard_id): {...}}"""
    out = {}
    for shard_id in sorted(shard_ids):
        shard_key = str(shard_id)
        result_path = os.path.join(
            run_dir, "graphsage_runs", f"shard_{shard_id}", "results.json"
        )
        if not os.path.exists(result_path):
            out[shard_key] = {"status": "missing_results", "shard_id": shard_id}
        else:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out[shard_key] = {
                "status": "ok",
                "shard_id": shard_id,
                "results_path": os.path.relpath(result_path, run_dir),
                "best_epoch": data.get("best_epoch"),
                "stopped_epoch": data.get("stopped_epoch"),
                "best_val_mrr": data.get("best_val_mrr"),
                "best_train_metrics": data.get("best_train_metrics", {}),
                "best_val_metrics": data.get("best_val_metrics", {}),
                "timing_ms": data.get("timing_ms", {}),
            }
    return out


# Continuation lines of a shard report start under the shard number of its first line,
# which follows "[HH:MM:SS]   " (timestamp_line + the two-space indent of the callers).
_SHARD_LINE_PAD = " " * 15


def format_shard_line(shard_id: int, result: dict, progress: str = "") -> str:
    """Three-line report of a trained shard from its results.json (or its entry in
    _collect_graphsage_results); shared with the unlearning retrain:
        shard  70  E21/36  492s  <progress>
                     train  MRR ..  ACC@1 ..  ACC@5 ..  ACC@10 ..   (at the best epoch)
                     val    MRR ..  ACC@1 ..  ACC@5 ..  ACC@10 ..
    """
    if result.get("status", "ok") != "ok":
        return f"shard {shard_id} {result.get('status', 'unknown')}"
    train_s = (result.get("timing_ms") or {}).get("train")
    head = (f"shard {shard_id:>3}  E{int(result['best_epoch']):02d}/{int(result['stopped_epoch']):02d}"
            + (f"  {train_s / 1000:>4.0f}s" if train_s is not None else "")
            + (f"  {progress}" if progress else ""))
    return "\n".join([
        head,
        f"{_SHARD_LINE_PAD}train  {format_metrics(result.get('best_train_metrics'), sep='  ')}",
        f"{_SHARD_LINE_PAD}val    {format_metrics(result.get('best_val_metrics'), sep='  ')}",
    ])


def run_train_stage(
    cfg,
    args,
    run_dir: str,
    train_days,
    by_day: dict,
    devices,
    log,
    run_log_path: str,
):
    """log prints to the console and the run log; run_log_path is that run log (the GPU 0
    training worker appends its shard lines to it)."""
    from guts.yjmob.data.shard_data import materialize_graphsage_shards  # noqa: PLC0415

    source_dir    = args.graphsage_source_processed_dir or cfg.data.nextgrid_processed_dir
    external_sharding_dir = args.sharded_dir

    sharding_dir   = os.path.abspath(external_sharding_dir) if external_sharding_dir \
                     else os.path.join(run_dir, "Sharding")
    processed_base = os.path.join(sharding_dir, "processed")
    gs_runs_dir    = os.path.join(run_dir, "graphsage_runs")

    all_tasks: List[Tuple[int, str]] = []

    if external_sharding_dir and os.path.isdir(processed_base):
        log(f"  shard data: reusing {relpath_for_log(processed_base)}")
        for name in sorted(os.listdir(processed_base)):
            if not name.startswith("shard_"):
                continue
            shard_id = int(name.split("_", 1)[1])
            processed_dir = os.path.join(processed_base, name)
            if os.path.exists(os.path.join(processed_dir, "next_location_meta.pt")):
                all_tasks.append((shard_id, processed_dir))
        preprocess_summary = {
            "mode": "reuse",
            "processed_base": processed_base,
            "total_shards": len(all_tasks),
        }
    else:
        ensure_dir(processed_base)

        expected_num_shards = int(cfg.partition.num_shards)
        existing_meta = [
            f for f in glob.glob(os.path.join(processed_base, "shard_*", "next_location_meta.pt"))
        ]
        if len(existing_meta) >= expected_num_shards:
            log(f"  shard data: reusing {len(existing_meta)}/{expected_num_shards} existing shards")
        else:
            if existing_meta:
                log(f"  [WARN] shard data incomplete ({len(existing_meta)}/{expected_num_shards} shards) — building again")
            t0_data = time.time()
            log(f"  shard data: building {expected_num_shards} shards")
            materialize_graphsage_shards(
                source_processed_dir=source_dir,
                out_dir=processed_base,
                days=list(train_days),
                by_day=by_day,
                splits=["train", "val", "test"],
                log_fn=log,
            )
            log(f"  shard data done  {time.time() - t0_data:.0f}s")

        for name in sorted(os.listdir(processed_base)):
            if not name.startswith("shard_"):
                continue
            shard_id = int(name.split("_", 1)[1])
            processed_dir = os.path.join(processed_base, name)
            if os.path.exists(os.path.join(processed_dir, "next_location_meta.pt")):
                all_tasks.append((shard_id, processed_dir))
        preprocess_summary = {
            "mode": "build",
            "source_processed_dir": repo_relpath(source_dir),
            "days": list(train_days),
            "total_shards": len(all_tasks),
        }

    if not all_tasks:
        raise RuntimeError(f"No GraphSAGE shard processed dirs found under {relpath_for_log(processed_base)}")

    temporal_graph = cfg.temporal_graph
    from guts.yjmob.data.rolling_graphs import (  # noqa: PLC0415
        update_shard_meta_with_rolling_graphs,
        write_shard_rolling_graphs,
    )

    window_size = int(temporal_graph.window_size)
    target_days = [d for d in sorted(train_days) if d >= window_size]
    t0_rolling = time.time()
    for shard_id, processed_dir in all_tasks:
        out_dir = os.path.join(processed_dir, f"rolling_graphs_w{window_size}")
        missing = [
            d for d in target_days
            if not os.path.exists(os.path.join(out_dir, f"target_day_{d:04d}.pt"))
        ]
        if not missing:
            continue
        meta = torch.load(os.path.join(processed_dir, "next_location_meta.pt"), map_location="cpu")
        num_nodes = int(meta["num_nodes"])
        entries = write_shard_rolling_graphs(processed_dir, target_days, window_size, num_nodes)
        update_shard_meta_with_rolling_graphs(processed_dir, window_size, entries)
    log(f"  rolling graphs (window {window_size}, target days {target_days[0]}-{target_days[-1]}) done"
        f"  {time.time() - t0_rolling:.0f}s")

    all_shard_ids = [sid for sid, _ in all_tasks]
    pending_tasks = []
    for sid, pdir in all_tasks:
        result_path = os.path.join(gs_runs_dir, f"shard_{sid}", "results.json")
        if not os.path.exists(result_path):
            pending_tasks.append((sid, pdir))
    skipped_count = len(all_tasks) - len(pending_tasks)
    all_tasks = pending_tasks

    if not all_tasks:
        log(f"  training: all {len(all_shard_ids)} shards already trained")
        shard_results = _collect_graphsage_results(run_dir, all_shard_ids)
        return preprocess_summary, shard_results

    num_shards = len(all_shard_ids)
    log(f"  training {len(all_tasks)} shards on {len(devices)} GPU(s)"
        + (f"  ({skipped_count} already trained)" if skipped_count else ""))
    t0 = time.time()

    if len(devices) == 1:
        for i, (shard_id, processed_dir) in enumerate(all_tasks, start=1):
            _run_graphsage_shard_train(cfg, shard_id, processed_dir, run_dir, str(devices[0]))
            _report_shard(run_dir, shard_id, f"{i}/{len(all_tasks)}", num_shards, run_log_path)
    else:
        device_strs = [str(d) for d in devices]
        log("  shards go to the next free GPU, largest first; lines below are the shards trained on GPU 0")
        mp.spawn(
            _multi_graphsage_worker,
            args=(shard_queue(all_tasks, len(devices)), cfg, run_dir, device_strs, num_shards, run_log_path),
            nprocs=len(devices),
            join=True,
        )

    shard_results = _collect_graphsage_results(run_dir, all_shard_ids)
    log(f"  training done  {_num_trained_shards(run_dir)}/{num_shards} shards  {time.time() - t0:.0f}s")
    return preprocess_summary, shard_results
