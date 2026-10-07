"""STEP 2 of guts/yjmob/pipeline.py: shard datasets, rolling graphs and shard training.

Writes one processed directory per shard (guts/yjmob/data/shard_data.py), adds the
rolling graphs (guts/yjmob/data/rolling_graphs.py), and trains every shard with
guts/yjmob/model/train_single.py, one subprocess per GPU lane. Shards with a
results.json are skipped; an interrupted shard resumes from last.pt (else best.pt).
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
from datetime import datetime
from typing import List, Tuple

import torch
import torch.multiprocessing as mp

from guts.utils import append_log, balanced_lanes, ensure_dir, estimate_shard_weight, relpath_for_log


_CURRENT_SHARD_PROC: list = [None]


def _kill_current_shard_subprocess(signum, frame):
    proc = _CURRENT_SHARD_PROC[0]
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    sys.exit(1)


def _multi_graphsage_worker(rank: int, task_groups, cfg, run_dir: str, device_strs, log_dir: str):
    signal.signal(signal.SIGTERM, _kill_current_shard_subprocess)
    device_str = device_strs[rank]
    for (shard_id, processed_dir) in task_groups[rank]:
        _run_graphsage_shard_train(cfg, shard_id, processed_dir, run_dir, device_str, log_dir)


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
    log_dir: str,
):
    """Run the GraphSAGE next-grid trainer (guts/yjmob/model/train_single.py) for one shard."""
    t = cfg.training
    train_py = os.path.join(_REPO_ROOT, "guts", "yjmob", "model", "train_single.py")
    shard_run_dir = os.path.join(run_dir, "graphsage_runs", f"shard_{shard_id}")
    ensure_dir(shard_run_dir)
    ensure_dir(log_dir)

    # Resume an interrupted shard: from last.pt (saved every epoch, exact continuation),
    # else from best.pt (best epoch; later epochs and the early-stop count are lost).
    resume_ckpt = None
    target_epochs = int(t.epochs)
    ckpt_dir = os.path.join(shard_run_dir, "checkpoints")
    for name in ("last.pt", "best.pt"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            resume_ckpt = path
            target_epochs = int(t.epochs) - int(torch.load(path, map_location="cpu")["epoch"])
            break

    cmd = [
        sys.executable,
        train_py,
        "--processed_dir", processed_dir,
        "--fixed_run_dir", shard_run_dir,
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
    env = os.environ.copy()
    visible = _cuda_visible_devices_for(device_str)
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
    stdout, _ = proc.communicate()
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
                "best_val_mrr": data.get("best_val_mrr"),
                "metrics": data.get("metrics", {}),
                "timing_ms": data.get("timing_ms", {}),
            }
    return out


def _format_graphsage_shard_summary(shard_id: int, result: dict) -> str:
    if result.get("status") != "ok":
        return f"  GraphSAGE shard{shard_id} status={result.get('status', 'unknown')}"
    metrics = result.get("metrics", {}) or {}
    support = int(metrics.get("support", 0) or 0)
    if support == 0:
        return f"  GraphSAGE shard{shard_id} pass support=0"
    return (
        f"  GraphSAGE shard{shard_id} done"
        f"  best_epoch={result.get('best_epoch')}"
        f"  best_val_mrr={float(result.get('best_val_mrr', float('nan'))):.6f}"
        f"  test_mrr={float(metrics.get('mrr', float('nan'))):.6f}"
        f"  acc@1={float(metrics.get('acc@1', float('nan'))):.6f}"
        f"  acc@5={float(metrics.get('acc@5', float('nan'))):.6f}"
        f"  acc@10={float(metrics.get('acc@10', float('nan'))):.6f}"
        f"  support={support}"
    )


def _write_graphsage_eval_log(eval_log_path: str, run_dir: str, shard_results: dict):
    append_log(eval_log_path, "=" * 60)
    append_log(eval_log_path, datetime.now().strftime("%Y/%m/%d - %H:%M:%S"))
    append_log(eval_log_path, f"run_dir: {run_dir}")
    append_log(eval_log_path, "[GRAPHSAGE SHARD RESULTS]")
    ok_metrics = []
    for shard_key in sorted(shard_results, key=lambda x: int(x)):
        result = shard_results[shard_key]
        append_log(eval_log_path, _format_graphsage_shard_summary(int(shard_key), result).strip())
        if result.get("status") == "ok":
            ok_metrics.append(result.get("metrics", {}) or {})

    if ok_metrics:
        keys = ("acc@1", "acc@5", "acc@10", "mrr")
        avg = {
            k: sum(float(m.get(k, 0.0)) for m in ok_metrics) / max(len(ok_metrics), 1)
            for k in keys
        }
        total_support = sum(int(m.get("support", 0) or 0) for m in ok_metrics)
        append_log(
            eval_log_path,
            (
                "[GRAPHSAGE SHARD AVG]"
                f" acc@1={avg['acc@1']:.6f}"
                f" | acc@5={avg['acc@5']:.6f}"
                f" | acc@10={avg['acc@10']:.6f}"
                f" | mrr={avg['mrr']:.6f}"
                f" | total_support={total_support}"
            ),
        )
    append_log(eval_log_path, f"=== {datetime.now().strftime('%Y/%m/%d - %H:%M:%S')} ===")


def run_train_stage(
    cfg,
    args,
    run_dir: str,
    train_days,
    by_day: dict,
    devices,
    eval_log_path,
    log,
):
    from guts.yjmob.data.shard_data import materialize_graphsage_shards  # noqa: PLC0415

    source_dir    = args.graphsage_source_processed_dir or cfg.data.nextgrid_processed_dir
    external_sharding_dir = args.sharded_dir

    sharding_dir   = os.path.abspath(external_sharding_dir) if external_sharding_dir \
                     else os.path.join(run_dir, "Sharding")
    processed_base = os.path.join(sharding_dir, "processed")
    gs_runs_dir    = os.path.join(run_dir, "graphsage_runs")

    all_tasks: List[Tuple[int, str]] = []

    if external_sharding_dir and os.path.isdir(processed_base):
        log(f"  [skip] using existing sharded data: {processed_base} (materialize skipped)")
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
            log(f"  [skip] GraphSAGE sharded data already exists ({len(existing_meta)}/{expected_num_shards} shards) — skipping materialize")
        else:
            if existing_meta:
                log(f"  [WARN] GraphSAGE sharded data incomplete ({len(existing_meta)}/{expected_num_shards} shards found) — rebuilding materialize")
            log(f"  GraphSAGE sharded data: build -> {processed_base}/shard_{{k}}/")
            materialize_graphsage_shards(
                source_processed_dir=source_dir,
                out_dir=processed_base,
                days=list(train_days),
                by_day=by_day,
                splits=["train", "val", "test"],
                log_fn=log,
            )

        for name in sorted(os.listdir(processed_base)):
            if not name.startswith("shard_"):
                continue
            shard_id = int(name.split("_", 1)[1])
            processed_dir = os.path.join(processed_base, name)
            if os.path.exists(os.path.join(processed_dir, "next_location_meta.pt")):
                all_tasks.append((shard_id, processed_dir))
        preprocess_summary = {
            "mode": "build",
            "source_processed_dir": os.path.abspath(source_dir),
            "days": list(train_days),
            "total_shards": len(all_tasks),
        }

    if not all_tasks:
        raise RuntimeError(f"No GraphSAGE shard processed dirs found under {processed_base}")

    temporal_graph = cfg.temporal_graph
    from guts.yjmob.data.rolling_graphs import (  # noqa: PLC0415
        update_shard_meta_with_rolling_graphs,
        write_shard_rolling_graphs,
    )

    window_size = int(temporal_graph.window_size)
    target_days = [d for d in sorted(train_days) if d >= window_size]
    log(f"  [rolling] window_size={window_size}  target_days={target_days}  shards={len(all_tasks)}")
    for i, (shard_id, processed_dir) in enumerate(all_tasks):
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
        if (i + 1) % 20 == 0 or i == len(all_tasks) - 1:
            log(f"    rolling graphs {i + 1}/{len(all_tasks)} shard(s) done")

    all_shard_ids = [sid for sid, _ in all_tasks]
    pending_tasks = []
    for sid, pdir in all_tasks:
        result_path = os.path.join(gs_runs_dir, f"shard_{sid}", "results.json")
        if os.path.exists(result_path):
            log(f"  [skip] shard {sid}: already trained (results.json found)")
        else:
            pending_tasks.append((sid, pdir))
    skipped_count = len(all_tasks) - len(pending_tasks)
    all_tasks = pending_tasks

    if not all_tasks:
        log("  [skip] all shards already trained — train step skipped")
        shard_results = _collect_graphsage_results(run_dir, all_shard_ids)
        return preprocess_summary, shard_results

    log(f"  GraphSAGE train: {len(all_tasks)} shard tasks  epochs={cfg.training.epochs}"
        + (f"  (skipped {skipped_count} already done)" if skipped_count else ""))
    t0 = time.time()
    log_dir = ensure_dir(os.path.join(run_dir, "logs"))

    if len(devices) == 1:
        for (shard_id, processed_dir) in all_tasks:
            log(f"  -> shard {shard_id}  processed_dir={relpath_for_log(processed_dir)}")
            _run_graphsage_shard_train(
                cfg, shard_id, processed_dir, run_dir, str(devices[0]), log_dir,
            )
            shard_result = _collect_graphsage_results(run_dir, [shard_id]).get(str(shard_id), {})
            log(_format_graphsage_shard_summary(shard_id, shard_result))
    else:
        task_weights = {task: estimate_shard_weight(task[1]) for task in all_tasks}
        task_groups = balanced_lanes(list(task_weights.items()), len(devices))
        device_strs = [str(d) for d in devices]
        for i, group in enumerate(task_groups):
            load = sum(task_weights[t] for t in group)
            log(f"  GPU {i} ({device_strs[i]}): {[s for s, _ in group]}  (est. load={load})")
        mp.spawn(
            _multi_graphsage_worker,
            args=(task_groups, cfg, run_dir, device_strs, log_dir),
            nprocs=len(devices),
            join=True,
        )

    shard_ids = [shard_id for shard_id, _ in all_tasks]
    shard_results = _collect_graphsage_results(run_dir, shard_ids)
    if len(devices) > 1:
        for shard_key, result in sorted(shard_results.items(), key=lambda x: int(x[0])):
            log(_format_graphsage_shard_summary(int(shard_key), result))
    _write_graphsage_eval_log(eval_log_path, run_dir, shard_results)
    log(f"  GraphSAGE training done  elapsed={time.time() - t0:.1f}s")
    return preprocess_summary, shard_results

