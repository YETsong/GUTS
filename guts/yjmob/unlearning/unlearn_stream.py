from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from guts.utils import ensure_dir, append_log, timestamp_id, relpath_for_log  # noqa: E402
from guts.config import load_config  # noqa: E402
from guts.yjmob.unlearning.unlearn import (  # noqa: E402
    _copy_graphsage_runs_selective,
    _copy_partition_metadata,
    _detect_method,
    _get_or_build_index,
    _link_untouched_shards,
    _rebuild_target_shards_from_clean,
    _resolve_eval_data_dir,
    _resolve_sharding_dir,
    retrain_shards_parallel,
    run_eval,
)

STREAM_LOG = "stream_log.jsonl"
RESULT_FILENAME = "results_stream.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="yjmob GraphSAGE stream unlearning, one request at a time (uid/node/path)")
    p.add_argument("--state_dir", required=True, help="source shard-runs directory")
    p.add_argument("--sharded-dir", dest="sharded_dir", default="",
                   help="source Sharding/ path (default: state_dir/Sharding)")
    p.add_argument("--forget_type", default="uid", choices=["uid", "node", "path"],
                   help="request type: uid / node / path")
    p.add_argument("--forget_set_path", required=True,
                   help="forget set JSON; records are processed in order")
    p.add_argument("--clean_source_template", required=True,
                   help="per-step clean processed dir template with '{step}', "
                        "e.g. processed/yjmob_1_14d_clean/uid-stream_seed2026/k{step:03d}")
    p.add_argument("--index_path", default="",
                   help="unlearn index cache (default: <source sharding_dir>/graphsage_unlearn_index.pkl)")
    p.add_argument("--out_root", default="",
                   help="output root (default: shard-unlearns/<dataset>/ next to state_dir)")
    p.add_argument("--out_dir", default="",
                   help="output directory (pass an existing one to resume); created if empty")
    p.add_argument("--run_tag", default="", help="tag appended to the output directory name")
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=0, help="process only the first N requests (0 = all)")
    p.add_argument("--skip_eval", action="store_true", help="skip the final evaluation")
    p.add_argument("--eval_data_dir", default="", help="override the test processed directory")
    return p.parse_args()


def _read_done_steps(stream_log_path: str) -> Dict[int, dict]:
    done: Dict[int, dict] = {}
    if os.path.exists(stream_log_path):
        with open(stream_log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[int(rec["step"])] = rec
    return done


_NUM = r"([-+0-9.eE]+|nan|inf)"
_TRAIN_EPOCH_RE = re.compile(rf"E (\d+) = Loss {_NUM} \| ACC@1 {_NUM} \| ACC@5 {_NUM} \| ACC@10 {_NUM} \| MRR {_NUM}")
_VAL_EPOCH_RE = re.compile(rf"\[VAL\] E (\d+) \| loss {_NUM} \| ACC@1 {_NUM} \| ACC@5 {_NUM} \| ACC@10 {_NUM} \| MRR {_NUM}"
                           r"(?: \| support (\d+))?")
_EPOCH_KEYS = ("loss", "acc@1", "acc@5", "acc@10", "mrr")


def _epoch_metrics(log_path: str, pattern: re.Pattern, epoch: int) -> Optional[dict]:
    if epoch is None or not os.path.exists(log_path):
        return None
    with open(log_path, errors="replace") as f:
        hits = [m for m in (pattern.search(line) for line in f) if m]
    starts = [i for i, m in enumerate(hits) if int(m.group(1)) == 1]
    for m in hits[starts[-1] if starts else 0:]:
        if int(m.group(1)) == int(epoch):
            rec = {k: float(v) for k, v in zip(_EPOCH_KEYS, m.groups()[1:6])}
            if m.lastindex and m.lastindex >= 7 and m.group(7):
                rec["support"] = int(m.group(7))
            return rec
    return None


def _shard_step_record(shard_run_dir: str) -> Optional[dict]:
    path = os.path.join(shard_run_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        res = json.load(f)
    t = res.get("timing_ms") or {}
    best = res.get("best_epoch")
    logs = os.path.join(shard_run_dir, "logs")
    return {
        "train_val_s": round(((t.get("train") or 0) + (t.get("val_eval") or 0)) / 1000, 2),
        "predict_s": round((t.get("predict") or 0) / 1000, 2),
        "best_epoch": best,
        "stopped_epoch": res.get("stopped_epoch"),
        "best_val_mrr": res.get("best_val_mrr"),
        "train_at_best": _epoch_metrics(os.path.join(logs, "train_rank0.txt"), _TRAIN_EPOCH_RE, best),
        "val_at_best": _epoch_metrics(os.path.join(logs, "val_log.txt"), _VAL_EPOCH_RE, best),
        "test_local": res.get("metrics"),
    }


def _shard_timing_s(dst_graphsage: str, shard_ids: List[int]) -> Dict[int, dict]:
    out = {}
    for s in shard_ids:
        rec = _shard_step_record(os.path.join(dst_graphsage, f"shard_{s}"))
        if rec is not None:
            out[s] = rec
    return out


def _path_key(r: dict) -> tuple:
    return (int(r["uid"]), int(r["day"]), int(r.get("start_time", -1)), int(r.get("end_time", -1)))


def _plan_steps(forget_type: str, records: List[dict], idx) -> List[dict]:
    """Label and affected shards of each request, in order."""
    if forget_type == "uid":
        return [{"label": int(r["uid"]), "shards": set(idx.affected_shards_for_uids([int(r["uid"])]))} for r in records]
    if forget_type == "node":
        return [{"label": int(r["nid"]), "shards": set(idx.affected_shards_for_nodes([int(r["nid"])]))} for r in records]
    return [{"label": list(_path_key(r)), "shards": set(idx.affected_shards_for_path_keys({_path_key(r)}))} for r in records]


def _train_sessions(processed_dir: str) -> int:
    return int(torch.load(os.path.join(processed_dir, "next_location_meta.pt"), map_location="cpu")
               ["sequence_index"]["train"]["sessions"])


def main() -> None:
    args = parse_args()
    wall_start = time.time()

    available_gpus = torch.cuda.device_count()
    if available_gpus > 0 and args.num_gpus > available_gpus:
        args.num_gpus = available_gpus

    state_dir = os.path.abspath(args.state_dir)
    sharding_dir = _resolve_sharding_dir(state_dir, args.sharded_dir)
    method = _detect_method(state_dir)

    if args.out_dir:
        out_dir = ensure_dir(os.path.abspath(args.out_dir))
    else:
        if args.out_root:
            out_root = args.out_root
        else:
            dataset_dir_name = os.path.basename(os.path.dirname(state_dir))
            results_root = os.path.dirname(os.path.dirname(os.path.dirname(state_dir)))
            out_root = os.path.join(results_root, "shard-unlearns", dataset_dir_name)
        run_name = f"{timestamp_id()}_{method}_{args.forget_type}-stream_w{max(1, args.num_gpus)}"
        if args.run_tag:
            run_name += f"_{args.run_tag}"
        out_dir = ensure_dir(os.path.join(out_root, run_name))

    log_path = os.path.join(ensure_dir(os.path.join(out_dir, "logs")), "unlearn_stream_log.txt")

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        append_log(log_path, line)

    run_cfg_path = os.path.join(state_dir, "config_used.yaml")
    cfg = load_config(run_cfg_path)

    with open(args.forget_set_path) as f:
        records_all = json.load(f)["records"]
    records = records_all[: args.max_steps] if args.max_steps else records_all

    log("=" * 60)
    log(f"GraphSAGE UNLEARNING STREAM  type={args.forget_type}  steps={len(records)}/{len(records_all)}")
    log(f"  state_dir    : {relpath_for_log(state_dir)}")
    log(f"  sharding_dir : {relpath_for_log(sharding_dir)}")
    log(f"  forget_set   : {relpath_for_log(args.forget_set_path)}")
    log(f"  clean_source : {args.clean_source_template}")
    log(f"  method       : {method}")
    log(f"  temporal_mode: rolling (window_size={int(cfg.temporal_graph.window_size)})")
    log(f"  out_dir      : {relpath_for_log(out_dir)}")
    log("=" * 60)

    eval_data_dir = "" if args.skip_eval else _resolve_eval_data_dir(state_dir, args.eval_data_dir)

    idx = _get_or_build_index(state_dir, sharding_dir, args.index_path, False, log)
    log(f"Index: {idx.summary()}")
    all_shard_ids = list(range(idx.num_shards))

    plans = _plan_steps(args.forget_type, records, idx)
    labels = [p["label"] for p in plans]
    step_shards: List[List[int]] = [sorted(p["shards"]) for p in plans]
    zero = [i + 1 for i, s in enumerate(step_shards) if not s]
    if zero:
        log(f"[WARN] {len(zero)} requests affect no shard (not in the train split) — steps {zero[:10]} pass without retraining")
    union_shards = sorted({s for ss in step_shards for s in ss})
    log(f"Affected shards: {sum(len(s) for s in step_shards) / max(1, len(step_shards)):.1f} per request on average, "
        f"{sum(len(s) for s in step_shards)} retrains in total, union {len(union_shards)}/{idx.num_shards}")

    num_days = sum(1 for n in os.listdir(os.path.join(sharding_dir, "partition")) if n.startswith("day") and n.endswith(".pkl"))
    clean_dirs = [args.clean_source_template.format(step=k) for k in range(1, len(records) + 1)]
    missing = [d for d in clean_dirs if len(os.listdir(os.path.join(d, "daily_graphs"))
                                            if os.path.isdir(os.path.join(d, "daily_graphs")) else []) != num_days]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prefix clean datasets are missing or lack {num_days} daily_graphs — e.g. {missing[:3]}")

    dst_graphsage = os.path.join(out_dir, "graphsage_runs")
    dst_sharding = os.path.join(out_dir, "Sharding")
    dst_processed = os.path.join(dst_sharding, "processed")
    stream_log_path = os.path.join(out_dir, STREAM_LOG)
    setup_marker = os.path.join(out_dir, ".stream_setup_done")

    if not os.path.exists(setup_marker):
        for partial in (dst_graphsage, dst_processed):
            if os.path.lexists(partial):
                shutil.rmtree(partial)
        log("Setup: graphsage_runs/ (copy args/results of all shards + link source best.pt as shard_run_best.pt) ...")
        _copy_graphsage_runs_selective(os.path.join(state_dir, "graphsage_runs"), dst_graphsage, set(), log)
        log("Setup: Sharding/processed/ (link every source shard; a shard is rebuilt when a request affects it) ...")
        _link_untouched_shards(os.path.join(sharding_dir, "processed"), dst_processed, all_shard_ids)
        _copy_partition_metadata(sharding_dir, dst_sharding, log)
        shutil.copy2(run_cfg_path, os.path.join(out_dir, "config_used.yaml"))
        with open(setup_marker, "w") as f:
            f.write(datetime.now().isoformat())
    else:
        log("Setup: reusing the existing working copy (resume)")

    done = _read_done_steps(stream_log_path)
    if done:
        log(f"Resume: {len(done)} steps already recorded in stream_log — skipped")

    label_key = {"uid": "uid", "node": "nid", "path": "path_key"}[args.forget_type]
    for step, (label, shards) in enumerate(zip(labels, step_shards), start=1):
        if step in done:
            continue
        t_step = time.time()
        clean_dir = clean_dirs[step - 1]
        log(f"\n=== [step {step}/{len(records)}] {label_key}={label}  affected={len(shards)} → {shards} ===")
        log(f"  clean source (requests 1..{step} removed): {relpath_for_log(clean_dir)}")

        # Affected shards are rebuilt from scratch from the clean data of requests 1..step
        # (the source partitions are kept); rerunning an interrupted step gives the same data.
        t0 = time.time()
        removed_sessions: Optional[int] = 0
        if shards:
            shard_dirs = [os.path.join(dst_processed, f"shard_{s}") for s in shards]
            # None if a shard was left incomplete by an interrupted run of this step.
            sessions_before = (sum(_train_sessions(d) for d in shard_dirs)
                               if all(os.path.exists(os.path.join(d, "next_location_meta.pt")) for d in shard_dirs) else None)
            for d in shard_dirs:
                if os.path.islink(d):
                    os.remove(d)
                elif os.path.isdir(d):
                    shutil.rmtree(d)
            sessions_after = _rebuild_target_shards_from_clean(cfg, sharding_dir, clean_dir, dst_processed, shards, log)
            removed_sessions = (None if sessions_before is None
                                else sessions_before - sum(v["train"] for v in sessions_after.values()))
        prep_s = time.time() - t0

        t0 = time.time()
        retrain_results = retrain_shards_parallel(
            dst_graphsage, dst_processed, shards, args.num_gpus, log,
        ) if shards else {}
        retrain_wallclock_s = time.time() - t0
        failed = [k for k, ok in retrain_results.items() if not ok]
        if failed:
            log(f"[ERROR] step {step} retrain failed: {failed} — stopping (rerun with the same --out_dir to resume from this step)")
            sys.exit(1)

        timing = _shard_timing_s(dst_graphsage, shards)
        train_sum_s = sum(v["train_val_s"] for v in timing.values())
        rec = {
            "step": step,
            label_key: label,
            "affected_shards": shards,
            "n_affected": len(shards),
            "removed_sessions": removed_sessions,
            "clean_source_dir": clean_dir,
            "prep_s": round(prep_s, 1),
            "retrain_wall_s": round(train_sum_s / max(1, args.num_gpus), 1),
            "retrain_wallclock_s": round(retrain_wallclock_s, 1),
            "step_wall_s": round(time.time() - t_step, 1),
            "per_shard": {str(k): v for k, v in timing.items()},
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(stream_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"[step {step}] done  removed_train_sessions={removed_sessions}  prep={prep_s:.0f}s  "
            f"retrain_wall(sum/{args.num_gpus})={rec['retrain_wall_s']:.0f}s  "
            f"wallclock={retrain_wallclock_s:.0f}s  step_wall={rec['step_wall_s']:.0f}s")

    done = _read_done_steps(stream_log_path)
    steps = [done[i] for i in sorted(done) if i <= len(records)]
    eval_results, agg_eval_ms = None, None
    if not args.skip_eval and len(steps) == len(records):
        t0 = time.perf_counter()
        eval_results = run_eval(out_dir, eval_data_dir, args.num_gpus, log,
                                cfg=cfg, sharding_dir=dst_sharding)
        agg_eval_ms = round((time.perf_counter() - t0) * 1000)

    results = {
        "experiment_type": f"graphsage_{args.forget_type}_unlearn_stream",
        "forget_type": args.forget_type,
        "source_state_dir": state_dir,
        "source_sharding_dir": sharding_dir,
        "out_dir": out_dir,
        "forget_set_path": os.path.abspath(args.forget_set_path),
        "clean_source_template": args.clean_source_template,
        "num_gpus": args.num_gpus,
        "eval_data_dir": eval_data_dir or None,
        "num_steps": len(records),
        "num_steps_done": len(steps),
        "union_affected_shards": union_shards,
        "total_shard_retrains": sum(r["n_affected"] for r in steps),
        "timing_s": {
            "retrain_wall_s_total": round(sum(r["retrain_wall_s"] for r in steps), 1),
            "retrain_wallclock_s_total": round(sum(r["retrain_wallclock_s"] for r in steps), 1),
            "prep_s_total": round(sum(r["prep_s"] for r in steps), 1),
            "agg_eval_test_s": round(agg_eval_ms / 1000, 1) if agg_eval_ms is not None else None,
            "wall_time_s_this_invocation": round(time.time() - wall_start, 1),
        },
        "eval": eval_results,
    }
    with open(os.path.join(out_dir, RESULT_FILENAME), "w") as f:
        json.dump(results, f, indent=2, default=str)

    log("=" * 60)
    log(f"UNLEARN STREAM COMPLETE  type={args.forget_type}  steps={len(steps)}/{len(records)}  "
        f"shard retrains={results['total_shard_retrains']}  "
        f"retrain_wall_total={results['timing_s']['retrain_wall_s_total'] / 3600:.2f}h")
    if eval_results:
        r = eval_results.get("routable", {})
        log(f"  [routable] ACC@1={r.get('acc@1', '?')}  MRR={r.get('mrr', '?')}")
    log(f"  results → {relpath_for_log(os.path.join(out_dir, RESULT_FILENAME))}")
    log("=" * 60)


if __name__ == "__main__":
    main()
