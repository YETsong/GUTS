"""Stream unlearning: process the requests of a forget set one at a time on a trained run.

Step k rebuilds the shards affected by request k from the clean data without requests
1..k (source partitions kept) and retrains them; the final run is evaluated once.
Each finished step is appended to <out>/stream_steps.jsonl, so rerunning with the same
--out_dir resumes. Console and <out>/run_log.txt: one line per step plus the retrained
shards of the GPU 0 lane; the evaluation details go to <out>/eval_log.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from guts.method.utils import ensure_dir, timestamp_id, relpath_for_log, repo_relpath, run_logger  # noqa: E402
from guts.yjmob.config import load_yjmob_config  # noqa: E402
from guts.yjmob.unlearning.unlearn_batch import (  # noqa: E402
    _copy_graphsage_runs_selective,
    _copy_partition_metadata,
    _detect_method,
    _build_index,
    _link_untouched_shards,
    _rebuild_target_shards_from_clean,
    _resolve_eval_data_dir,
    _resolve_sharding_dir,
    retrain_shards_parallel,
    run_eval,
)

STEPS_FILE = "stream_steps.jsonl"
RESULT_FILENAME = "results_stream.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="yjmob GraphSAGE stream unlearning, one request at a time (user/node/traj)")
    p.add_argument("--state_dir", required=True, help="source shard-runs directory")
    p.add_argument("--sharded-dir", dest="sharded_dir", default="",
                   help="source Sharding/ path (default: state_dir/Sharding)")
    p.add_argument("--forget_type", default="user", choices=["user", "node", "traj"],
                   help="request type: user / node / traj")
    p.add_argument("--forget_set_path", required=True,
                   help="forget set JSON; records are processed in order")
    p.add_argument("--clean_source_template", required=True,
                   help="per-step clean processed dir template with '{step}', "
                        "e.g. processed/yjmob_1_14d_clean/user-stream_seed2026/k{step:03d}")
    p.add_argument("--out_root", default="",
                   help="output root (default: shard-unlearns/<dataset>/ next to state_dir)")
    p.add_argument("--out_dir", default="",
                   help="output directory (pass an existing one to resume); created if empty")
    p.add_argument("--run_tag", default="", help="tag appended to the output directory name")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="process only the first N requests (0 = all)")
    p.add_argument("--skip_eval", action="store_true", help="skip the final evaluation")
    p.add_argument("--eval_data_dir", default="", help="override the test processed directory")
    return p.parse_args()


def _read_done_steps(steps_path: str) -> Dict[int, dict]:
    done: Dict[int, dict] = {}
    if os.path.exists(steps_path):
        with open(steps_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[int(rec["step"])] = rec
    return done


def _shard_step_record(shard_run_dir: str) -> Optional[dict]:
    path = os.path.join(shard_run_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        res = json.load(f)
    t = res.get("timing_ms") or {}
    return {
        "train_val_s": round(((t.get("train") or 0) + (t.get("val_eval") or 0)) / 1000, 2),
        "best_epoch": res.get("best_epoch"),
        "stopped_epoch": res.get("stopped_epoch"),
        "best_val_mrr": res.get("best_val_mrr"),
        "train_at_best": res.get("best_train_metrics"),
        "val_at_best": res.get("best_val_metrics"),
    }


def _shard_timing_s(dst_graphsage: str, shard_ids: List[int]) -> Dict[int, dict]:
    out = {}
    for s in shard_ids:
        rec = _shard_step_record(os.path.join(dst_graphsage, f"shard_{s}"))
        if rec is not None:
            out[s] = rec
    return out


def _traj_key(r: dict) -> tuple:
    return (int(r["uid"]), int(r["day"]), int(r["start_time"]), int(r["end_time"]))


def _plan_steps(forget_type: str, records: List[dict], idx) -> List[dict]:
    """Label and affected shards of each request, in order."""
    if forget_type == "user":
        return [{"label": int(r["uid"]), "shards": set(idx.affected_shards_for_users([int(r["uid"])]))} for r in records]
    if forget_type == "node":
        return [{"label": int(r["nid"]), "shards": set(idx.affected_shards_for_nodes([int(r["nid"])]))} for r in records]
    return [{"label": list(_traj_key(r)), "shards": set(idx.affected_shards_for_trajs({_traj_key(r)}))} for r in records]


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

    log = run_logger(os.path.join(out_dir, "run_log.txt"))

    run_cfg_path = os.path.join(state_dir, "config_used.yaml")
    cfg = load_yjmob_config(run_cfg_path)

    with open(args.forget_set_path) as f:
        records_all = json.load(f)["records"]
    records = records_all[: args.max_steps] if args.max_steps else records_all

    log(f"GUTS stream unlearn ({args.forget_type}, {len(records)}/{len(records_all)} requests)  {time.strftime('%Y-%m-%d')}"
        f"  source={relpath_for_log(state_dir)}  out={relpath_for_log(out_dir)}  GPUs={args.num_gpus}")

    eval_data_dir = "" if args.skip_eval else _resolve_eval_data_dir(state_dir, args.eval_data_dir)

    idx = _build_index(sharding_dir, log)
    all_shard_ids = list(range(idx.num_shards))

    plans = _plan_steps(args.forget_type, records, idx)
    labels = [p["label"] for p in plans]
    step_shards: List[List[int]] = [sorted(p["shards"]) for p in plans]
    zero = [i + 1 for i, s in enumerate(step_shards) if not s]
    if zero:
        log(f"[WARN] {len(zero)} requests affect no shard (not in the train split) — steps {zero[:10]} pass without retraining")
    union_shards = sorted({s for ss in step_shards for s in ss})
    log(f"  affected shards: {sum(len(s) for s in step_shards) / max(1, len(step_shards)):.1f} per request on average, "
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
    steps_path = os.path.join(out_dir, STEPS_FILE)
    setup_marker = os.path.join(out_dir, ".stream_setup_done")

    if not os.path.exists(setup_marker):
        for partial in (dst_graphsage, dst_processed):
            if os.path.lexists(partial):
                shutil.rmtree(partial)
        _copy_graphsage_runs_selective(os.path.join(state_dir, "graphsage_runs"), dst_graphsage, set())
        _link_untouched_shards(os.path.join(sharding_dir, "processed"), dst_processed, all_shard_ids)
        _copy_partition_metadata(sharding_dir, dst_sharding)
        shutil.copy2(run_cfg_path, os.path.join(out_dir, "config_used.yaml"))
        with open(setup_marker, "w") as f:
            f.write(datetime.now().isoformat())

    done = _read_done_steps(steps_path)
    if done:
        log(f"  resume: {len(done)} steps already recorded in {STEPS_FILE}")

    label_key = {"user": "uid", "node": "nid", "traj": "traj_key"}[args.forget_type]
    for step, (label, shards) in enumerate(zip(labels, step_shards), start=1):
        if step in done:
            continue
        t_step = time.time()
        clean_dir = clean_dirs[step - 1]
        log(f"step {step}/{len(records)}  {label_key}={label}  affected {len(shards)} shards")

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
            # Rebuild progress is not printed per step (one line per step is enough).
            sessions_after = _rebuild_target_shards_from_clean(cfg, sharding_dir, clean_dir, dst_processed, shards,
                                                               lambda msg: None)
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
            "clean_source_dir": repo_relpath(clean_dir),
            "prep_s": round(prep_s, 1),
            "retrain_wall_s": round(train_sum_s / max(1, args.num_gpus), 1),
            "retrain_wallclock_s": round(retrain_wallclock_s, 1),
            "step_wall_s": round(time.time() - t_step, 1),
            "per_shard": {str(k): v for k, v in timing.items()},
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(steps_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"  step {step} done  removed train sessions {removed_sessions}  rebuild {prep_s:.0f}s"
            f"  retrain {retrain_wallclock_s:.0f}s  step {rec['step_wall_s']:.0f}s")

    done = _read_done_steps(steps_path)
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
        "source_state_dir": repo_relpath(state_dir),
        "source_sharding_dir": repo_relpath(sharding_dir),
        "forget_set_path": repo_relpath(args.forget_set_path),
        "clean_source_template": repo_relpath(args.clean_source_template),
        "num_gpus": args.num_gpus,
        "eval_data_dir": repo_relpath(eval_data_dir) or None,
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

    log(f"done  {len(steps)}/{len(records)} steps, {results['total_shard_retrains']} shard retrains"
        f"  -> {relpath_for_log(os.path.join(out_dir, RESULT_FILENAME))}")


if __name__ == "__main__":
    main()
