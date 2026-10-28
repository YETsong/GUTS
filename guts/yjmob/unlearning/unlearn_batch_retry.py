"""Finish an unlearn_batch.py run whose shard retraining failed.

Reads <out_dir>/unlearn_request.json written by unlearn_batch.py, retrains the affected
shards that have no results.json yet (resuming from checkpoints/last.pt when an
interrupted run left one), evaluates, and writes results_unlearn.json in the same
format as unlearn_batch.py. Console lines are appended to the run's run_log.txt.
Only for unlearn_batch.py outputs; unlearn_stream.py resumes by
itself when rerun with the same --out_dir.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.method.utils import relpath_for_log, repo_relpath, resolve_repo_path, run_logger
from guts.yjmob.config import load_yjmob_config
from guts.yjmob.unlearning.unlearn_batch import (
    _resolve_eval_data_dir,
    load_unlearn_request,
    retrain_shards_parallel,
    run_eval,
    write_unlearn_results,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="output directory of guts/yjmob/unlearning/unlearn_batch.py")
    p.add_argument("--shard_ids", type=int, nargs="*", default=None,
                   help="shards to retrain (default: affected shards without results.json)")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--eval_data_dir", default="", help="test data (default: the one recorded by unlearn_batch.py)")
    return p.parse_args(argv)


def _pending_shards(dst_graphsage: str, affected: List[int]) -> List[int]:
    return [s for s in affected
            if not os.path.exists(os.path.join(dst_graphsage, f"shard_{s}", "results.json"))]


def main(argv: Optional[List[str]] = None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)
    wall_start = time.time()

    available_gpus = torch.cuda.device_count()
    if available_gpus > 0 and args.num_gpus > available_gpus:
        print(f"[WARN] --num_gpus {args.num_gpus} > available GPUs {available_gpus} — using {available_gpus}", flush=True)
        args.num_gpus = available_gpus

    out_dir = os.path.abspath(args.out_dir)
    request = load_unlearn_request(out_dir)
    dst_graphsage = os.path.join(out_dir, "graphsage_runs")
    dst_processed = os.path.join(out_dir, "Sharding", "processed")
    log = run_logger(os.path.join(out_dir, "run_log.txt"))
    log(f"GUTS unlearn retry  {time.strftime('%Y-%m-%d')}  out={relpath_for_log(out_dir)}  GPUs={args.num_gpus}")

    affected = [int(s) for s in request["affected_shards"]]
    shard_ids = args.shard_ids if args.shard_ids else _pending_shards(dst_graphsage, affected)
    unknown = sorted(set(shard_ids) - set(affected))
    if unknown:
        raise ValueError(f"--shard_ids {unknown} are not affected shards of this run: {affected}")
    for s in shard_ids:
        if not os.path.exists(os.path.join(dst_processed, f"shard_{s}", "next_location_meta.pt")):
            raise FileNotFoundError(
                f"shard_{s} data was not rebuilt (unlearn_batch.py stopped before retraining); rerun unlearn_batch.py"
            )

    if not shard_ids:
        log("  no shards to retry (all affected shards have results.json)")
    else:
        results = retrain_shards_parallel(
            dst_graphsage, dst_processed, shard_ids, args.num_gpus, log, resume=True,
        )
        failed = [k for k, ok in results.items() if not ok]
        if failed:
            log(f"[ERROR] still failing after retry: {failed}")
            sys.exit(1)

    eval_results = None
    agg_eval_ms = None
    if not args.skip_eval:
        state_dir = resolve_repo_path(request["source_state_dir"])
        eval_data_dir = _resolve_eval_data_dir(state_dir, args.eval_data_dir or request["eval_data_dir"] or "")
        request["eval_data_dir"] = repo_relpath(eval_data_dir)
        cfg = load_yjmob_config(os.path.join(state_dir, "config_used.yaml"))
        t0_eval = time.perf_counter()
        eval_results = run_eval(
            out_dir, eval_data_dir, args.num_gpus, log,
            cfg=cfg, sharding_dir=os.path.join(out_dir, "Sharding"),
        )
        agg_eval_ms = round((time.perf_counter() - t0_eval) * 1000)

    retry_wall_time = round(time.time() - wall_start, 1)
    retrain_results = {f"shard_{s}": True for s in affected}
    # The failed unlearn_batch.py run never recorded its total time; only the retry run's time is known.
    results_path = write_unlearn_results(
        out_dir, request, retrain_results, eval_results, agg_eval_ms, None,
        extra={"retry": {"retried_shards": shard_ids, "wall_time_s": retry_wall_time}},
    )
    log(f"done  {retry_wall_time:.0f}s  -> {relpath_for_log(results_path)}")


if __name__ == "__main__":
    main()
