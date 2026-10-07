from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.yjmob.unlearning.raw_index import RawUnlearnIndex


# Node: 0.5%, 1%, 5% (main) + 10% (stress)
NODE_RATIOS = [0.005, 0.01, 0.05, 0.10]

NODE_FIXED_COUNTS = [10, 15, 20, 25, 30]

# Edge: 0.05%, 0.1%, 0.5% (main) + 1%, 5% (stress)
EDGE_RATIOS = [0.0005, 0.001, 0.005, 0.01, 0.05]

EDGE_FIXED_COUNTS = [1000, 1500, 2000, 2500, 3000]

# User: 0.5%, 1%, 5% (main) + 10% (stress)
UID_RATIOS = [0.005, 0.01, 0.05, 0.10]

UID_FIXED_COUNTS = [1, 10, 15, 20, 25]

DAY_COUNTS = [1, 2, 3, 4, 5, 7, "top7", "vol10pct"]

# Path: 0.5%, 1%, 5% (main) + 10% (stress)
PATH_RATIOS = [0.005, 0.01, 0.05, 0.10]

PATH_FIXED_COUNTS = [100, 110, 120, 130, 140]


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _degree_bucket(degree: int) -> str:
    if degree <= 4:
        return "1-4"
    if degree <= 49:
        return "5-49"
    if degree <= 499:
        return "50-499"
    return "500+"


def _weight_bucket(weight: int) -> str:
    if weight == 1:
        return "1"
    if weight <= 4:
        return "2-4"
    if weight <= 9:
        return "5-9"
    if weight <= 49:
        return "10-49"
    return "50+"


def _uid_activity_bucket(sessions: int) -> str:
    if sessions <= 30:
        return "low"
    if sessions <= 60:
        return "mid"
    if sessions <= 90:
        return "high"
    return "very_high"


def _path_length_bucket(length: int) -> str:
    if length <= 5:
        return "short"
    if length <= 10:
        return "medium"
    if length <= 20:
        return "long"
    return "very_long"


def _stratified_sample(pool: list, n: int, key_fn, rng: random.Random) -> list:
    if n >= len(pool):
        return list(pool)
    buckets: Dict[str, list] = defaultdict(list)
    for item in pool:
        buckets[key_fn(item)].append(item)
    return _sample_from_buckets(buckets, len(pool), n, rng)


def _sample_from_buckets(buckets: Dict[str, list], pool_size: int, n: int, rng: random.Random) -> list:
    result: list = []
    result_set: set = set()
    remaining = n
    for bname, bpool in sorted(buckets.items()):
        quota = max(1, round(n * len(bpool) / pool_size))
        quota = min(quota, len(bpool), remaining)
        sampled = rng.sample(bpool, quota)
        result.extend(sampled)
        result_set.update(sampled)
        remaining -= quota

    if remaining > 0:
        bucket_list = list(buckets.values())
        rng.shuffle(bucket_list)
        for bpool in bucket_list:
            if remaining <= 0:
                break
            k = min(len(bpool), remaining * 3 + 5)
            for cand in rng.sample(bpool, k):
                if cand not in result_set:
                    result.append(cand)
                    result_set.add(cand)
                    remaining -= 1
                    if remaining <= 0:
                        break
    return result[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Per-type generators
# ─────────────────────────────────────────────────────────────────────────────

def gen_node_forget_sets(idx: RawUnlearnIndex, out_dir: str, seed: int) -> None:
    rng = random.Random(seed)
    all_nodes = list(idx.node_to_paths.keys())
    node_degree = idx.node_out_degree()

    buckets: Dict[str, list] = defaultdict(list)
    for node in all_nodes:
        buckets[_degree_bucket(node_degree.get(node, 0))].append(node)

    os.makedirs(out_dir, exist_ok=True)

    def _build_and_save(n: int, tag: str, ratio: Optional[float]) -> None:
        chosen = _sample_from_buckets(buckets, len(all_nodes), n, rng)
        records = []
        for nid in chosen:
            path_ids = idx.node_to_paths.get(nid, [])
            records.append({
                "nid": nid,
                "out_degree": node_degree.get(nid, 0),
                "degree_bucket": _degree_bucket(node_degree.get(nid, 0)),
                "affected_path_ids": path_ids,
                "affected_path_count": len(path_ids),
                "affected_session_count": len(path_ids),
                "affected_step_count": sum(
                    max(0, len(idx.path_meta[pid]["cells"]) - 1)
                    for pid in path_ids
                ),
            })
        fname = f"forget_nodes_seed{seed}_{tag}.json"
        _save(out_dir, fname, {
            "forget_type": "node",
            "ratio": ratio,
            "target_count": len(chosen),
            "seed": seed,
            "records": records,
        })

    for ratio in NODE_RATIOS:
        n = max(1, math.ceil(len(all_nodes) * ratio))
        tag = f"ratio{ratio:.4f}".rstrip("0")
        _build_and_save(n, tag, ratio)

    for count in NODE_FIXED_COUNTS:
        n = min(count, len(all_nodes))
        _build_and_save(n, f"n{n}", None)


def gen_edge_forget_sets(idx: RawUnlearnIndex, out_dir: str, seed: int) -> None:
    rng = random.Random(seed)
    edge_weight = idx.edge_weight()
    all_edges = list(edge_weight.keys())

    buckets: Dict[str, list] = defaultdict(list)
    for e in all_edges:
        buckets[_weight_bucket(edge_weight.get(e, 1))].append(e)

    os.makedirs(out_dir, exist_ok=True)

    def _build_and_save(n: int, tag: str, ratio: Optional[float]) -> None:
        chosen = _sample_from_buckets(buckets, len(all_edges), n, rng)
        records = []
        for src, dst in chosen:
            path_ids = idx.edge_to_paths.get((src, dst), [])
            w = edge_weight.get((src, dst), 0)
            records.append({
                "src": src,
                "dst": dst,
                "weight": w,
                "weight_bucket": _weight_bucket(edge_weight.get((src, dst), 1)),
                "affected_path_ids": path_ids,
                "affected_transition_occurrences": w,
                "affected_session_count": len(path_ids),
                "affected_step_count": sum(
                    max(0, len(idx.path_meta[pid]["cells"]) - 1)
                    for pid in path_ids
                ),
            })
        fname = f"forget_edges_seed{seed}_{tag}.json"
        _save(out_dir, fname, {
            "forget_type": "edge",
            "ratio": ratio,
            "target_count": len(chosen),
            "seed": seed,
            "records": records,
        })

    for ratio in EDGE_RATIOS:
        n = max(1, math.ceil(len(all_edges) * ratio))
        tag = f"ratio{ratio:.5f}".rstrip("0")
        _build_and_save(n, tag, ratio)

    for count in EDGE_FIXED_COUNTS:
        n = min(count, len(all_edges))
        _build_and_save(n, f"n{n}", None)


def gen_uid_forget_sets(idx: RawUnlearnIndex, out_dir: str, seed: int) -> None:
    rng = random.Random(seed)
    uid_stats = idx.uid_stats()
    all_uids = list(uid_stats.keys())

    buckets: Dict[str, list] = defaultdict(list)
    for uid in all_uids:
        buckets[_uid_activity_bucket(uid_stats[uid]["session_count"])].append(uid)

    os.makedirs(out_dir, exist_ok=True)

    def _build_and_save(n: int, tag: str, ratio: Optional[float]) -> None:
        chosen = _sample_from_buckets(buckets, len(all_uids), n, rng)
        records = []
        for uid in chosen:
            path_ids = idx.uid_to_paths.get(uid, [])
            affected_nodes: Set[int] = set()
            affected_edges: Set[Tuple[int, int]] = set()
            for pid in path_ids:
                affected_nodes.update(idx.path_meta[pid]["cells"])
                affected_edges.update(idx.path_to_edges.get(pid, []))
            records.append({
                "uid": uid,
                "session_count": uid_stats[uid]["session_count"],
                "step_count": uid_stats[uid]["step_count"],
                "path_ids": path_ids,
                "affected_node_count": len(affected_nodes),
                "affected_edge_count": len(affected_edges),
            })
        fname = f"forget_users_seed{seed}_{tag}.json"
        _save(out_dir, fname, {
            "forget_type": "uid",
            "ratio": ratio,
            "target_count": len(chosen),
            "seed": seed,
            "records": records,
        })

    for ratio in UID_RATIOS:
        n = max(1, math.ceil(len(all_uids) * ratio))
        tag = f"ratio{ratio:.4f}".rstrip("0")
        _build_and_save(n, tag, ratio)

    for count in UID_FIXED_COUNTS:
        n = min(count, len(all_uids))
        tag = f"n{n}"
        _build_and_save(n, tag, None)


def gen_day_forget_sets(idx: RawUnlearnIndex, out_dir: str, seed: int) -> None:
    rng = random.Random(seed)
    day_stats = idx.day_stats()
    day_to_uids = idx.day_to_uid_set()

    counts = sorted(s["session_count"] for s in day_stats.values())
    threshold = counts[max(0, len(counts) // 10)]
    candidate_days = [
        d for d, s in day_stats.items()
        if s["session_count"] > threshold
    ]
    total_train_sessions = sum(s["session_count"] for s in day_stats.values())

    os.makedirs(out_dir, exist_ok=True)

    for k in DAY_COUNTS:
        if k == "top7":
            chosen = sorted(
                candidate_days,
                key=lambda d: -day_stats[d]["session_count"]
            )[:7]
            tag = "top7"
        elif k == "vol10pct":
            target_volume = total_train_sessions * 0.10
            ranked = sorted(
                candidate_days,
                key=lambda d: -day_stats[d]["session_count"]
            )
            chosen = []
            cum = 0
            for d in ranked:
                if cum >= target_volume:
                    break
                chosen.append(d)
                cum += day_stats[d]["session_count"]
            tag = "vol10pct"
        else:
            # stratified: low/mid/high volume day
            vol_bucket = lambda d: (
                "low" if day_stats[d]["session_count"] <= counts[len(counts) // 3] else
                "high" if day_stats[d]["session_count"] > counts[2 * len(counts) // 3] else
                "mid"
            )
            chosen = _stratified_sample(candidate_days, int(k), vol_bucket, rng)
            tag = f"k{k}"

        records = []
        for day in chosen:
            path_ids = idx.day_to_paths.get(day, [])
            affected_nodes: Set[int] = set()
            affected_edges: Set[Tuple[int, int]] = set()
            for pid in path_ids:
                affected_nodes.update(idx.path_meta[pid]["cells"])
                affected_edges.update(idx.path_to_edges.get(pid, []))
            records.append({
                "day": day,
                "train_session_count": day_stats[day]["session_count"],
                "train_step_count": day_stats[day]["step_count"],
                "train_user_count": len(day_to_uids.get(day, set())),
                "path_ids": path_ids,
                "affected_node_count": len(affected_nodes),
                "affected_edge_count": len(affected_edges),
            })

        fname = f"forget_days_seed{seed}_{tag}.json"
        _save(out_dir, fname, {
            "forget_type": "day",
            "k": k,
            "target_count": len(chosen),
            "seed": seed,
            "records": records,
        })


def gen_path_forget_sets(idx: RawUnlearnIndex, out_dir: str, seed: int) -> None:
    rng = random.Random(seed)
    all_pids = list(idx.path_meta.keys())

    buckets: Dict[str, list] = defaultdict(list)
    for pid in all_pids:
        buckets[_path_length_bucket(len(idx.path_meta[pid]["cells"]))].append(pid)

    os.makedirs(out_dir, exist_ok=True)

    def _build_and_save(n: int, tag: str, ratio: Optional[float]) -> None:
        chosen = _sample_from_buckets(buckets, len(all_pids), n, rng)
        records = []
        for pid in chosen:
            m = idx.path_meta[pid]
            affected_edges = idx.path_to_edges.get(pid, [])
            records.append({
                "path_id": pid,
                "uid": m["uid"],
                "day": m["day"],
                "start_time": m["start_time"],
                "end_time": m["end_time"],
                "path_length": len(m["cells"]),
                "path_length_bucket": _path_length_bucket(len(m["cells"])),
                "node_sequence": m["cells"],
                "edge_sequence": [list(e) for e in affected_edges],
                "affected_node_count": len(set(m["cells"])),
                "affected_edge_count": len(set(affected_edges)),
            })
        fname = f"forget_paths_seed{seed}_{tag}.json"
        _save(out_dir, fname, {
            "forget_type": "path",
            "ratio": ratio,
            "target_count": len(chosen),
            "seed": seed,
            "records": records,
        })

    for ratio in PATH_RATIOS:
        n = max(1, math.ceil(len(all_pids) * ratio))
        tag = f"ratio{ratio:.4f}".rstrip("0")
        _build_and_save(n, tag, ratio)

    for count in PATH_FIXED_COUNTS:
        n = min(count, len(all_pids))
        _build_and_save(n, f"n{n}", None)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_COMPACT_LIST_FIELDS = ("affected_path_ids",)


def _compact_id_list_field(json_text: str, field_name: str) -> str:
    pattern = re.compile(
        r'("' + re.escape(field_name) + r'": )\[\s*([^\[\]]*?)\s*\]',
        re.DOTALL,
    )

    def _collapse(m: "re.Match[str]") -> str:
        items = [x.strip() for x in m.group(2).split(",") if x.strip()]
        return m.group(1) + "[" + ", ".join(items) + "]"

    return pattern.sub(_collapse, json_text)


def _save(out_dir: str, fname: str, data: dict) -> None:
    path = os.path.join(out_dir, fname)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    for field_name in _COMPACT_LIST_FIELDS:
        text = _compact_id_list_field(text, field_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  saved → {path}  (targets={data['target_count']})", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_TYPES = ["node", "edge", "uid", "day", "path"]

_DEFAULT_SOURCE_DIR = os.path.join(_REPO_ROOT, "processed", "yjmob_1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GraphSAGE unlearning forget set generator (from the original data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source_dir", default=_DEFAULT_SOURCE_DIR,
                   help=f"original (unsharded) next-grid processed directory (default: {_DEFAULT_SOURCE_DIR})")
    p.add_argument("--index_path", default="",
                   help="RawUnlearnIndex cache (default: <source_dir>/raw_unlearn_index.pkl)")
    p.add_argument("--out_root", default="",
                   help="output root (default: processed/forget_sets/yjmob/)")
    p.add_argument("--types", nargs="+", choices=ALL_TYPES, default=ALL_TYPES,
                   help="forget set types to build (default: all)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--rebuild_index", action="store_true",
                   help="force rebuilding the index")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    index_path = args.index_path or os.path.join(args.source_dir, "raw_unlearn_index.pkl")
    if os.path.exists(index_path) and not args.rebuild_index:
        print(f"Loading existing index: {index_path}", flush=True)
        idx = RawUnlearnIndex.load(index_path)
    else:
        print(f"Building index from: {args.source_dir}", flush=True)
        idx = RawUnlearnIndex.build(args.source_dir)
        idx.save(index_path)

    print(f"Index summary: {idx.summary()}", flush=True)

    out_root = args.out_root or os.path.join(_REPO_ROOT, "processed", "forget_sets", "yjmob")

    type_dir_map = {
        "node": os.path.join(out_root, "Node"),
        "edge": os.path.join(out_root, "Edge"),
        "uid":  os.path.join(out_root, "Uid"),
        "day":  os.path.join(out_root, "Day"),
        "path": os.path.join(out_root, "Path"),
    }
    type_gen_map = {
        "node": gen_node_forget_sets,
        "edge": gen_edge_forget_sets,
        "uid":  gen_uid_forget_sets,
        "day":  gen_day_forget_sets,
        "path": gen_path_forget_sets,
    }

    for t in args.types:
        print(f"\n=== Generating [{t}] forget sets ===", flush=True)
        type_gen_map[t](idx, type_dir_map[t], args.seed)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
