from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.yjmob.unlearning.forget_sets import (  # noqa: E402
    _degree_bucket, _path_length_bucket, _sample_from_buckets, _save,
)
from guts.yjmob.unlearning.raw_index import RawUnlearnIndex  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="node/path forget sets for stream unlearning (stratified, shuffled)")
    p.add_argument("--index_path", default=os.path.join(_REPO_ROOT, "processed/yjmob_1_14d/raw_unlearn_index.pkl"))
    p.add_argument("--base_exclusion", default="",
                   help="JSON of users excluded when building the source run (records[].uid); none if empty")
    p.add_argument("--out_root", default=os.path.join(_REPO_ROOT, "processed", "forget_sets", "yjmob"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    idx = RawUnlearnIndex.load(args.index_path)
    base_uids: set = set()
    if args.base_exclusion:
        with open(args.base_exclusion) as f:
            base_uids = {int(r["uid"]) for r in json.load(f)["records"]}
    meta_common = {"seed": args.seed, "order": "shuffled", "purpose": "stream unlearning (one request at a time, in record order)"}

    # ── node ──
    rng = random.Random(args.seed)
    degree = idx.node_out_degree()
    buckets = defaultdict(list)
    for nid in idx.node_to_paths:
        buckets[_degree_bucket(degree.get(nid, 0))].append(nid)
    chosen = _sample_from_buckets(buckets, sum(map(len, buckets.values())), args.n, rng)
    rng.shuffle(chosen)
    records = []
    for nid in chosen:
        pids = idx.node_to_paths.get(nid, [])
        records.append({
            "nid": nid, "out_degree": degree.get(nid, 0), "degree_bucket": _degree_bucket(degree.get(nid, 0)),
            "affected_path_ids": pids, "affected_path_count": len(pids), "affected_session_count": len(pids),
            "affected_step_count": sum(max(0, len(idx.path_meta[p]["cells"]) - 1) for p in pids),
        })
    _save(os.path.join(args.out_root, "Node"), f"forget_nodes_seed{args.seed}_n{args.n}_stream.json",
          {"forget_type": "node", "ratio": None, "target_count": len(records), **meta_common, "records": records})
    print("node:", [(r["nid"], r["degree_bucket"]) for r in records])

    # ── path ──
    rng = random.Random(args.seed)
    buckets = defaultdict(list)
    for pid, m in idx.path_meta.items():
        if int(m["uid"]) not in base_uids:
            buckets[_path_length_bucket(len(m["cells"]))].append(pid)
    chosen = _sample_from_buckets(buckets, sum(map(len, buckets.values())), args.n, rng)
    rng.shuffle(chosen)
    records = []
    for pid in chosen:
        m = idx.path_meta[pid]
        edges = idx.path_to_edges.get(pid, [])
        records.append({
            "path_id": pid, "uid": m["uid"], "day": m["day"], "start_time": m["start_time"], "end_time": m["end_time"],
            "path_length": len(m["cells"]), "path_length_bucket": _path_length_bucket(len(m["cells"])),
            "node_sequence": m["cells"], "edge_sequence": [list(e) for e in edges],
            "affected_node_count": len(set(m["cells"])), "affected_edge_count": len(set(edges)),
        })
    _save(os.path.join(args.out_root, "Path"), f"forget_paths_seed{args.seed}_n{args.n}_stream.json",
          {"forget_type": "path", "ratio": None, "target_count": len(records), **meta_common, "records": records})
    print("path:", [(r["uid"], r["path_length_bucket"]) for r in records])


if __name__ == "__main__":
    main()
