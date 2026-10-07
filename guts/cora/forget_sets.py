from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.cora.data import load_cora, DEFAULT_CORA_ROOT


NODE_RATIOS = [0.05, 0.10, 0.15, 0.20]
NODE_FIXED_COUNTS = [2, 4, 10, 20, 100]

EDGE_RATIOS = [0.005, 0.02, 0.05, 0.10]
EDGE_FIXED_COUNTS = [1, 2, 5, 10, 100]

DEFAULT_SEEDS = list(range(2026, 2036))


def _degree_bucket(degree: int) -> str:
    if degree <= 2:
        return "1-2"
    if degree <= 5:
        return "3-5"
    if degree <= 15:
        return "6-15"
    return "16+"


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


def _save(out_dir: str, fname: str, data: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  saved -> {path}  (targets={data['target_count']})", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-type generators
# ─────────────────────────────────────────────────────────────────────────────

def gen_node_forget_sets(degree: Dict[int, int], out_dir: str, seed: int) -> None:
    rng = random.Random(seed)
    all_nodes = list(degree.keys())

    buckets: Dict[str, list] = defaultdict(list)
    for node in all_nodes:
        buckets[_degree_bucket(degree[node])].append(node)

    def _build_and_save(n: int, tag: str, ratio: Optional[float]) -> None:
        chosen = _sample_from_buckets(buckets, len(all_nodes), n, rng)
        records = [
            {
                "nid": nid,
                "degree": degree[nid],
                "degree_bucket": _degree_bucket(degree[nid]),
            }
            for nid in chosen
        ]
        fname = f"forget_nodes_seed{seed}_{tag}.json"
        _save(out_dir, fname, {
            "forget_type": "node",
            "dataset": "cora",
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


def gen_edge_forget_sets(unique_edges: List[Tuple[int, int]], degree: Dict[int, int], out_dir: str, seed: int) -> None:
    rng = random.Random(seed)

    def _edge_bucket(e: Tuple[int, int]) -> str:
        return _degree_bucket(degree[e[0]] + degree[e[1]])

    buckets: Dict[str, list] = defaultdict(list)
    for e in unique_edges:
        buckets[_edge_bucket(e)].append(e)

    def _build_and_save(n: int, tag: str, ratio: Optional[float]) -> None:
        chosen = _sample_from_buckets(buckets, len(unique_edges), n, rng)
        records = [
            {
                "src": src,
                "dst": dst,
                "endpoint_degree_sum": degree[src] + degree[dst],
            }
            for src, dst in chosen
        ]
        fname = f"forget_edges_seed{seed}_{tag}.json"
        _save(out_dir, fname, {
            "forget_type": "edge",
            "dataset": "cora",
            "ratio": ratio,
            "target_count": len(chosen),
            "seed": seed,
            "records": records,
        })

    for ratio in EDGE_RATIOS:
        n = max(1, math.ceil(len(unique_edges) * ratio))
        tag = f"ratio{ratio:.5f}".rstrip("0")
        _build_and_save(n, tag, ratio)

    for count in EDGE_FIXED_COUNTS:
        n = min(count, len(unique_edges))
        _build_and_save(n, f"n{n}", None)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_TYPES = ["node", "edge"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cora forget set generator (node/edge only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cora_root", default=DEFAULT_CORA_ROOT)
    p.add_argument("--out_root", default="", help="output root (default: processed/forget_sets/cora/)")
    p.add_argument("--types", nargs="+", choices=ALL_TYPES, default=ALL_TYPES)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="split seeds; a forget set is built from each seed's train pool (default: 42-51)")
    return p.parse_args()


def _gen_for_one_split(sample, seed: int, out_root: str, types: list) -> None:
    src = sample.edge_index[0].tolist()
    dst = sample.edge_index[1].tolist()
    train_ids = set(int(i) for i in sample.train_mask.nonzero(as_tuple=True)[0].tolist())

    degree: Dict[int, int] = defaultdict(int)
    for u, v in zip(src, dst):
        degree[u] += 1
    for nid in range(sample.num_nodes):
        degree.setdefault(nid, 0)

    unique_edges = sorted({(min(u, v), max(u, v)) for u, v in zip(src, dst) if u != v})
    train_degree = {nid: d for nid, d in degree.items() if nid in train_ids}
    train_edges = [e for e in unique_edges if e[0] in train_ids and e[1] in train_ids]

    print(f"[split_seed={seed}] train_nodes={len(train_ids)}  "
          f"train-only_unique_edges={len(train_edges)}")

    if "node" in types:
        gen_node_forget_sets(train_degree, os.path.join(out_root, "Node"), seed)

    if "edge" in types:
        gen_edge_forget_sets(train_edges, dict(degree), os.path.join(out_root, "Edge"), seed)


def main() -> None:
    args = parse_args()
    out_root = args.out_root or os.path.join(_REPO_ROOT, "processed", "forget_sets", "cora")

    probe = load_cora(args.cora_root, split_seed=args.seeds[0])
    print(f"Cora loaded: nodes={probe.num_nodes}  directed_edges={probe.num_edges}  "
          f"unique_undirected_edges={probe.num_edges // 2}  split_seeds={args.seeds}")

    for seed in args.seeds:
        sample = load_cora(args.cora_root, split_seed=seed)
        _gen_for_one_split(sample, seed, out_root, args.types)

    print("Done.")


if __name__ == "__main__":
    main()
