"""Static-graph forget sets: node or edge deletion requests drawn from a split file's train part.

Targets are sampled in proportion to quartile strata of a measure (node: degree, edge: the
sum of its endpoints' degrees), computed on the target pool of the given split each time.
One file per percentage, each drawn with a fresh RNG(seed) so sizes are independent:
<out_root>/{Node,Edge}/forget_<type>_seed<S>_<P>pct.json. The split file's path is recorded
as `source` (repository-relative); the clean build and unlearning warn when they are given
different data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from guts.method.forget_sampling import NUM_STRATA, quartile_bounds, sample_from_strata, stratum_labels, stratum_of
from guts.method.reporting import output_row
from guts.method.utils import relpath_for_log, repo_relpath, resolve_seed
from guts.static_graph.cora_data import DATASET, load_cora_sample

DIR_NAMES = {"node": "Node", "edge": "Edge"}
MEASURES = {"node": "degree", "edge": "endpoint degree sum"}


def _pools(sample):
    """Per type: the targets (train nodes; undirected edges between train nodes) and their measure."""
    src = sample.edge_index[0].tolist()
    dst = sample.edge_index[1].tolist()
    degree: Dict[int, int] = defaultdict(int)
    for u in src:
        degree[u] += 1
    train_ids = set(int(i) for i in sample.train_mask.nonzero(as_tuple=True)[0].tolist())
    nodes = sorted(train_ids)
    edges = sorted({(min(u, v), max(u, v)) for u, v in zip(src, dst)
                    if u != v and u in train_ids and v in train_ids})
    return {
        "node": (nodes, lambda n: degree[n], lambda n, s: {"nid": n, "degree": degree[n], "stratum": s}),
        "edge": (edges, lambda e: degree[e[0]] + degree[e[1]],
                 lambda e, s: {"src": e[0], "dst": e[1], "endpoint_degree_sum": degree[e[0]] + degree[e[1]],
                               "stratum": s}),
    }


def generate(forget_type: str, pool: list, measure, record, percents: List[float], seed: int,
             out_root: str, source: str) -> None:
    values = [measure(x) for x in pool]
    bounds = quartile_bounds(values)
    labels = stratum_labels(bounds)
    buckets: Dict[int, list] = defaultdict(list)
    stratum = {}
    for item, value in zip(pool, values):
        stratum[item] = stratum_of(value, bounds)
        buckets[stratum[item]].append(item)

    out_dir = os.path.join(out_root, DIR_NAMES[forget_type])
    os.makedirs(out_dir, exist_ok=True)
    for pct in percents:
        n = max(1, math.ceil(len(pool) * pct / 100))
        chosen = sample_from_strata(buckets, len(pool), n, random.Random(seed))
        path = os.path.join(out_dir, f"forget_{forget_type}_seed{seed}_{pct:g}pct.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "forget_type": forget_type, "percent": pct, "target_count": len(chosen), "seed": seed,
                "source": source,
                "strata": {"method": f"quantile on the target pool, {NUM_STRATA} strata (equal bounds merged)",
                           "measure": MEASURES[forget_type], "upper_bounds": bounds},
                "records": [record(x, labels[stratum[x]]) for x in chosen],
            }, f, indent=2)
        print(output_row(forget_type, f"{pct:g}%{len(chosen):>8} targets -> {relpath_for_log(path)}"), flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{DATASET} forget sets (node/edge)")
    p.add_argument("--data_path", required=True,
                   help="graph and split to draw from (cora_data.py split); targets come from its train part")
    p.add_argument("--percent", type=float, nargs="+", default=[1.0],
                   help="forget set size(s) in percent of the target pool (default: 1)")
    p.add_argument("--types", nargs="+", choices=list(DIR_NAMES), default=list(DIR_NAMES))
    p.add_argument("--seed", type=int, default=None,
                   help="sampling seed (default: random; it is part of the output file names)")
    p.add_argument("--out_root", default="", help=f"output root (default: processed/forget_sets/{DATASET})")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """argv: argument list (as on the command line); None reads sys.argv."""
    args = parse_args(argv)
    bad = [p for p in args.percent if not 0 < p <= 100]
    if bad:
        raise ValueError(f"--percent must be in (0, 100]: {bad}")
    out_root = args.out_root or os.path.join(_REPO_ROOT, "processed", "forget_sets", DATASET)
    seed = resolve_seed(args.seed)
    sample = load_cora_sample(args.data_path)
    print(f"forget sets  {DATASET}  seed {seed}" + (" (random)" if args.seed is None else "")
          + f"  source {relpath_for_log(args.data_path)}", flush=True)
    pools = _pools(sample)
    print(output_row("pool", "   ".join(f"{t} {len(pools[t][0]):,} ({MEASURES[t]})" for t in args.types)), flush=True)
    for t in args.types:
        pool, measure, record = pools[t]
        generate(t, pool, measure, record, args.percent, seed, out_root, repo_relpath(args.data_path))


if __name__ == "__main__":
    main()
