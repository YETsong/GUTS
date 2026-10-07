from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from guts.cora.data import load_cora, save_cora_clean, DEFAULT_CORA_ROOT

DEFAULT_SEEDS = list(range(2026, 2036))
DEFAULT_CLEAN_ROOT = os.path.join(_REPO_ROOT, "processed", "Cora_clean")
FORGET_SET_ROOT = os.path.join(_REPO_ROOT, "processed", "forget_sets", "cora")

_FNAME_RE = re.compile(r"forget_(nodes|edges)_seed(\d+)_(.+)\.json$")


def _tag_from_filename(path: str) -> str:
    m = _FNAME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Unexpected forget set file name: {path}")
    kind, seed, preset = m.groups()
    return f"{kind}_{preset}_seed{seed}"


def _build_node_clean(sample, forget: dict):
    deleted_ids = [int(r["nid"]) for r in forget["records"]]
    assert all(bool(sample.train_mask[nid]) for nid in deleted_ids), \
        "forget set contains non-train nids"

    deleted = torch.zeros(sample.num_nodes, dtype=torch.bool)
    deleted[deleted_ids] = True

    x_clean = sample.x.clone()
    x_clean[deleted] = 0.0

    src, dst = sample.edge_index[0], sample.edge_index[1]
    keep_edge = ~(deleted[src] | deleted[dst])

    train_mask_clean = sample.train_mask.clone()
    train_mask_clean[deleted] = False

    info = f"deleted={len(deleted_ids)} nodes, train {int(sample.train_mask.sum())}->{int(train_mask_clean.sum())}, edges {sample.num_edges}->{int(keep_edge.sum())}"
    return x_clean, sample.edge_index[:, keep_edge], sample.edge_weight[keep_edge], train_mask_clean, info


def _build_edge_clean(sample, forget: dict):
    deleted_pairs = {(int(r["src"]), int(r["dst"])) for r in forget["records"]}
    deleted_pairs |= {(d, s) for s, d in deleted_pairs}

    src, dst = sample.edge_index[0].tolist(), sample.edge_index[1].tolist()
    keep_edge = torch.tensor(
        [(s, d) not in deleted_pairs for s, d in zip(src, dst)], dtype=torch.bool
    )
    info = f"deleted={len(forget['records'])} edges (removed {int((~keep_edge).sum())} directed entries), edges {sample.num_edges}->{int(keep_edge.sum())}"
    return sample.x, sample.edge_index[:, keep_edge], sample.edge_weight[keep_edge], sample.train_mask, info


def build_one(cora_root: str, forget_set_path: str, out_root: str) -> str:
    with open(forget_set_path) as f:
        forget = json.load(f)
    seed = int(forget["seed"])
    sample = load_cora(cora_root, split_seed=seed)

    if forget["forget_type"] == "node":
        x, edge_index, edge_weight, train_mask, info = _build_node_clean(sample, forget)
    elif forget["forget_type"] == "edge":
        x, edge_index, edge_weight, train_mask, info = _build_edge_clean(sample, forget)
    else:
        raise ValueError(f"Unsupported forget_type: {forget['forget_type']}")

    clean_sample = type(sample)(
        day=sample.day, x=x, edge_index=edge_index, edge_weight=edge_weight, y=sample.y,
        train_mask=train_mask, val_mask=sample.val_mask, test_mask=sample.test_mask,
        num_nodes=sample.num_nodes, num_classes=sample.num_classes,
    )

    tag = _tag_from_filename(forget_set_path)
    out_path = os.path.join(out_root, tag, "cora_clean.pt")
    save_cora_clean(clean_sample, out_path)
    print(f"  saved -> {out_path}  ({info})")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build clean Cora data for M_retrain (node/edge)")
    p.add_argument("--cora_root", default=DEFAULT_CORA_ROOT)
    p.add_argument("--out_root", default=DEFAULT_CLEAN_ROOT)
    p.add_argument("--forget_set", default="", help="process a single forget set JSON")
    p.add_argument("--all", action="store_true",
                   help="process all forget_*.json under processed/forget_sets/cora/{Node,Edge}/")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--preset", default="n100", help="preset tag to process per seed when --all is not set")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.forget_set:
        build_one(args.cora_root, args.forget_set, args.out_root)
        return

    if args.all:
        paths = sorted(glob.glob(os.path.join(FORGET_SET_ROOT, "Node", "forget_*.json"))) + \
                sorted(glob.glob(os.path.join(FORGET_SET_ROOT, "Edge", "forget_*.json")))
        for path in paths:
            build_one(args.cora_root, path, args.out_root)
        print(f"Done. ({len(paths)} files)")
        return

    node_dir = os.path.join(FORGET_SET_ROOT, "Node")
    for seed in args.seeds:
        path = os.path.join(node_dir, f"forget_nodes_seed{seed}_{args.preset}.json")
        if not os.path.exists(path):
            print(f"  [WARN] missing, skipped: {path}")
            continue
        build_one(args.cora_root, path, args.out_root)
    print("Done.")


if __name__ == "__main__":
    main()
