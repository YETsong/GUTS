"""Cora data: the stratified 80/20 split, clean graphs for M_retrain, and the train subgraph.

Two commands write the data every other script reads:

  split  makes the split once and saves it (processed/Cora_split/cora_seed<S>.pt).
         The pipeline, forget sets, clean graphs and unlearning all read this file,
         so they share one split without recomputing it from a seed.
  clean  applies forget sets to a split file (the source run's data) and saves the
         clean graphs (processed/Cora_clean/<forget set name>.pt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph

from guts.method.reporting import output_row
from guts.method.utils import relpath_for_log, resolve_repo_path, resolve_seed

DATASET = "Cora"  # the dataset name used in logs, results and default output paths
DEFAULT_CORA_ROOT = os.path.join(_REPO_ROOT, "processed", "Cora")
DEFAULT_SPLIT_ROOT = os.path.join(_REPO_ROOT, "processed", "Cora_split")
DEFAULT_CLEAN_ROOT = os.path.join(_REPO_ROOT, "processed", "Cora_clean")


@dataclass
class CoraSample:
    x: torch.Tensor              # [N, 1433] bag-of-words feature
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    test_mask: torch.Tensor      # [N] bool
    num_nodes: int
    num_classes: int

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def in_dim(self) -> int:
        return int(self.x.size(1))


_CORA_SAMPLE_FIELDS = ["x", "edge_index", "y", "train_mask", "test_mask", "num_nodes", "num_classes"]


# ─────────────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_80_20_split(y: torch.Tensor, num_nodes: int, split_seed: int, train_ratio: float = 0.8):
    rng = np.random.RandomState(split_seed)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in torch.unique(y).tolist():
        idx = (y == c).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_ratio))
        train_mask[idx[:n_train]] = True
        test_mask[idx[n_train:]] = True

    return train_mask, test_mask


def load_cora(root: str = DEFAULT_CORA_ROOT, *, split_seed: int) -> CoraSample:
    os.makedirs(root, exist_ok=True)
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    num_nodes = int(data.num_nodes)
    y = data.y.long()

    train_mask, test_mask = _stratified_80_20_split(y, num_nodes, split_seed)

    return CoraSample(
        x=data.x.float(),
        edge_index=data.edge_index.long(),
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_classes=int(dataset.num_classes),
    )


def induce_train_subgraph(sample: CoraSample) -> Tuple[CoraSample, torch.Tensor]:
    train_idx = sample.train_mask.nonzero(as_tuple=True)[0]
    edge_index_new, _ = subgraph(train_idx, sample.edge_index, relabel_nodes=True)
    n_train = int(train_idx.numel())

    train_sample = CoraSample(
        x=sample.x[train_idx],
        edge_index=edge_index_new,
        y=sample.y[train_idx],
        train_mask=torch.ones(n_train, dtype=torch.bool),
        test_mask=torch.zeros(n_train, dtype=torch.bool),
        num_nodes=n_train,
        num_classes=sample.num_classes,
    )
    return train_sample, train_idx


def save_cora_sample(sample: CoraSample, path: str) -> None:
    """Save a graph with its split masks (a split file or a clean graph)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({k: getattr(sample, k) for k in _CORA_SAMPLE_FIELDS}, path)


def load_cora_sample(path: str) -> CoraSample:
    payload = torch.load(path, map_location="cpu")
    return CoraSample(**payload)


# ─────────────────────────────────────────────────────────────────────────────
# Clean graphs (M_retrain)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_nodes(sample: CoraSample, forget: dict) -> Tuple[CoraSample, str]:
    """Deleted nodes stay as isolated, zero-feature, non-train nodes (ids are kept)."""
    deleted_ids = [int(r["nid"]) for r in forget["records"]]
    bad = [nid for nid in deleted_ids if not bool(sample.train_mask[nid])]
    if bad:
        raise ValueError(f"forget set has {len(bad)} non-train node(s) in this split, e.g. {bad[:5]}: "
                         "it was drawn from a different split file")

    deleted = torch.zeros(sample.num_nodes, dtype=torch.bool)
    deleted[deleted_ids] = True
    x_clean = sample.x.clone()
    x_clean[deleted] = 0.0
    keep_edge = ~(deleted[sample.edge_index[0]] | deleted[sample.edge_index[1]])
    train_mask_clean = sample.train_mask.clone()
    train_mask_clean[deleted] = False

    clean = CoraSample(
        x=x_clean, edge_index=sample.edge_index[:, keep_edge], y=sample.y,
        train_mask=train_mask_clean, test_mask=sample.test_mask,
        num_nodes=sample.num_nodes, num_classes=sample.num_classes,
    )
    info = (f"deleted={len(deleted_ids)} nodes, train {int(sample.train_mask.sum())}->{int(train_mask_clean.sum())}, "
            f"edges {sample.num_edges}->{clean.num_edges}")
    return clean, info


def _clean_edges(sample: CoraSample, forget: dict) -> Tuple[CoraSample, str]:
    """Each forget edge is removed in both directions."""
    pairs = [(int(r["src"]), int(r["dst"])) for r in forget["records"]]
    src, dst = sample.edge_index[0].tolist(), sample.edge_index[1].tolist()
    existing = set(zip(src, dst))
    missing = [(s, d) for s, d in pairs if (s, d) not in existing and (d, s) not in existing]
    non_train = [(s, d) for s, d in pairs if not (bool(sample.train_mask[s]) and bool(sample.train_mask[d]))]
    if missing or non_train:
        raise ValueError(f"forget set has {len(missing)} edge(s) not in this graph (e.g. {missing[:5]}) and "
                         f"{len(non_train)} with a non-train endpoint (e.g. {non_train[:5]}): "
                         "it was drawn from a different split file")

    deleted_pairs = set(pairs) | {(d, s) for s, d in pairs}
    keep_edge = torch.tensor([(s, d) not in deleted_pairs for s, d in zip(src, dst)], dtype=torch.bool)
    clean = CoraSample(
        x=sample.x, edge_index=sample.edge_index[:, keep_edge], y=sample.y,
        train_mask=sample.train_mask, test_mask=sample.test_mask,
        num_nodes=sample.num_nodes, num_classes=sample.num_classes,
    )
    info = (f"deleted={len(pairs)} edges (removed {int((~keep_edge).sum())} directed entries), "
            f"edges {sample.num_edges}->{clean.num_edges}")
    return clean, info


def source_mismatch_warning(forget: dict, data_path: str) -> Optional[str]:
    """A warning when a forget set records a different source split file than `data_path`."""
    source = forget.get("source")
    if source and resolve_repo_path(source) != os.path.abspath(data_path):
        return (f"[WARN] forget set was drawn from {relpath_for_log(resolve_repo_path(source))}, "
                f"not from {relpath_for_log(data_path)}; using it anyway")
    return None


def make_clean(sample: CoraSample, forget: dict) -> Tuple[CoraSample, str]:
    """The clean graph of `sample` without the forget set (in memory) and a one-line summary.
    Raises ValueError when the forget set does not match this split."""
    if forget["forget_type"] == "node":
        return _clean_nodes(sample, forget)
    if forget["forget_type"] == "edge":
        return _clean_edges(sample, forget)
    raise ValueError(f"Unsupported forget_type: {forget['forget_type']}")


def build_clean(data_path: str, forget_set_path: str, out_root: str = DEFAULT_CLEAN_ROOT) -> str:
    """Apply a forget set to the graph and split in `data_path` (the source run's data);
    the clean graph is written to <out_root>/<forget set file name>.pt."""
    with open(forget_set_path) as f:
        forget = json.load(f)
    warning = source_mismatch_warning(forget, data_path)
    if warning:
        print(warning, flush=True)
    clean, info = make_clean(load_cora_sample(data_path), forget)

    name = os.path.splitext(os.path.basename(forget_set_path))[0]
    out_path = os.path.join(out_root, f"{name}.pt")
    save_cora_sample(clean, out_path)
    print(output_row(forget["forget_type"], f"{info} -> {relpath_for_log(out_path)}"), flush=True)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make the Cora split file and clean graphs.")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("split", help="make and save the stratified 80/20 split")
    sp.add_argument("--seed", type=int, default=None, help="split seed (default: random)")
    sp.add_argument("--cora_root", default=DEFAULT_CORA_ROOT, help="PyG Planetoid download/cache directory")
    sp.add_argument("--out", default="", help=f"output file (default: {DEFAULT_SPLIT_ROOT}/cora_seed<seed>.pt)")

    cp = sub.add_parser("clean", help="apply forget sets to a split file (clean graphs for M_retrain)")
    cp.add_argument("--data_path", required=True, help="the source run's split file (from the split command)")
    cp.add_argument("--forget_set", nargs="+", required=True, help="forget set JSON(s) drawn from that split")
    cp.add_argument("--out_root", default=DEFAULT_CLEAN_ROOT)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "split":
        seed = resolve_seed(args.seed)
        out = args.out or os.path.join(DEFAULT_SPLIT_ROOT, f"cora_seed{seed}.pt")
        sample = load_cora(args.cora_root, split_seed=seed)
        save_cora_sample(sample, out)
        print(f"{DATASET} split  seed {seed}" + (" (random)" if args.seed is None else ""), flush=True)
        print(output_row("done", f"nodes {sample.num_nodes}  train {int(sample.train_mask.sum())}"
                                 f"  test {int(sample.test_mask.sum())} -> {relpath_for_log(out)}"), flush=True)
    else:
        print(f"{DATASET} clean graphs  data {relpath_for_log(args.data_path)}", flush=True)
        for path in args.forget_set:
            build_clean(args.data_path, path, args.out_root)


if __name__ == "__main__":
    main()
