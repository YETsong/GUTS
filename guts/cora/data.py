from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_CORA_ROOT = os.path.join(_REPO_ROOT, "processed", "Cora")


@dataclass
class CoraSample:
    day: int
    x: torch.Tensor              # [N, 1433] bag-of-words feature
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor      # [N] bool
    num_nodes: int
    num_classes: int

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def in_dim(self) -> int:
        return int(self.x.size(1))



def _stratified_80_20_split(y: torch.Tensor, num_nodes: int, split_seed: int, train_ratio: float = 0.8):
    import numpy as np

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


def load_cora(root: str = DEFAULT_CORA_ROOT, split_seed: int = 2026) -> CoraSample:
    from torch_geometric.datasets import Planetoid

    os.makedirs(root, exist_ok=True)
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    num_nodes = int(data.num_nodes)
    y = data.y.long()

    train_mask, test_mask = _stratified_80_20_split(y, num_nodes, split_seed)

    return CoraSample(
        day=0,
        x=data.x.float(),
        edge_index=data.edge_index.long(),
        edge_weight=torch.ones(data.edge_index.size(1), dtype=torch.float32),
        y=y,
        train_mask=train_mask,
        val_mask=torch.zeros(num_nodes, dtype=torch.bool),
        test_mask=test_mask,
        num_nodes=num_nodes,
        num_classes=int(dataset.num_classes),
    )


def induce_train_subgraph(sample: CoraSample) -> Tuple[CoraSample, torch.Tensor]:
    from torch_geometric.utils import subgraph

    train_idx = sample.train_mask.nonzero(as_tuple=True)[0]
    edge_index_new, _, edge_mask = subgraph(
        train_idx, sample.edge_index, relabel_nodes=True, return_edge_mask=True
    )
    n_train = int(train_idx.numel())

    train_sample = CoraSample(
        day=sample.day,
        x=sample.x[train_idx],
        edge_index=edge_index_new,
        edge_weight=sample.edge_weight[edge_mask],
        y=sample.y[train_idx],
        train_mask=torch.ones(n_train, dtype=torch.bool),
        val_mask=torch.zeros(n_train, dtype=torch.bool),
        test_mask=torch.zeros(n_train, dtype=torch.bool),
        num_nodes=n_train,
        num_classes=sample.num_classes,
    )
    return train_sample, train_idx


_CORA_SAMPLE_FIELDS = [
    "day", "x", "edge_index", "edge_weight", "y",
    "train_mask", "val_mask", "test_mask", "num_nodes", "num_classes",
]


def save_cora_clean(sample: CoraSample, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: getattr(sample, k) for k in _CORA_SAMPLE_FIELDS}, path)


def load_cora_clean(path: str) -> CoraSample:
    payload = torch.load(path, map_location="cpu")
    return CoraSample(**payload)
