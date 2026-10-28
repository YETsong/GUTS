"""Node classifiers for static graphs: GraphSAGE, GCN, GAT and GIN, each with two message-passing layers."""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


# ─────────────────────────────────────────────────────────────────────────────
# Model-level defaults
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DEFAULTS = {
    "graphsage": dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.5),
    "gcn":       dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.5),
    "gat":       dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.6,
                      heads1=8, heads2=1, attn_dropout=0.6, negative_slope=0.2),
    "gin":       dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.5),
}


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGEClassifier(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim, normalize=True),
            SAGEConv(hidden_dim, hidden_dim, normalize=True),
        ])
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.dropout(self.act(self.layers[0](x, edge_index)))
        h = self.layers[1](h, edge_index)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# GCN
# ─────────────────────────────────────────────────────────────────────────────

class GCNClassifier(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.dropout(self.act(self.convs[0](x, edge_index)))
        h = self.convs[1](h, edge_index)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# GAT
# ─────────────────────────────────────────────────────────────────────────────

class GATClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        heads1: int,
        heads2: int,
        hidden_per_head: int,
        dropout: float,
        attn_dropout: float,
        negative_slope: float,
    ) -> None:
        super().__init__()
        self.gat1 = GATConv(
            in_dim, hidden_per_head, heads=heads1, concat=True,
            dropout=attn_dropout, negative_slope=negative_slope,
        )
        self.gat2 = GATConv(
            hidden_per_head * heads1, num_classes, heads=heads2, concat=False,
            dropout=attn_dropout, negative_slope=negative_slope,
        )
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.dropout(self.act(self.gat1(x, edge_index)))
        return self.gat2(h, edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# GIN
# ─────────────────────────────────────────────────────────────────────────────

class GINClassifier(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        def mlp(d_in: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.convs = nn.ModuleList([GINConv(mlp(in_dim), train_eps=True), GINConv(mlp(hidden_dim), train_eps=True)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.dropout(self.act(self.bns[0](self.convs[0](x, edge_index))))
        h = self.bns[1](self.convs[1](h, edge_index))
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def resolve_model_hp(model_type: str, **overrides) -> dict:
    """MODEL_DEFAULTS[model_type] with the given (non-None) values replaced."""
    hp = dict(MODEL_DEFAULTS[model_type])
    for k, v in overrides.items():
        if v is not None:
            hp[k] = v
    return hp


def build_model(model_type: str, in_dim: int, num_classes: int, hp: dict) -> nn.Module:
    """A classifier from resolved hyperparameters (resolve_model_hp, or a run's args.json)."""
    if model_type == "graphsage":
        return GraphSAGEClassifier(in_dim, num_classes, hidden_dim=hp["hidden_dim"], dropout=hp["dropout"])
    if model_type == "gcn":
        return GCNClassifier(in_dim, num_classes, hidden_dim=hp["hidden_dim"], dropout=hp["dropout"])
    if model_type == "gat":
        return GATClassifier(in_dim, num_classes, heads1=hp["heads1"], heads2=hp["heads2"],
                             hidden_per_head=hp["hidden_dim"] // hp["heads1"],
                             dropout=hp["dropout"], attn_dropout=hp["attn_dropout"],
                             negative_slope=hp["negative_slope"])
    if model_type == "gin":
        return GINClassifier(in_dim, num_classes, hidden_dim=hp["hidden_dim"], dropout=hp["dropout"])
    raise ValueError(f"Unknown model_type: {model_type!r} (one of graphsage/gcn/gat/gin)")
