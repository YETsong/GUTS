from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv
except ImportError as exc:  # pragma: no cover - dependency is required at runtime
    SAGEConv = None
    _SAGE_IMPORT_ERROR = exc


class _ChunkedWeightedAgg(torch.autograd.Function):
    """Weighted mean neighbor aggregation with O(chunk_size × D) peak memory.

    Forward:  neigh[dst] += h[src] * w  then divide by per-node weight sum.
    Backward: grad_h[src] += (grad_neigh[dst] / denom[dst]) * w

    Only src, dst, weight, and denom (all 1-D) are saved for backward —
    the large [E, D] message tensor is never materialised in full.
    """

    @staticmethod
    def forward(ctx, h, src, dst, weight, chunk_size):
        num_nodes, D = h.size()
        neigh = h.new_zeros(num_nodes, D)
        denom = h.new_zeros(num_nodes)
        for s in range(0, src.size(0), chunk_size):
            e = min(s + chunk_size, src.size(0))
            sc, dc, wc = src[s:e], dst[s:e], weight[s:e]
            msg = h.detach().index_select(0, sc) * wc.view(-1, 1)
            neigh.index_add_(0, dc, msg)
            denom.index_add_(0, dc, wc)
            del msg
        neigh.div_(denom.clamp_min_(1e-12).view(-1, 1))
        ctx.save_for_backward(src, dst, weight, denom)
        ctx.chunk_size = chunk_size
        return neigh

    @staticmethod
    def backward(ctx, grad_neigh):
        src, dst, weight, denom = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        # grad_h[src] += (grad_neigh[dst] / denom[dst]) * w
        grad_neigh_norm = grad_neigh / denom.view(-1, 1)  # [N, D]
        grad_h = grad_neigh.new_zeros(grad_neigh.shape)
        for s in range(0, src.size(0), chunk_size):
            e = min(s + chunk_size, src.size(0))
            sc, dc, wc = src[s:e], dst[s:e], weight[s:e]
            g = grad_neigh_norm.index_select(0, dc) * wc.view(-1, 1)
            grad_h.index_add_(0, sc, g)
            del g
        return grad_h, None, None, None, None


class WeightedSAGEConv(nn.Module):
    """GraphSAGE-style layer with weighted mean neighbor aggregation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None,
                chunk_size: int = 0) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.lin_self(x)

        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            weight = x.new_ones((src.numel(),))
        else:
            weight = edge_weight.to(dtype=x.dtype)

        num_edges = src.size(0)

        if chunk_size > 0 and num_edges > chunk_size:
            # Custom autograd function: saves only 1-D index/weight tensors,
            # never materialises the full [E, D] message tensor.
            neigh = _ChunkedWeightedAgg.apply(x, src, dst, weight, chunk_size)
        else:
            neigh = x.new_zeros(x.shape)
            denom = x.new_zeros((x.size(0),))
            msg = x.index_select(0, src) * weight.view(-1, 1)
            neigh.index_add_(0, dst, msg)
            denom.index_add_(0, dst, weight)
            neigh = neigh / denom.clamp_min(1e-12).view(-1, 1)

        return self.lin_self(x) + self.lin_neigh(neigh)


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        weighted: bool = False,
        edge_direction: str = "original",
        edge_weight_transform: str = "log1p",
        edge_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphSAGEEncoder") from _SAGE_IMPORT_ERROR
        layers = []
        dims = [in_dim] + [hidden_dim] * max(0, num_layers - 1) + [out_dim]
        # When chunked aggregation is requested, always use WeightedSAGEConv
        # (which we control) regardless of the weighted flag.
        use_weighted_conv = bool(weighted) or (edge_chunk_size > 0)
        for i in range(num_layers):
            if use_weighted_conv:
                layers.append(WeightedSAGEConv(dims[i], dims[i + 1]))
            else:
                layers.append(SAGEConv(dims[i], dims[i + 1], normalize=True))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.ReLU()
        self.weighted = bool(weighted)
        self.use_weighted_conv = use_weighted_conv
        self.edge_direction = str(edge_direction)
        self.edge_weight_transform = str(edge_weight_transform)
        self.edge_chunk_size = int(edge_chunk_size)

    def _prepare_edges(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.edge_direction == "reverse" and edge_index.numel() > 0:
            edge_index = edge_index.flip(0)
        if edge_weight is not None:
            edge_weight = edge_weight.to(dtype=torch.float32)
            if self.edge_weight_transform == "log1p":
                edge_weight = torch.log1p(edge_weight)
        return edge_index, edge_weight

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        edge_index, edge_weight = self._prepare_edges(edge_index, edge_weight)
        for i, conv in enumerate(self.layers):
            if self.use_weighted_conv:
                h = conv(h, edge_index, edge_weight=edge_weight,
                         chunk_size=self.edge_chunk_size)
            else:
                h = conv(h, edge_index)
            if i + 1 < len(self.layers):
                h = self.act(h)
                h = self.dropout(h)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Model-level defaults
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DEFAULTS = {
    "graphsage": dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.5, num_layers=2),
    "gcn":       dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.5, num_layers=2),
    "gat":       dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.6, num_layers=2,
                       heads1=8, heads2=1, attn_dropout=0.6, negative_slope=0.2),
    "gin":       dict(hidden_dim=128, lr=0.01, weight_decay=5e-4, dropout=0.5, num_layers=2),
}


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGEClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, weighted=False,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        return self.classifier(z)


# ─────────────────────────────────────────────────────────────────────────────
# GCN
# ─────────────────────────────────────────────────────────────────────────────

class GCNClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i == 0:
                h = self.act(h)
                h = self.dropout(h)
            elif i + 1 < len(self.convs):
                h = self.act(h)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# GAT
# ─────────────────────────────────────────────────────────────────────────────

class GATClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        heads1: int = 8,
        heads2: int = 1,
        hidden_per_head: int = 8,
        dropout: float = 0.6,
        attn_dropout: float = 0.6,
        negative_slope: float = 0.2,
        **_ignore,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

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

    def forward(self, x, edge_index, edge_weight=None):
        h = self.gat1(x, edge_index)
        h = self.act(h)
        h = self.dropout(h)
        return self.gat2(h, edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# GIN
# ─────────────────────────────────────────────────────────────────────────────

class GINClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GINConv

        dims = [in_dim] + [hidden_dim] * num_layers
        convs, bns = [], []
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            convs.append(GINConv(mlp, train_eps=True))
            bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index)
            h = bn(h)
            if i == 0:
                h = self.act(h)
                h = self.dropout(h)
            elif i + 1 < len(self.convs):
                h = self.act(h)
        return self.classifier(h)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def resolve_model_hp(model_type: str, **overrides) -> dict:
    hp = dict(MODEL_DEFAULTS[model_type])
    for k, v in overrides.items():
        if v is not None:
            hp[k] = v
    return hp


def build_cora_model(model_type: str, in_dim: int, num_classes: int, **overrides) -> nn.Module:
    hp = resolve_model_hp(model_type, **overrides)
    if model_type == "graphsage":
        return GraphSAGEClassifier(in_dim, num_classes, hidden_dim=hp["hidden_dim"],
                                    num_layers=hp["num_layers"], dropout=hp["dropout"])
    if model_type == "gcn":
        return GCNClassifier(in_dim, num_classes, hidden_dim=hp["hidden_dim"],
                              num_layers=hp["num_layers"], dropout=hp["dropout"])
    if model_type == "gat":
        return GATClassifier(in_dim, num_classes, heads1=hp["heads1"], heads2=hp["heads2"],
                              hidden_per_head=hp["hidden_dim"] // hp["heads1"],
                              dropout=hp["dropout"], attn_dropout=hp["attn_dropout"],
                              negative_slope=hp["negative_slope"])
    if model_type == "gin":
        return GINClassifier(in_dim, num_classes, hidden_dim=hp["hidden_dim"],
                              num_layers=hp["num_layers"], dropout=hp["dropout"])
    raise ValueError(f"Unknown model_type: {model_type!r} (one of graphsage/gcn/gat/gin)")


CoraNodeClassifier = GraphSAGEClassifier
