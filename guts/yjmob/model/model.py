"""GraphSAGE + GRU model for next-grid prediction.

Each input cell is embedded by an ID embedding and fed to a GRU. The GraphSAGE
embedding z of the cell (from the shard's rolling graph) is added to the GRU
output (head fusion: h = h_gru + beta * z). Logits over all grid cells are the
inner products with the ID embedding table (tied weights), so the GRU hidden
size must equal the ID embedding size.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
    """GraphSAGE-style layer with (weighted) mean neighbor aggregation."""

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

        if chunk_size > 0 and src.size(0) > chunk_size:
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
    """Stack of WeightedSAGEConv layers.

    Edge weights are used only when ``weighted`` is set; otherwise any given
    weights are ignored and neighbors are averaged uniformly. Training and
    evaluation both go through this check.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        weighted: bool = False,
        edge_weight_transform: str = "log1p",
        edge_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(0, num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(WeightedSAGEConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.ReLU()
        self.weighted = bool(weighted)
        self.edge_weight_transform = str(edge_weight_transform)
        self.edge_chunk_size = int(edge_chunk_size)

    def _prepare_edge_weight(self, edge_weight: torch.Tensor | None) -> torch.Tensor | None:
        if not self.weighted or edge_weight is None:
            return None
        edge_weight = edge_weight.to(dtype=torch.float32)
        if self.edge_weight_transform == "log1p":
            edge_weight = torch.log1p(edge_weight)
        return edge_weight

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        edge_weight = self._prepare_edge_weight(edge_weight)
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index, edge_weight=edge_weight, chunk_size=self.edge_chunk_size)
            if i + 1 < len(self.layers):
                h = self.act(h)
                h = self.dropout(h)
        return h


class NextGridPredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        id_embedding_dim: int = 128,
        sage_hidden_dim: int = 128,
        sage_out_dim: int = 128,
        sage_layers: int = 2,
        sage_dropout: float = 0.1,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        gru_dropout: float = 0.0,
        sage_weighted: bool = False,
        sage_edge_weight_transform: str = "log1p",
        sage_edge_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if id_embedding_dim != gru_hidden_dim:
            raise ValueError(
                f"tied output weights require id_embedding_dim({id_embedding_dim}) == gru_hidden_dim({gru_hidden_dim})."
            )
        if sage_out_dim != gru_hidden_dim:
            raise ValueError(
                f"head fusion requires sage_out_dim({sage_out_dim}) == gru_hidden_dim({gru_hidden_dim})."
            )
        self.num_nodes = int(num_nodes)
        self.id_embedding = nn.Embedding(num_nodes, id_embedding_dim)
        self.graphsage = GraphSAGEEncoder(
            in_dim=node_feat_dim,
            hidden_dim=sage_hidden_dim,
            out_dim=sage_out_dim,
            num_layers=sage_layers,
            dropout=sage_dropout,
            weighted=bool(sage_weighted),
            edge_weight_transform=sage_edge_weight_transform,
            edge_chunk_size=int(sage_edge_chunk_size),
        )
        self.sage_beta = nn.Parameter(torch.tensor(0.1))
        self.gru = nn.GRU(
            input_size=id_embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Scores over all grid cells (tied to the ID embedding table)."""
        return h @ self.id_embedding.weight.T

    def encode_graph(
        self,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.graphsage(graph_x, edge_index, edge_weight=edge_weight)

    def graph_embeddings_for_inputs(
        self,
        z: torch.Tensor,
        input_cells: torch.Tensor,
        node_lookup: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z of each input cell; zero for cells absent from the graph (node_lookup == -1)."""
        if node_lookup is None:
            return z[input_cells]
        local_idx = node_lookup[input_cells]
        valid = local_idx >= 0
        out = z.new_zeros((*input_cells.shape, z.size(-1)))
        if valid.any():
            out[valid] = z[local_idx[valid]]
        return out

    def sequence_features(
        self,
        input_cells: torch.Tensor,
        lengths: torch.Tensor,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        node_lookup: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused hidden state at every step, [B, T, gru_hidden_dim]."""
        z = self.encode_graph(graph_x, edge_index, edge_weight=edge_weight)
        emb = self.id_embedding(input_cells)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=input_cells.size(1))
        return out + self.sage_beta * self.graph_embeddings_for_inputs(z, input_cells, node_lookup=node_lookup)
