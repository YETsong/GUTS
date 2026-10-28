"""Aggregated evaluation for static graphs, shared by the pipeline (STEP 4) and unlearning."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from guts.static_graph.metrics import compute_metrics


def evaluate_aggregated(shard_models: dict, sample, device: str) -> dict:
    """Average the softmax of every shard model on the full graph (uniform weights) and
    score the test nodes (accuracy, macro F1)."""
    weights = {sid: 1.0 / max(len(shard_models), 1) for sid in shard_models}
    x_full = sample.x.to(device)
    edge_index_full = sample.edge_index.to(device)
    prob_sum = torch.zeros(sample.num_nodes, sample.num_classes, device=device)
    for shard_id, model in shard_models.items():
        model.eval()
        with torch.no_grad():
            logits = model(x_full, edge_index_full)
            prob_sum += weights[shard_id] * F.softmax(logits, dim=-1)

    test_idx = sample.test_mask.nonzero(as_tuple=True)[0]
    if shard_models and test_idx.numel() > 0:
        metrics = compute_metrics(prob_sum[test_idx].argmax(dim=-1).cpu(), sample.y[test_idx])
    else:
        metrics = {"test_acc": 0.0, "test_f1_macro": 0.0}
    return {**metrics, "test_total": int(test_idx.numel()), "weights": weights}
