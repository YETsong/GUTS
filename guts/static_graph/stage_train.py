"""Shard training for static graphs, shared by the pipeline (STEP 3) and unlearning (retraining)."""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from guts.static_graph.model import build_model


def train_shard(model_type: str, hp: dict, epochs: int, x, edge_index, y, train_mask,
                num_classes: int, device: str):
    """Full-batch training for `epochs`; the last epoch's model is kept (no validation).

    Returns (model, timing); model is None when the shard has no train node."""
    n_train = int(train_mask.sum())
    if n_train == 0:
        return None, {"train_nodes": 0, "train_ms": None, "epoch0_train_ms": None}

    model = build_model(model_type, int(x.size(1)), num_classes, hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)

    epoch0_train_ms = 0.0
    train_ms_acc = 0.0
    for epoch in range(epochs):
        t0_epoch = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        dt = (time.perf_counter() - t0_epoch) * 1000
        if epoch == 0:
            epoch0_train_ms += dt
        else:
            train_ms_acc += dt

    return model, {"train_nodes": n_train, "train_ms": round(train_ms_acc), "epoch0_train_ms": round(epoch0_train_ms)}


def train_seconds(summary: dict) -> float:
    """Training time of a train_shard summary in seconds (first epoch included)."""
    return (summary["epoch0_train_ms"] + summary["train_ms"]) / 1000


def run_train_stage(shards: dict, train_sample, model_type: str, hp: dict, epochs: int, device: str,
                    ckpt_dir: str):
    """Train one model per shard on its subgraph of the train graph and save
    <ckpt_dir>/shard_<id>.pt. Returns (models, per-shard summaries)."""
    shard_models: dict = {}
    shard_summaries: dict = {}
    for shard_id in sorted(shards.keys()):
        data = shards[shard_id]
        model, summary = train_shard(
            model_type, hp, epochs, data.x, data.edge_index, train_sample.y[data.uids],
            train_sample.train_mask[data.uids], train_sample.num_classes, device,
        )
        shard_summaries[shard_id] = summary
        if model is None:
            continue  # no train node, no model
        shard_models[shard_id] = model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"shard_{shard_id}.pt"))
    return shard_models, shard_summaries
