"""Metrics for next-grid prediction."""

from __future__ import annotations

from typing import Dict, Iterable

import torch


@torch.no_grad()
def topk_correct(logits: torch.Tensor, target: torch.Tensor, ks: Iterable[int] = (1, 5, 10)) -> Dict[int, int]:
    if logits.numel() == 0:
        return {int(k): 0 for k in ks}
    # A row with non-finite (NaN/inf) logits means the model diverged for that
    # sample — force it to count as incorrect instead of letting torch.topk's
    # NaN ordering accidentally match the target.
    finite_row = torch.isfinite(logits).all(dim=1)
    max_k = min(max(int(k) for k in ks), int(logits.size(1)))
    _, pred = torch.topk(logits, k=max_k, dim=1)
    correct = pred.eq(target.view(-1, 1)) & finite_row.view(-1, 1)
    out: Dict[int, int] = {}
    for k in ks:
        kk = min(int(k), int(logits.size(1)))
        out[int(k)] = int(correct[:, :kk].any(dim=1).sum().item())
    return out


@torch.no_grad()
def reciprocal_rank_sum(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    num_classes = logits.size(1)
    target_scores = logits.gather(1, target.view(-1, 1))
    ranks = (logits > target_scores).sum(dim=1).float() + 1.0
    # NaN comparisons are always False, so a fully-diverged (all-NaN) row would
    # otherwise get rank=1 (a spurious perfect score) instead of being treated
    # as a total miss. Force those rows to the worst possible rank.
    finite_row = torch.isfinite(logits).all(dim=1)
    ranks = torch.where(finite_row, ranks, torch.full_like(ranks, float(num_classes)))
    return float((1.0 / ranks).sum().item())


def finalize_metrics(total: int, loss_sum: float, correct: Dict[int, int], rr_sum: float) -> Dict[str, float]:
    denom = max(int(total), 1)
    return {
        "loss": float(loss_sum) / denom,
        "acc@1": float(correct.get(1, 0)) / denom,
        "acc@5": float(correct.get(5, 0)) / denom,
        "acc@10": float(correct.get(10, 0)) / denom,
        "mrr": float(rr_sum) / denom,
        "support": int(total),
    }

