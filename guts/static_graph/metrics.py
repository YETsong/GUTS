from __future__ import annotations

import torch
from sklearn.metrics import f1_score


def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict:
    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    acc = float((pred == true).float().mean()) if pred.numel() else 0.0
    f1_macro = float(f1_score(true_np, pred_np, average="macro", zero_division=0)) if pred.numel() else 0.0
    return {"test_acc": acc, "test_f1_macro": f1_macro}
