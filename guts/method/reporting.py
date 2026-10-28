"""Summary statistics and metric formatting shared by the dataset pipelines."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

# Column order of every metric table and log line.
METRIC_ORDER = (("mrr", "MRR"), ("acc@1", "ACC@1"), ("acc@5", "ACC@5"), ("acc@10", "ACC@10"))


def output_row(label: str, text: str, width: int = 9) -> str:
    """Indented output row with the label in a fixed-width column (aligned console summaries)."""
    return f"  {label:<{width}}{text}"


def format_metrics(metrics: Optional[Mapping], digits: int = 4, sep: str = " ") -> str:
    """'MRR 0.1234 ACC@1 0.0567 ACC@5 ... ACC@10 ...' (missing values as '-'), items joined by sep."""
    metrics = metrics or {}
    parts = []
    for key, label in METRIC_ORDER:
        value = metrics.get(key)
        parts.append(f"{label} {value:.{digits}f}" if isinstance(value, (int, float)) else f"{label} -")
    return sep.join(parts)


def mean_std(values: Iterable, digits: int = 6, extremes: bool = False) -> Optional[dict]:
    """Mean and sample standard deviation (ddof=1; 0 for a single value) of the non-None values.

    Returns None when there is no value. extremes=True adds min and max."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    out = {"mean": round(m, digits), "std": round(var ** 0.5, digits)}
    if extremes:
        out.update(min=min(values), max=max(values))
    out["n"] = n
    return out
