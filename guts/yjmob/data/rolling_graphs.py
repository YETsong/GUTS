"""Shard rolling-window graphs: the graph a shard model uses for target day d.

The rolling graph of target day d merges the shard's day graphs of the previous
window_size days (d - window_size .. d - 1; day d itself is not included):
  - edges: union of the window's transitions, weights (counts) summed;
  - visit_count, in_count, out_count: summed over the window;
  - unique_in, unique_out: recounted on the merged edges;
  - dwell_time: mean dwell per visit over the window (total dwell / total visits),
    the same unit as the day graphs;
  - active_nodes: union over the window.
Features are stored as log1p, like the day graphs, with one row of x per active node
(in the order of active_nodes, sorted global cell ids); edge_index uses global ids.
Files go to
shard_<k>/rolling_graphs_w<window_size>/target_day_XXXX.pt and are listed in the shard's
next_location_meta.pt under "rolling_graphs".
"""

from __future__ import annotations

import os
from typing import List

import torch

from guts.utils import ensure_dir

FEATURE_NAMES = ["visit_count", "in_count", "out_count", "unique_in", "unique_out", "dwell_time"]
_VISIT_COL = 0
_SUM_COLS = [0, 1, 2]  # visit_count, in_count, out_count
_UNIQUE_IN_COL, _UNIQUE_OUT_COL = 3, 4
_DWELL_COL = 5


def _empty_day_graph() -> dict:
    return {
        "x": torch.zeros((0, len(FEATURE_NAMES)), dtype=torch.float32),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "edge_weight": torch.empty((0,), dtype=torch.float32),
        "active_nodes": torch.empty((0,), dtype=torch.long),
    }


def _load_day_graph(shard_dir: str, day: int) -> dict:
    """A shard day graph, or an empty one when the day has no file (e.g. before day 0)."""
    path = os.path.join(shard_dir, "daily_graphs", f"day_{day:04d}.pt")
    if not os.path.exists(path):
        return _empty_day_graph()
    g = torch.load(path, map_location="cpu")
    x = g["x"]
    active_nodes = g["active_nodes"].to(torch.long)
    assert x.shape[0] == active_nodes.numel(), (
        f"{path}: x.shape[0]={x.shape[0]} != len(active_nodes)={active_nodes.numel()} "
        "(shard day graphs keep one row per active node)"
    )
    return {
        "x": x.to(torch.float32),
        "edge_index": g["edge_index"].to(torch.long),
        "edge_weight": g.get("edge_weight", torch.ones(g["edge_index"].size(1), dtype=torch.float32)).to(torch.float32),
        "active_nodes": active_nodes,
    }


def build_shard_rolling_graph(
    shard_dir: str,
    target_day: int,
    num_nodes: int,
    window_size: int = 3,
) -> dict:
    """Merge the shard's day graphs of the window before target_day (see module docstring)."""
    window_days = list(range(target_day - window_size, target_day))
    graphs = [_load_day_graph(shard_dir, d) for d in window_days]

    active_list = [g["active_nodes"] for g in graphs if g["active_nodes"].numel() > 0]
    active_nodes = torch.unique(torch.cat(active_list)) if active_list else torch.empty((0,), dtype=torch.long)
    num_active = active_nodes.numel()

    raw_sum = torch.zeros((num_active, len(_SUM_COLS)), dtype=torch.float32)
    dwell_total = torch.zeros(num_active, dtype=torch.float32)
    for g in graphs:
        rows = torch.searchsorted(active_nodes, g["active_nodes"])
        raw = torch.expm1(g["x"]).clamp_min_(0)
        raw_sum.index_add_(0, rows, raw[:, _SUM_COLS])
        dwell_total.index_add_(0, rows, raw[:, _DWELL_COL] * raw[:, _VISIT_COL])
    visits = raw_sum[:, _SUM_COLS.index(_VISIT_COL)]
    dwell_mean = torch.where(visits > 0, dwell_total / visits.clamp_min(1e-12), torch.zeros_like(visits))

    x_merged = torch.zeros((num_active, len(FEATURE_NAMES)), dtype=torch.float32)
    x_merged[:, _SUM_COLS] = torch.log1p(raw_sum)
    x_merged[:, _DWELL_COL] = torch.log1p(dwell_mean)

    edge_index_list = [g["edge_index"] for g in graphs if g["edge_index"].numel() > 0]
    edge_weight_list = [g["edge_weight"] for g in graphs if g["edge_index"].numel() > 0]
    if edge_index_list:
        all_ei = torch.cat(edge_index_list, dim=1)
        all_ew = torch.cat(edge_weight_list, dim=0)
        keep = all_ei[0] != all_ei[1]
        all_ei = all_ei[:, keep]
        all_ew = all_ew[keep]
        flat = all_ei[0] * num_nodes + all_ei[1]
        uniq, inv = torch.unique(flat, return_inverse=True)
        merged_ew = torch.zeros(uniq.numel(), dtype=torch.float32)
        merged_ew.scatter_add_(0, inv, all_ew)
        merged_src = torch.div(uniq, num_nodes, rounding_mode="floor")
        merged_dst = uniq % num_nodes
        merged_ei = torch.stack([merged_src, merged_dst], dim=0)
    else:
        merged_ei = torch.empty((2, 0), dtype=torch.long)
        merged_ew = torch.empty((0,), dtype=torch.float32)

    # Edge endpoints are shard nodes of their day, hence in active_nodes.
    local_ei = torch.searchsorted(active_nodes, merged_ei)
    x_merged[:, _UNIQUE_IN_COL] = torch.log1p(torch.bincount(local_ei[1], minlength=num_active).float())
    x_merged[:, _UNIQUE_OUT_COL] = torch.log1p(torch.bincount(local_ei[0], minlength=num_active).float())

    return {
        "target_day": int(target_day),
        "window_days": window_days,
        "window_size": int(window_size),
        "num_global_nodes": int(num_nodes),
        "x": x_merged,
        "edge_index": merged_ei.to(torch.int32),
        "edge_weight": merged_ew,
        "active_nodes": active_nodes.to(torch.int32),
        "feature_names": FEATURE_NAMES,
    }


def write_shard_rolling_graphs(
    shard_dir: str,
    target_days: List[int],
    window_size: int,
    num_nodes: int,
) -> List[dict]:
    """Write the rolling graphs of target_days; returns their index entries."""
    out_dir = ensure_dir(os.path.join(shard_dir, f"rolling_graphs_w{window_size}"))
    entries = []
    for d in target_days:
        graph = build_shard_rolling_graph(shard_dir, d, num_nodes, window_size)
        path = os.path.join(out_dir, f"target_day_{d:04d}.pt")
        torch.save(graph, path)
        entries.append(
            {
                "target_day": int(d),
                "window_days": graph["window_days"],
                "window_size": int(window_size),
                "path": os.path.relpath(path, start=shard_dir),
                "num_edges": int(graph["edge_index"].size(1)),
                "num_active_nodes": int(graph["active_nodes"].numel()),
            }
        )
    return entries


def update_shard_meta_with_rolling_graphs(shard_dir: str, window_size: int, entries: List[dict]) -> None:
    """Record the rolling graph entries in the shard's next_location_meta.pt."""
    meta_path = os.path.join(shard_dir, "next_location_meta.pt")
    meta = torch.load(meta_path, map_location="cpu")
    rolling_graphs_by_w = dict(meta.get("rolling_graphs", {}))
    rolling_graphs_by_w[int(window_size)] = entries
    meta["rolling_graphs"] = rolling_graphs_by_w
    torch.save(meta, meta_path)
