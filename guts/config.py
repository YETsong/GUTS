"""
Configuration loading and validation for shard-based experiments.

Wraps a YAML config file into typed dataclass containers (hierarchical YAML for
reproducible multi-step experiments).

Usage:
    cfg = load_config("config/default.yaml")
    cfg = load_config("config/default.yaml", overrides={"partition.num_shards": 4})
    cfg = resolve_output_dirs(cfg)   # creates all output subdirectories
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass, field
from dataclasses import MISSING
from typing import Any, Dict, Optional

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs (one dataclass per YAML section)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PartitionConfig:
    """Controls the selected partitioning strategy and its parameters."""
    num_shards: int = 8
    strategy: str = "guts"
    epsilon: float = 0.05

    # deletion influence
    di_alpha: float = 0.0
    di_beta: float = 0.0
    di_gamma: float = 0.0
    di_alpha_static: float = 0.0
    include_isolated_nodes: bool = False
    # weighted hyperedge
    use_di: bool = True
    # boundary refinement
    use_boundary_refinement: bool = True
    refine_epsilon: float = 0.05
    refine_max_passes: int = 5
    refine_candidate_max_ratio: float = 0.10
    lambda_di: float = 0.0
    lambda_c: float = 0.5
    lambda_l: float = 0.0
    refine_class_balance_passes: bool = False
    use_temporal_subgraph_assignment: bool = True


@dataclass
class TrainingConfig:
    """
    Per-shard training hyperparameters.

    guts/yjmob/pipeline.py materializes sharded
    next-grid processed dirs and trains guts/yjmob/model/train_single.py per shard.
    """
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 2026
    batch_size: int = 512
    id_embedding_dim: int = 128
    sage_hidden_dim: int = 128
    sage_out_dim: int = 128
    sage_layers: int = 2
    sage_dropout: float = 0.1
    gru_hidden_dim: int = 128
    gru_layers: int = 1
    gru_dropout: float = 0.0
    sage_num_neighbors: list = field(default_factory=lambda: [10, 10])
    early_stop_warmup: int = 20
    early_stop_patience: int = 15


@dataclass
class DataConfig:
    nextgrid_processed_dir: str = "processed/yjmob_1_14d"


@dataclass
class TemporalGraphConfig:
    window_size: int = 3


@dataclass
class OutputConfig:
    """
    Output directory structure.
    Resolved layout:
        {root}/{run_name}/
            logs/
    """
    root: str = ""
    run_name: str = ""


@dataclass
class ShardConfig:
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    temporal_graph: TemporalGraphConfig = field(default_factory=TemporalGraphConfig)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

# Map YAML section name → dataclass type
_SUB_CLS_MAP = {
    "partition":      PartitionConfig,
    "training":       TrainingConfig,
    "data":           DataConfig,
    "output":         OutputConfig,
    "temporal_graph": TemporalGraphConfig,
}


def load_config(
    yaml_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> ShardConfig:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Build each sub-config, silently ignoring unknown YAML keys
    sub_cfgs: Dict[str, Any] = {}
    for section, cls in _SUB_CLS_MAP.items():
        section_dict = raw.get(section, {}) or {}
        field_map = {fd.name: fd for fd in dataclasses.fields(cls)}
        filtered = {
            k: _coerce_value(v, field_map[k])
            for k, v in section_dict.items()
            if k in field_map
        }
        sub_cfgs[section] = cls(**filtered)

    cfg = ShardConfig(**sub_cfgs)

    # Apply dot-notation overrides: "partition.num_shards" → cfg.partition.num_shards = val
    if overrides:
        for dotkey, val in overrides.items():
            parts = dotkey.split(".", 1)
            if len(parts) == 2:
                sub_name, field_name = parts
                sub_cfg = getattr(cfg, sub_name, None)
                if sub_cfg is not None and hasattr(sub_cfg, field_name):
                    field_def = {fd.name: fd for fd in dataclasses.fields(type(sub_cfg))}[field_name]
                    setattr(sub_cfg, field_name, _coerce_value(val, field_def))
            else:
                if hasattr(cfg, dotkey):
                    field_def = {fd.name: fd for fd in dataclasses.fields(type(cfg))}[dotkey]
                    setattr(cfg, dotkey, _coerce_value(val, field_def))

    return cfg


def _coerce_value(value: Any, field_def: dataclasses.Field) -> Any:
    if not isinstance(value, str):
        return value

    default = field_def.default
    if default is MISSING:
        return value

    try:
        if isinstance(default, bool):
            lowered = value.lower()
            if lowered in ("true", "false"):
                return lowered == "true"
        if isinstance(default, int) and not isinstance(default, bool):
            return int(value)
        if isinstance(default, float):
            return float(value)
    except ValueError:
        return value
    return value


def config_to_dict(cfg: ShardConfig) -> Dict[str, Any]:
    """
    Serialize a ShardConfig to a nested dict for logging or checkpointing.

    Returns the same structure as the YAML file so that the dict can be
    written back with yaml.dump() for experiment reproducibility.
    """
    return dataclasses.asdict(cfg)


def align_retrain_seed(cfg: ShardConfig, source_dir: str, overrides: Dict[str, Any]) -> Optional[str]:
    """Use the forget-set seed N as training.seed when source_dir ends with `_seed{N}`.

    Raises ValueError if training.seed is overridden with a different value.
    Returns a log message when the seed is changed, otherwise None."""
    import re  # noqa: PLC0415

    m = re.search(r"_seed(\d+)$", os.path.basename(os.path.normpath(source_dir or "")))
    if not m:
        return None
    forget_seed = int(m.group(1))
    if "training.seed" in overrides:
        if int(overrides["training.seed"]) != forget_seed:
            raise ValueError(f"training.seed={overrides['training.seed']} != input data seed {forget_seed} "
                             f"({source_dir}): a retrain must use the forget-set seed as its training seed")
        return None
    cfg.training.seed = forget_seed
    return f"training.seed -> {forget_seed} (matched to input data {os.path.basename(os.path.normpath(source_dir))})"


def resolve_output_dirs(cfg: ShardConfig) -> ShardConfig:
    """
    Create all output subdirectories and fill in auto-generated run_name.

    Directory layout created:
        {root}/{run_name}/
            logs/

    Uses timestamp_id() for run_name auto-generation when
    cfg.output.run_name is empty.

    Args:
        cfg: ShardConfig (mutated in-place for run_name, then returned).

    Returns:
        The same cfg with run_name filled in and directories created.
    """
    _REPO_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from guts.utils import timestamp_id, ensure_dir  # noqa: PLC0415

    if not cfg.output.run_name:
        cfg.output.run_name = timestamp_id()
    if not cfg.output.root:
        cfg.output.root = os.path.join(_REPO_ROOT, "Results", "shard-runs", "yjmob")

    base = os.path.join(cfg.output.root, cfg.output.run_name)
    ensure_dir(os.path.join(base, "logs"))

    return cfg
