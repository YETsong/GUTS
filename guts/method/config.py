"""
Configuration loading shared by every dataset.

Wraps a YAML config file into typed dataclass containers, one per YAML section.
This module holds the GUTS sections (partition, output); dataset-specific
sections live with the dataset (guts/yjmob/config.py).

Usage:
    cfg = load_config("configs/cora.yaml")
    cfg = load_config("configs/cora.yaml", overrides={"partition.num_shards": 4})
"""

from __future__ import annotations

import dataclasses
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
    refine_epsilon: float = 1.0
    refine_max_passes: int = 5
    refine_candidate_max_ratio: float = 0.10
    lambda_di: float = 0.0
    lambda_c: float = 0.5
    lambda_l: float = 0.0
    refine_class_balance_passes: bool = False
    use_temporal_subgraph_assignment: bool = True


@dataclass
class OutputConfig:
    """
    Output directory: {root}/{run_name}/ (logs are files in it, e.g. run_log.txt).
    """
    root: str = ""
    run_name: str = ""


@dataclass
class GUTSConfig:
    """Sections shared by every dataset; dataset configs subclass it with their own sections."""
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_config(
    yaml_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    cfg_cls: type = GUTSConfig,
):
    """Load a YAML file into `cfg_cls`, one dataclass per section field.

    Unknown sections and keys are ignored. `overrides` maps "section.key" to a value."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    sub_cfgs: Dict[str, Any] = {}
    for section_field in dataclasses.fields(cfg_cls):
        cls = type(section_field.default_factory())
        section_dict = raw.get(section_field.name, {}) or {}
        field_map = {fd.name: fd for fd in dataclasses.fields(cls)}
        filtered = {
            k: _coerce_value(v, field_map[k])
            for k, v in section_dict.items()
            if k in field_map
        }
        sub_cfgs[section_field.name] = cls(**filtered)

    cfg = cfg_cls(**sub_cfgs)

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

    if cfg.partition.strategy != "guts":
        raise ValueError(f"Unknown partition.strategy: {cfg.partition.strategy}")
    return cfg


def parse_overrides(pairs) -> Dict[str, str]:
    """["section.key=value", ...] -> {"section.key": "value"}; load_config converts each value
    to its field's type."""
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _coerce_value(value: Any, field_def: dataclasses.Field) -> Any:
    if not isinstance(value, str):
        return value

    default = field_def.default
    if default is MISSING:
        return value
    if default is None:
        # Optional[int] fields (seed): "null"/"none" -> None, digits -> int.
        if value.lower() in ("null", "none"):
            return None
        try:
            return int(value)
        except ValueError:
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


def config_to_dict(cfg) -> Dict[str, Any]:
    """
    Serialize a config to a nested dict for logging or checkpointing.

    Returns the same structure as the YAML file so that the dict can be
    written back with yaml.dump() for experiment reproducibility.
    """
    return dataclasses.asdict(cfg)
