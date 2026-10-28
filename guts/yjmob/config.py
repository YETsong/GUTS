"""YJMob configuration: the shared GUTS sections plus the yjmob training/data sections.

Usage:
    cfg = load_yjmob_config("configs/yjmob_14d.yaml", overrides={"partition.num_shards": 4})
    cfg = resolve_output_dirs(cfg)   # creates all output subdirectories
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from guts.method.config import GUTSConfig, load_config
from guts.method.utils import ensure_dir, timestamp_id

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


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
    seed: Optional[int] = None   # None: a random seed drawn at run start and saved to config_used.yaml
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
class YJMobConfig(GUTSConfig):
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    temporal_graph: TemporalGraphConfig = field(default_factory=TemporalGraphConfig)


def load_yjmob_config(yaml_path: str, overrides: Optional[Dict[str, Any]] = None) -> YJMobConfig:
    return load_config(yaml_path, overrides=overrides, cfg_cls=YJMobConfig)


def resolve_output_dirs(cfg: YJMobConfig) -> YJMobConfig:
    """
    Create the run directory {root}/{run_name}/ and fill in auto-generated run_name.

    Uses timestamp_id() for run_name auto-generation when
    cfg.output.run_name is empty.

    Args:
        cfg: YJMobConfig (mutated in-place for run_name, then returned).

    Returns:
        The same cfg with run_name filled in and directories created.
    """
    if not cfg.output.run_name:
        cfg.output.run_name = timestamp_id()
    if not cfg.output.root:
        cfg.output.root = os.path.join(_REPO_ROOT, "Results", "shard-runs", "yjmob")

    base = os.path.join(cfg.output.root, cfg.output.run_name)
    ensure_dir(base)

    return cfg
