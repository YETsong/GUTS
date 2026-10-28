"""Static-graph configuration: the shared GUTS sections plus the run seed.

Usage:
    cfg = load_static_config("configs/cora.yaml", overrides={"training.seed": 42})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from guts.method.config import GUTSConfig, load_config


@dataclass
class TrainingConfig:
    # One seed drives the train/test split, the KaHyPar partition and model training.
    # None: a random seed drawn at run start and saved to config_used.yaml / args.json.
    seed: Optional[int] = None


@dataclass
class StaticGraphConfig(GUTSConfig):
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_static_config(yaml_path: str, overrides: Optional[Dict[str, Any]] = None) -> StaticGraphConfig:
    return load_config(yaml_path, overrides=overrides, cfg_cls=StaticGraphConfig)
