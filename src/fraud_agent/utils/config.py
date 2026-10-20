from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return data


def ensure_directories(config: dict[str, Any]) -> None:
    """Create configured artifact directories when missing."""
    for key in ("processed_dir", "models_dir", "metrics_dir", "figures_dir", "tables_dir"):
        value = config.get("paths", {}).get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)

