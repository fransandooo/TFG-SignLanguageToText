"""I/O helpers for config loading and result saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, returning a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; all other values are replaced.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and merge with its ``base_config`` if present."""
    config = load_yaml(path)
    base_path = config.pop("base_config", None)
    if base_path is not None:
        base = load_yaml(Path(path).parent / base_path)
        config = merge_configs(base, config)
    return config


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write *data* as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")
