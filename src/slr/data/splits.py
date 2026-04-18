"""Metadata and split loading for WLASL / MS-ASL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_gloss_mapping(metadata_path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Read a metadata JSON and return (gloss_to_idx, idx_to_gloss).

    Supports two JSON layouts:
    - ``{"gloss_to_idx": {"word": 0, ...}}``  (directory-based metadata)
    - ``{"glosses": ["word", ...]}``           (older format)
    """
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())

    gloss_to_idx = metadata.get("gloss_to_idx")
    if not gloss_to_idx:
        glosses = metadata.get("glosses")
        if glosses:
            gloss_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}
        else:
            raise ValueError(f"No gloss mapping found in {metadata_path}")

    gloss_to_idx = {str(k): int(v) for k, v in gloss_to_idx.items()}
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    return gloss_to_idx, idx_to_gloss


def load_subset_meta(subset_path: str | Path) -> dict[str, Any]:
    """Load a subset JSON (produced by notebook 16) and return its contents."""
    with open(subset_path) as f:
        return json.load(f)
