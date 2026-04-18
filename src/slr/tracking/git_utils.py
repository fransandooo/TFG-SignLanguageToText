"""Capture current git commit for experiment tracking."""

from __future__ import annotations

import subprocess


def get_git_commit() -> str:
    """Return the short hash of the current HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"
