"""Automatic experiment manifest generation."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slr.tracking.git_utils import get_git_commit
from slr.utils.io import save_json


def generate_manifest(
    config: dict[str, Any],
    training_result: dict[str, Any],
    eval_summary: dict[str, Any] | None,
    output_dir: str | Path,
    experiment_id: str = "",
    title: str = "",
    research_question: str = "",
) -> dict[str, Any]:
    """Build and save an experiment manifest JSON.

    Captures git commit, hardware info, timing, config, and metrics.
    """
    output_dir = Path(output_dir)
    now = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    manifest: dict[str, Any] = {
        "experiment_id": experiment_id or output_dir.name,
        "title": title,
        "date": now,
        "owner": "Francisco Sandoval",
        "git_commit": git_commit,
        "research_question": research_question,
        "dataset": config.get("dataset", {}).get("name", ""),
        "dataset_version_or_path": config.get("dataset", {}).get("path", ""),
        "split_strategy": config.get("dataset", {}).get("split_strategy", "train/val/test"),
        "num_classes": config.get("model", {}).get("num_classes", config.get("num_classes", 0)),
        "num_samples_train": config.get("dataset", {}).get("num_train"),
        "num_samples_val": config.get("dataset", {}).get("num_val"),
        "num_samples_test": config.get("dataset", {}).get("num_test"),
        "input_representation": "MediaPipe hand landmarks",
        "preprocessing": [
            "wrist-centered normalization",
            "wrist-to-middle-finger scale",
            "linear interpolation for missing frames",
            f"pad/truncate to {config.get('dataset', {}).get('max_frames', 64)} frames",
        ],
        "model_architecture": config.get("model", {}).get("name", ""),
        "model_config": config.get("model", {}),
        "training_config": config.get("training", {}),
        "seed": config.get("training", {}).get("seed"),
        "hardware": {
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "runtime": {
            "training_time_s": training_result.get("training_time_s"),
            "total_epochs": training_result.get("total_epochs"),
            "best_epoch": training_result.get("best_epoch"),
        },
        "max_frames": config.get("dataset", {}).get("max_frames"),
        "num_hands": 1,
        "landmark_features": 63,
        "normalization_strategy": "wrist-centered, wrist-to-middle-finger scale",
        "detection_rate_mean": None,
        "detection_rate_by_split": {"train": None, "val": None, "test": None},
        "interpolation_strategy": "linear interpolation with edge fill",
    }

    if eval_summary:
        manifest["top1"] = eval_summary.get("top1_acc")
        manifest["top5"] = eval_summary.get("top5_acc")
        manifest["macro_f1"] = eval_summary.get("macro_f1")
        manifest["balanced_accuracy"] = eval_summary.get("balanced_accuracy")
    else:
        manifest["top1"] = training_result.get("best_val_top1")
        manifest["top5"] = None
        manifest["macro_f1"] = None
        manifest["balanced_accuracy"] = None

    manifest["metrics"] = {
        "val_top1": training_result.get("best_val_top1"),
        **(eval_summary or {}),
    }
    manifest["artifacts"] = {
        "checkpoint": str(output_dir / "best_model.pt"),
        "training_curves": str(output_dir / "training_curves.png"),
        "training_history": str(output_dir / "training_history.json"),
        "confusion_matrix": str(output_dir / "confusion_matrix_normalized_true.png"),
        "per_class_metrics": str(output_dir / "per_class_metrics.csv"),
        "manifest": str(output_dir / "experiment_manifest.json"),
    }
    manifest["confusion_matrix_path"] = str(output_dir / "confusion_matrix_normalized_true.png")
    manifest["per_class_accuracy_path"] = str(output_dir / "per_class_metrics.csv")
    manifest["main_findings"] = []
    manifest["limitations"] = []
    manifest["next_actions"] = []

    save_json(manifest, output_dir / "experiment_manifest.json")
    return manifest
