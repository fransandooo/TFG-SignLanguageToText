"""Tests for experiment manifest generation."""

import json
import tempfile
from pathlib import Path

from slr.tracking.manifest import generate_manifest


class TestGenerateManifest:
    def test_creates_json(self):
        config = {
            "dataset": {"name": "test", "max_frames": 64},
            "model": {"name": "gru_attention", "num_classes": 10},
            "training": {"seed": 42},
        }
        training_result = {
            "best_val_top1": 65.0,
            "best_epoch": 50,
            "total_epochs": 80,
            "training_time_s": 120.0,
        }
        eval_summary = {
            "top1_acc": 60.0,
            "top5_acc": 85.0,
            "macro_f1": 55.0,
            "balanced_accuracy": 58.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest(
                config=config,
                training_result=training_result,
                eval_summary=eval_summary,
                output_dir=tmpdir,
                experiment_id="test_exp",
            )
            path = Path(tmpdir) / "experiment_manifest.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["experiment_id"] == "test_exp"
            assert data["top1"] == 60.0
            assert data["macro_f1"] == 55.0
            assert "git_commit" in data
            assert data["hardware"]["torch_version"]

    def test_without_eval(self):
        config = {
            "dataset": {"name": "test", "max_frames": 32},
            "model": {"name": "gru_attention", "num_classes": 5},
            "training": {"seed": 0},
        }
        training_result = {
            "best_val_top1": 70.0,
            "best_epoch": 20,
            "total_epochs": 40,
            "training_time_s": 60.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest(
                config=config,
                training_result=training_result,
                eval_summary=None,
                output_dir=tmpdir,
            )
            assert manifest["top1"] == 70.0
            assert manifest["macro_f1"] is None
