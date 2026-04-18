"""Full evaluation pipeline: metrics + confusion matrix + per-class CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from slr.evaluation.confusion import (
    compute_confusion,
    plot_confusion_matrix,
    top_confusions,
    write_per_class_csv,
)
from slr.utils.io import save_json


def run_full_evaluation(
    preds: np.ndarray,
    labels: np.ndarray,
    top1_acc: float,
    top5_acc: float,
    label_names: list[str],
    output_dir: str | Path,
    model_id: str = "",
) -> dict[str, Any]:
    """Generate all evaluation artifacts and return a summary dict.

    Artifacts written to *output_dir*:
    - ``per_class_metrics.csv``
    - ``confusion_matrix_counts.png``
    - ``confusion_matrix_normalized_true.png``
    - ``top_confusions.json``
    - ``results.json``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(label_names)

    counts, normalized = compute_confusion(preds, labels, num_classes)

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0) * 100)
    bal_acc = float(balanced_accuracy_score(labels, preds) * 100)

    write_per_class_csv(output_dir / "per_class_metrics.csv", labels, preds, label_names)

    plot_confusion_matrix(
        counts, label_names,
        output_dir / "confusion_matrix_counts.png",
        title=f"{model_id} - confusion matrix counts" if model_id else "Confusion Matrix (counts)",
        normalized=False,
    )
    plot_confusion_matrix(
        normalized, label_names,
        output_dir / "confusion_matrix_normalized_true.png",
        title=f"{model_id} - row-normalized confusion matrix" if model_id else "Confusion Matrix (normalized)",
        normalized=True,
    )

    top_conf = top_confusions(counts, normalized, label_names, limit=20)
    save_json(top_conf, output_dir / "top_confusions.json")

    summary = {
        "top1_acc": round(top1_acc, 2),
        "top5_acc": round(top5_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "num_samples": int(len(labels)),
        "num_classes": num_classes,
    }
    save_json(summary, output_dir / "results.json")
    return summary
