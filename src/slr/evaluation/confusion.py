"""Confusion matrix generation and visualization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, normalized_true) confusion matrices."""
    all_labels = list(range(num_classes))
    counts = confusion_matrix(y_true, y_pred, labels=all_labels)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts.astype(np.float64),
        row_sums,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_sums != 0,
    )
    return counts, normalized


def top_confusions(
    counts: np.ndarray,
    normalized: np.ndarray,
    labels: list[str],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the *limit* most frequent off-diagonal confusions."""
    rows: list[dict[str, Any]] = []
    for true_idx, true_label in enumerate(labels):
        support = int(counts[true_idx].sum())
        for pred_idx, pred_label in enumerate(labels):
            if true_idx == pred_idx:
                continue
            count = int(counts[true_idx, pred_idx])
            if count == 0:
                continue
            rows.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "count": count,
                "support": support,
                "normalized_true": float(normalized[true_idx, pred_idx]),
            })
    rows.sort(key=lambda r: (r["count"], r["normalized_true"]), reverse=True)
    return rows[:limit]


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
    normalized: bool = False,
) -> None:
    """Save a confusion-matrix heatmap to *output_path*."""
    fig_size = max(12, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(
        matrix, interpolation="nearest", cmap="Blues",
        vmin=0, vmax=1 if normalized else None,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if len(labels) <= 35:
        threshold = (matrix.max() if matrix.size else 0) * 0.55
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if val == 0:
                    continue
                text = f"{val:.2f}" if normalized else str(int(val))
                ax.text(
                    j, i, text, ha="center", va="center",
                    color="white" if val > threshold else "black",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_per_class_csv(
    path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> None:
    """Write per-class precision/recall/F1 to a CSV file."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0,
    )
    counts_cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class_idx", "label", "support", "correct",
            "accuracy", "precision", "recall", "f1",
        ])
        writer.writeheader()
        for idx, label in enumerate(labels):
            s = int(support[idx])
            correct = int(counts_cm[idx, idx])
            writer.writerow({
                "class_idx": idx,
                "label": label,
                "support": s,
                "correct": correct,
                "accuracy": f"{correct / s:.6f}" if s else "0.000000",
                "precision": f"{precision[idx]:.6f}",
                "recall": f"{recall[idx]:.6f}",
                "f1": f"{f1[idx]:.6f}",
            })
