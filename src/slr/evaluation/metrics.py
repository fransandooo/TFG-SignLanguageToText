"""Evaluation metrics for landmark-sequence classifiers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    top_k: Sequence[int] = (1, 5),
) -> dict:
    """Run inference on *loader* and return a metrics dict.

    Returns
    -------
    dict with keys:
        top{k}_acc  – percentage accuracy for each k in *top_k*
        preds       – int array of predicted labels
        labels      – int array of ground-truth labels
    """
    model.eval()
    device = torch.device(device)

    total = 0
    topk_correct: dict[int, int] = {k: 0 for k in top_k}
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        # Support both (x, y) and (x, y, extra...) tuples
        x = batch[0].to(device)
        y = batch[1].to(device)
        logits = model(x)

        total += y.size(0)
        _, pred_topk = logits.topk(max(top_k), dim=1, largest=True, sorted=True)

        for k in top_k:
            hit = pred_topk[:, :k].eq(y.view(-1, 1).expand_as(pred_topk[:, :k]))
            topk_correct[k] += hit.any(dim=1).sum().item()

        all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    metrics: dict = {}
    for k in top_k:
        metrics[f"top{k}_acc"] = 100.0 * topk_correct[k] / total if total else 0.0
    metrics["preds"] = np.array(all_preds, dtype=np.int64)
    metrics["labels"] = np.array(all_labels, dtype=np.int64)
    return metrics


def per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    idx_to_gloss: dict[int, str],
) -> dict[str, dict]:
    """Compute per-class accuracy from flat arrays.

    Returns a dict keyed by gloss name::

        {"drink": {"accuracy": 66.7, "correct": 10, "support": 15}, ...}
    """
    result: dict[str, dict] = {}
    for cls_idx, gloss in sorted(idx_to_gloss.items()):
        mask = labels == cls_idx
        support = int(mask.sum())
        if support == 0:
            continue
        correct = int((preds[mask] == cls_idx).sum())
        result[gloss] = {
            "accuracy": 100.0 * correct / support,
            "correct": correct,
            "support": support,
        }
    return result
