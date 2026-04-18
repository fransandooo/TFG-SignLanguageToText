"""Tests for evaluation metrics."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from slr.evaluation.metrics import evaluate, per_class_accuracy


class TestEvaluate:
    def test_perfect_predictions(self):
        num_classes = 5
        n = 20
        x = torch.randn(n, 10, 63)
        y = torch.arange(n) % num_classes
        # Encode true labels into x so the model can recover them
        for i in range(n):
            x[i, 0, 0] = float(y[i])
        model = _PeekLabelModel(num_classes=num_classes)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = evaluate(model, loader, device="cpu")
        assert metrics["top1_acc"] == 100.0
        assert metrics["top5_acc"] == 100.0
        assert len(metrics["preds"]) == n

    def test_random_predictions(self):
        model = _DummyModel(num_classes=5, correct=False)
        x = torch.randn(20, 10, 63)
        y = torch.arange(20) % 5
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        metrics = evaluate(model, loader, device="cpu")
        assert 0 <= metrics["top1_acc"] <= 100.0
        assert 0 <= metrics["top5_acc"] <= 100.0


class TestPerClassAccuracy:
    def test_basic(self):
        preds = np.array([0, 0, 1, 1, 2])
        labels = np.array([0, 1, 1, 1, 2])
        idx_to_gloss = {0: "a", 1: "b", 2: "c"}
        result = per_class_accuracy(preds, labels, idx_to_gloss)
        assert result["a"]["accuracy"] == 100.0
        assert result["b"]["correct"] == 2
        assert result["c"]["accuracy"] == 100.0


class _PeekLabelModel(nn.Module):
    """Reads the true label from x[i, 0, 0] and returns one-hot logits."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.num_classes)
        labels = x[:, 0, 0].long()
        for i in range(batch_size):
            logits[i, labels[i]] = 10.0
        return logits


class _DummyModel(nn.Module):
    def __init__(self, num_classes: int, correct: bool):
        super().__init__()
        self.num_classes = num_classes
        self.correct = correct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if self.correct:
            labels = torch.arange(batch_size) % self.num_classes
            logits = torch.zeros(batch_size, self.num_classes)
            for i, label in enumerate(labels):
                logits[i, label] = 10.0
            return logits
        return torch.randn(batch_size, self.num_classes)
