#!/usr/bin/env python3
"""Generate confusion-matrix artifacts for final WLASL landmark models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, rnn_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(rnn_output).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), rnn_output).squeeze(1)
        return context, weights


class PoseClassifierAttention(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        num_classes = config["num_classes"]
        dropout = config["dropout"]
        bidirectional = config.get("bidirectional", True)

        self.num_directions = 2 if bidirectional else 1
        rnn_hidden = hidden_size * self.num_directions

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.attention = TemporalAttention(rnn_hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        rnn_out, _ = self.gru(x)
        context, _ = self.attention(rnn_out)
        return self.classifier(context)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 65, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoseTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        input_size = config["input_size"]
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_transformer_layers", 4)
        num_classes = config["num_classes"]
        dropout = config["dropout"]
        max_len = config.get("max_frames", 64)

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x[:, 0])


@dataclass(frozen=True)
class LandmarkSample:
    path: Path
    gloss: str
    label: int


class LandmarkSplitDataset(Dataset):
    def __init__(self, split_dir: Path, gloss_to_idx: dict[str, int], max_frames: int, input_size: int):
        self.split_dir = split_dir
        self.gloss_to_idx = gloss_to_idx
        self.max_frames = max_frames
        self.input_size = input_size
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(f"No .npy samples found in {split_dir}")

    def _collect_samples(self) -> list[LandmarkSample]:
        samples: list[LandmarkSample] = []
        for gloss in sorted(self.gloss_to_idx):
            class_dir = self.split_dir / gloss
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.npy")):
                samples.append(LandmarkSample(path=path, gloss=gloss, label=self.gloss_to_idx[gloss]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        landmarks = np.load(sample.path).astype(np.float32)
        landmarks = pad_or_truncate(landmarks, self.max_frames)
        sequence = landmarks.reshape(self.max_frames, self.input_size)
        return torch.from_numpy(sequence), sample.label, str(sample.path)


def pad_or_truncate(landmarks: np.ndarray, max_frames: int) -> np.ndarray:
    if landmarks.shape[0] > max_frames:
        indices = np.linspace(0, landmarks.shape[0] - 1, max_frames, dtype=int)
        return landmarks[indices]
    if landmarks.shape[0] < max_frames:
        padding_shape = (max_frames - landmarks.shape[0],) + landmarks.shape[1:]
        padding = np.zeros(padding_shape, dtype=landmarks.dtype)
        return np.concatenate([landmarks, padding], axis=0)
    return landmarks


def load_metadata(metadata_path: Path) -> tuple[list[str], dict[str, int]]:
    metadata = json.loads(metadata_path.read_text())
    gloss_to_idx = metadata.get("gloss_to_idx")
    if not gloss_to_idx:
        glosses = metadata.get("glosses")
        if not glosses:
            raise ValueError(f"No gloss mapping found in {metadata_path}")
        gloss_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}

    normalized = {str(gloss): int(idx) for gloss, idx in gloss_to_idx.items()}
    labels = [gloss for gloss, _ in sorted(normalized.items(), key=lambda item: item[1])]
    return labels, normalized


def build_model(architecture: str, config: dict[str, Any]) -> nn.Module:
    if architecture == "gru_attention":
        return PoseClassifierAttention(config)
    if architecture == "transformer":
        return PoseTransformer(config)
    raise ValueError(f"Unsupported model architecture: {architecture}")


def load_checkpoint_model(checkpoint_path: Path, architecture: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not config:
        raise ValueError(f"Checkpoint has no config: {checkpoint_path}")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_model(architecture, config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def evaluate(
    model: nn.Module,
    dataset: LandmarkSplitDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for sequences, labels, paths in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences)
            probabilities = F.softmax(logits, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            topk = torch.topk(probabilities, k=min(5, probabilities.shape[1]), dim=1)

            top1_correct += int((preds == labels).sum().item())
            top5_correct += int((topk.indices == labels.unsqueeze(1)).any(dim=1).sum().item())

            for row_idx in range(labels.shape[0]):
                y_true.append(int(labels[row_idx].item()))
                y_pred.append(int(preds[row_idx].item()))
                prediction_rows.append(
                    {
                        "path": paths[row_idx],
                        "true_idx": int(labels[row_idx].item()),
                        "pred_idx": int(preds[row_idx].item()),
                        "confidence": float(probabilities[row_idx, preds[row_idx]].item()),
                        "top5_indices": [int(value) for value in topk.indices[row_idx].cpu().tolist()],
                        "top5_confidences": [float(value) for value in topk.values[row_idx].cpu().tolist()],
                    }
                )

    total = len(dataset)
    return (
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.int64),
        prediction_rows,
        100.0 * top1_correct / total,
        100.0 * top5_correct / total,
    )


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums != 0)


def write_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str], float_format: str | None = None) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/pred_label", *labels])
        for label, row in zip(labels, matrix):
            if float_format:
                writer.writerow([label, *[float_format.format(value) for value in row]])
            else:
                writer.writerow([label, *[int(value) for value in row]])


def write_predictions_csv(path: Path, rows: list[dict[str, Any]], labels: list[str]) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = [
            "path",
            "true_label",
            "pred_label",
            "true_idx",
            "pred_idx",
            "confidence",
            "top5_labels",
            "top5_confidences",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            top5_labels = [labels[index] for index in row["top5_indices"]]
            writer.writerow(
                {
                    "path": row["path"],
                    "true_label": labels[row["true_idx"]],
                    "pred_label": labels[row["pred_idx"]],
                    "true_idx": row["true_idx"],
                    "pred_idx": row["pred_idx"],
                    "confidence": f"{row['confidence']:.6f}",
                    "top5_labels": "|".join(top5_labels),
                    "top5_confidences": "|".join(f"{value:.6f}" for value in row["top5_confidences"]),
                }
            )


def write_per_class_metrics(path: Path, counts: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    with path.open("w", newline="") as handle:
        fieldnames = [
            "class_idx",
            "label",
            "support",
            "correct",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, label in enumerate(labels):
            correct = int(counts[idx, idx])
            class_support = int(support[idx])
            accuracy = correct / class_support if class_support else 0.0
            writer.writerow(
                {
                    "class_idx": idx,
                    "label": label,
                    "support": class_support,
                    "correct": correct,
                    "accuracy": f"{accuracy:.6f}",
                    "precision": f"{precision[idx]:.6f}",
                    "recall": f"{recall[idx]:.6f}",
                    "f1": f"{f1[idx]:.6f}",
                }
            )


def collect_top_confusions(counts: np.ndarray, normalized: np.ndarray, labels: list[str], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for true_idx, true_label in enumerate(labels):
        support = int(counts[true_idx].sum())
        for pred_idx, pred_label in enumerate(labels):
            if true_idx == pred_idx:
                continue
            count = int(counts[true_idx, pred_idx])
            if count == 0:
                continue
            rows.append(
                {
                    "true_idx": true_idx,
                    "true_label": true_label,
                    "pred_idx": pred_idx,
                    "pred_label": pred_label,
                    "count": count,
                    "true_support": support,
                    "normalized_true": float(normalized[true_idx, pred_idx]),
                }
            )
    rows.sort(key=lambda item: (item["count"], item["normalized_true"], item["true_support"]), reverse=True)
    return rows[:limit]


def write_top_confusions_json(
    path: Path,
    top_confusions: list[dict[str, Any]],
    counts: np.ndarray,
    labels: list[str],
    args: argparse.Namespace,
    top1: float,
    top5: float,
) -> None:
    support = {label: int(counts[idx].sum()) for idx, label in enumerate(labels)}
    payload = {
        "model_id": args.model_id,
        "model_architecture": args.model_architecture,
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "dataset_split": str(args.dataset_split),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top1_acc": top1,
        "top5_acc": top5,
        "num_samples": int(counts.sum()),
        "labels": labels,
        "per_class_support": support,
        "note": "Rows are true labels; columns are predicted labels. normalized_true rows sum to 1 when support > 0.",
        "top_confusions": top_confusions,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def plot_matrix(path: Path, matrix: np.ndarray, labels: list[str], title: str, normalized: bool) -> None:
    fig_size = max(12, len(labels) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
                value = matrix[i, j]
                if value == 0:
                    continue
                text = f"{value:.2f}" if normalized else str(int(value))
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-architecture", choices=["gru_attention", "transformer"], required=True)
    parser.add_argument("--model-id", default="unknown_model")
    parser.add_argument("--artifact-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-confusions", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    labels, gloss_to_idx = load_metadata(args.metadata)
    model, config = load_checkpoint_model(args.checkpoint, args.model_architecture, device)

    max_frames = int(config.get("max_frames", 64))
    input_size = int(config.get("input_size", 63))
    dataset = LandmarkSplitDataset(args.dataset_split, gloss_to_idx, max_frames=max_frames, input_size=input_size)
    y_true, y_pred, prediction_rows, top1, top5 = evaluate(model, dataset, device=device, batch_size=args.batch_size)

    all_label_indices = list(range(len(labels)))
    counts = confusion_matrix(y_true, y_pred, labels=all_label_indices)
    normalized_true = row_normalize(counts)
    top_confusions = collect_top_confusions(counts, normalized_true, labels, limit=args.top_confusions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.artifact_prefix
    write_matrix_csv(args.output_dir / f"{prefix}confusion_matrix_counts.csv", counts, labels)
    write_matrix_csv(
        args.output_dir / f"{prefix}confusion_matrix_normalized_true.csv",
        normalized_true,
        labels,
        float_format="{:.6f}",
    )
    write_per_class_metrics(args.output_dir / f"{prefix}per_class_metrics.csv", counts, y_true, y_pred, labels)
    write_predictions_csv(args.output_dir / f"{prefix}predictions.csv", prediction_rows, labels)
    write_top_confusions_json(
        args.output_dir / f"{prefix}top_confusions.json",
        top_confusions,
        counts,
        labels,
        args,
        top1,
        top5,
    )

    title_prefix = args.model_id.replace("_", " ")
    plot_matrix(
        args.output_dir / f"{prefix}confusion_matrix_counts.png",
        counts,
        labels,
        f"{title_prefix} - confusion matrix counts",
        normalized=False,
    )
    plot_matrix(
        args.output_dir / f"{prefix}confusion_matrix_normalized_true.png",
        normalized_true,
        labels,
        f"{title_prefix} - row-normalized confusion matrix",
        normalized=True,
    )

    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "num_samples": len(dataset),
                "top1_acc": top1,
                "top5_acc": top5,
                "output_dir": str(args.output_dir),
                "artifact_prefix": prefix,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
