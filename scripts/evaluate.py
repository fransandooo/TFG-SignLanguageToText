#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a dataset split.

Usage:
    python scripts/evaluate.py \
        --checkpoint 05_experiments/final_assault/best_model_attention.pt \
        --metadata 06_datasets/wlasl/landmarks/wlasl30/metadata.json \
        --dataset-split 06_datasets/wlasl/landmarks/wlasl30/test \
        --output-dir 05_experiments/final_assault/eval_test \
        --architecture gru_attention
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from slr.data.splits import load_gloss_mapping
from slr.data.transforms import pad_or_truncate
from slr.evaluation.evaluator import run_full_evaluation
from slr.evaluation.metrics import evaluate
from slr.models.registry import build_model, ensure_registry_populated


class SplitDataset(Dataset):
    """Simple dataset that reads .npy files from a single split directory."""

    def __init__(self, split_dir: Path, gloss_to_idx: dict[str, int],
                 max_frames: int, input_size: int):
        self.max_frames = max_frames
        self.input_size = input_size
        self.samples: list[tuple[Path, int]] = []

        for gloss in sorted(gloss_to_idx):
            class_dir = split_dir / gloss
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.npy")):
                self.samples.append((path, gloss_to_idx[gloss]))

        if not self.samples:
            raise ValueError(f"No .npy files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        lm = np.load(path).astype(np.float32)
        lm = pad_or_truncate(lm, self.max_frames)
        x = torch.from_numpy(lm.reshape(self.max_frames, self.input_size))
        return x, label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--dataset-split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--architecture", default="gru_attention",
                   choices=["gru_attention", "transformer", "gru_baseline"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-id", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ensure_registry_populated()
    gloss_to_idx, idx_to_gloss = load_gloss_mapping(args.metadata)
    label_names = [idx_to_gloss[i] for i in range(len(idx_to_gloss))]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    max_frames = int(config.get("max_frames", 64))
    input_size = int(config.get("input_size", 63))

    model = build_model(
        args.architecture,
        input_size=input_size,
        num_classes=config["num_classes"],
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.4),
        bidirectional=config.get("bidirectional", True),
        max_frames=max_frames,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = SplitDataset(args.dataset_split, gloss_to_idx, max_frames, input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    metrics = evaluate(model, loader, device=device)
    print(f"Samples: {len(dataset)}")
    print(f"Top-1: {metrics['top1_acc']:.1f}%")
    print(f"Top-5: {metrics['top5_acc']:.1f}%")

    summary = run_full_evaluation(
        preds=metrics["preds"],
        labels=metrics["labels"],
        top1_acc=metrics["top1_acc"],
        top5_acc=metrics["top5_acc"],
        label_names=label_names,
        output_dir=args.output_dir,
        model_id=args.model_id or args.architecture,
    )

    print(f"Macro-F1: {summary['macro_f1']:.1f}%")
    print(f"Balanced Accuracy: {summary['balanced_accuracy']:.1f}%")
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
