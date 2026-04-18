#!/usr/bin/env python3
"""Train a landmark-sequence classifier from a YAML config.

Usage:
    python scripts/train.py --config configs/experiments/wlasl30_baseline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from slr.data.dataset import DirectoryDataset, SubsetDataset
from slr.data.splits import load_gloss_mapping, load_subset_meta
from slr.data.transforms import SkeletalAugmentation
from slr.evaluation.evaluator import run_full_evaluation
from slr.evaluation.metrics import evaluate
from slr.models.registry import build_model, ensure_registry_populated
from slr.tracking.manifest import generate_manifest
from slr.training.losses import build_criterion
from slr.training.trainer import Trainer
from slr.utils.io import resolve_config
from slr.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, or auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_config(args.config)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Output directory
    output_dir = Path(config.get("output_dir", "05_experiments/unnamed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Dataset
    ds_cfg = config["dataset"]
    max_frames = ds_cfg.get("max_frames", 64)
    input_size = ds_cfg.get("input_size", 63)
    aug = SkeletalAugmentation()

    if "subset_json" in ds_cfg:
        meta = load_subset_meta(ds_cfg["subset_json"])
        num_classes = meta["num_classes"]
        landmarks_root = ds_cfg["landmarks_root"]
        train_ds = SubsetDataset(meta["samples"]["train"], landmarks_root,
                                 max_frames, input_size, augmentation=aug)
        val_ds = SubsetDataset(meta["samples"]["val"], landmarks_root,
                               max_frames, input_size)
        test_ds = SubsetDataset(meta["samples"]["test"], landmarks_root,
                                max_frames, input_size)
        gloss_to_idx = {g: int(i) for g, i in meta["gloss_to_idx"].items()}
        idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    else:
        metadata_path = ds_cfg["metadata"]
        root_dir = ds_cfg["root"]
        gloss_to_idx, idx_to_gloss = load_gloss_mapping(metadata_path)
        num_classes = len(gloss_to_idx)
        train_ds = DirectoryDataset(root_dir, gloss_to_idx, max_frames,
                                     input_size, splits=("train",),
                                     augmentation=aug)
        val_ds = DirectoryDataset(root_dir, gloss_to_idx, max_frames,
                                   input_size, splits=("val",))
        test_ds = DirectoryDataset(root_dir, gloss_to_idx, max_frames,
                                    input_size, splits=("test",))

    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"Classes: {num_classes}")

    # Model
    ensure_registry_populated()
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "gru_attention")
    model = build_model(
        model_name,
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=model_cfg.get("hidden_size", 256),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.4),
        bidirectional=model_cfg.get("bidirectional", True),
        max_frames=max_frames,
        **{k: v for k, v in model_cfg.items()
           if k not in ("name", "hidden_size", "num_layers", "dropout",
                         "bidirectional", "input_size", "num_classes", "max_frames")},
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params:,} params)")

    # Flatten config for checkpoint compatibility
    flat_config = {
        "max_frames": max_frames,
        "input_size": input_size,
        "num_classes": num_classes,
        "hidden_size": model_cfg.get("hidden_size", 256),
        "num_layers": model_cfg.get("num_layers", 2),
        "dropout": model_cfg.get("dropout", 0.4),
        "bidirectional": model_cfg.get("bidirectional", True),
        **config.get("training", {}),
    }

    # Optimizer and scheduler
    train_cfg = config.get("training", {})
    optimizer = AdamW(model.parameters(),
                      lr=train_cfg.get("lr", 1e-3),
                      weight_decay=train_cfg.get("weight_decay", 5e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.get("epochs", 150))
    criterion = build_criterion(train_cfg.get("label_smoothing", 0.1))

    # Override config for trainer (needs flat config for checkpoint)
    trainer_config = {
        **flat_config,
        "epochs": train_cfg.get("epochs", 150),
        "patience": train_cfg.get("patience", 30),
    }

    # Train
    print("=" * 60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=trainer_config,
        output_dir=output_dir,
        device=device,
        scheduler=scheduler,
    )
    result = trainer.fit()
    print("=" * 60)
    print(f"Best Val Top-1: {result['best_val_top1']:.1f}% (epoch {result['best_epoch']})")
    print(f"Training time: {result['training_time_s']:.0f}s")

    # Load best checkpoint and evaluate on test
    ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device=device)
    label_names = [idx_to_gloss[i] for i in range(num_classes)]

    print(f"\nTest Top-1: {test_metrics['top1_acc']:.1f}%")
    print(f"Test Top-5: {test_metrics['top5_acc']:.1f}%")

    # Full evaluation artifacts
    eval_summary = run_full_evaluation(
        preds=test_metrics["preds"],
        labels=test_metrics["labels"],
        top1_acc=test_metrics["top1_acc"],
        top5_acc=test_metrics["top5_acc"],
        label_names=label_names,
        output_dir=output_dir,
        model_id=config.get("experiment_id", ""),
    )

    # Manifest
    config["dataset"]["num_train"] = len(train_ds)
    config["dataset"]["num_val"] = len(val_ds)
    config["dataset"]["num_test"] = len(test_ds)
    generate_manifest(
        config=config,
        training_result=result,
        eval_summary=eval_summary,
        output_dir=output_dir,
        experiment_id=config.get("experiment_id", ""),
        title=config.get("title", ""),
        research_question=config.get("research_question", ""),
    )
    print(f"\nAll artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
