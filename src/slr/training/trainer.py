"""Reusable training loop with early stopping, checkpointing, and TensorBoard."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from slr.evaluation.metrics import evaluate


class Trainer:
    """Trains a model with early stopping, checkpointing, and TensorBoard logging.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    optimizer : Optimizer
    criterion : nn.Module
    scheduler : _LRScheduler or None
    config : dict
        Must contain ``epochs``, ``patience``. May contain ``seed``.
    output_dir : str or Path
    device : torch.device or str
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        config: dict[str, Any],
        output_dir: str | Path,
        device: torch.device | str = "cpu",
        scheduler: _LRScheduler | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

    def fit(self) -> dict[str, Any]:
        """Run the full training loop. Returns a history dict."""
        epochs = self.config["epochs"]
        patience = self.config["patience"]

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_top1": [],
            "val_top5": [],
        }
        best_val_top1 = 0.0
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            train_loss, train_acc = self._train_one_epoch()
            val_metrics = evaluate(
                self.model, self.val_loader, device=self.device
            )
            val_top1 = val_metrics["top1_acc"]
            val_top5 = val_metrics["top5_acc"]

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_top1"].append(val_top1)
            history["val_top5"].append(val_top5)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val_top1", val_top1, epoch)
            self.writer.add_scalar("Accuracy/val_top5", val_top5, epoch)

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_epoch = epoch + 1
                patience_counter = 0
                self._save_checkpoint(epoch, best_val_top1)
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Ep {epoch + 1:3d}/{epochs} | "
                    f"Loss {train_loss:.3f} | "
                    f"Train {train_acc:.1f}% | "
                    f"Val {val_top1:.1f}% (Top-5 {val_top5:.1f}%) | "
                    f"Best {best_val_top1:.1f}%"
                )

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        self.writer.close()
        elapsed = time.time() - start_time

        self._save_training_curves(history)
        self._save_history(history)

        return {
            "best_val_top1": best_val_top1,
            "best_epoch": best_epoch,
            "total_epochs": len(history["train_loss"]),
            "training_time_s": round(elapsed, 1),
            "history": history,
        }

    def _train_one_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += x.size(0)

        return total_loss / total, 100.0 * correct / total

    def _save_checkpoint(self, epoch: int, val_top1: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_top1": val_top1,
                "config": self.config,
            },
            self.output_dir / "best_model.pt",
        )

    def _save_training_curves(self, history: dict[str, list[float]]) -> None:
        epochs_ran = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(epochs_ran, history["train_loss"], color="steelblue")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_ran, history["train_acc"], label="Train", color="steelblue")
        axes[1].plot(epochs_ran, history["val_top1"], label="Val Top-1", color="darkorange")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs_ran, history["val_top5"], color="green", label="Val Top-5")
        axes[2].plot(epochs_ran, history["val_top1"], color="darkorange", label="Val Top-1")
        axes[2].set_title("Val Top-1 vs Top-5")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy (%)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("Training Curves", fontsize=13)
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_history(self, history: dict[str, list[float]]) -> None:
        serializable = {
            k: [round(float(v), 4) for v in vals]
            for k, vals in history.items()
        }
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
