"""PyTorch Dataset classes for landmark sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from slr.data.transforms import pad_or_truncate


class DirectoryDataset(Dataset):
    """Loads landmark ``.npy`` files from a split-based directory tree.

    Directory layout::

        root/
          train/
            gloss_a/  001.npy  002.npy ...
            gloss_b/  ...
          val/  ...
          test/ ...

    Each item returns ``(tensor[max_frames, input_size], label_idx)``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        gloss_to_idx: dict[str, int],
        max_frames: int = 64,
        input_size: int = 63,
        splits: tuple[str, ...] = ("train", "val", "test"),
        augmentation: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames
        self.input_size = input_size
        self.augmentation = augmentation
        self.gloss_to_idx = gloss_to_idx

        self.samples: list[tuple[Path, int, str]] = []
        self.labels: np.ndarray

        for split in splits:
            split_dir = self.root_dir / split
            if not split_dir.exists():
                continue
            for gloss_dir in sorted(split_dir.iterdir()):
                if not gloss_dir.is_dir():
                    continue
                gloss = gloss_dir.name
                if gloss not in gloss_to_idx:
                    continue
                label = gloss_to_idx[gloss]
                for npy_file in sorted(gloss_dir.glob("*.npy")):
                    self.samples.append((npy_file, label, gloss))

        self.labels = np.array([s[1] for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, label, _ = self.samples[idx]
        landmarks = np.load(npy_path)
        landmarks = pad_or_truncate(landmarks, self.max_frames)
        if self.augmentation is not None:
            landmarks = self.augmentation(landmarks)
        x = torch.tensor(
            landmarks.reshape(self.max_frames, self.input_size),
            dtype=torch.float32,
        )
        return x, label


class SubsetDataset(Dataset):
    """Loads landmark files using paths from a subset JSON.

    Each sample dict has ``{"path": "train/gloss/id.npy", "gloss": str, "gloss_idx": int}``.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        landmarks_root: str | Path,
        max_frames: int = 64,
        input_size: int = 63,
        augmentation: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.samples = samples
        self.root = Path(landmarks_root)
        self.max_frames = max_frames
        self.input_size = input_size
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.samples[idx]
        landmarks = np.load(self.root / item["path"])
        landmarks = pad_or_truncate(landmarks, self.max_frames)
        if self.augmentation is not None:
            landmarks = self.augmentation(landmarks)
        x = torch.tensor(
            landmarks.reshape(self.max_frames, self.input_size),
            dtype=torch.float32,
        )
        return x, item["gloss_idx"]
