"""Smoke tests for loading the MS-ASL50 landmark subset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from slr.data.dataset import SubsetDataset


METADATA_PATH = Path("06_datasets/MS-ASL/subsets/msasl50/metadata.json")
LANDMARKS_ROOT = Path("06_datasets/MS-ASL/subsets/msasl50/landmarks")


def test_msasl50_subset_dataset_loads_first_sample() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    train_samples = metadata["samples"]["train"]

    dataset = SubsetDataset(
        samples=train_samples,
        landmarks_root=LANDMARKS_ROOT,
        max_frames=64,
        input_size=63,
    )

    x, label = dataset[0]

    assert x.shape == (64, 63)
    assert x.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < 50


def test_msasl50_subset_dataset_batches_to_expected_shape() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    train_samples = metadata["samples"]["train"]
    dataset = SubsetDataset(
        samples=train_samples,
        landmarks_root=LANDMARKS_ROOT,
        max_frames=64,
        input_size=63,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    x, y = next(iter(loader))

    assert x.shape == (1, 64, 63)
    assert y.shape == (1,)
    assert 0 <= int(y.item()) < 50
