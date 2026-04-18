"""Sequence transforms: augmentation, padding, and interpolation."""

from __future__ import annotations

import random

import numpy as np


class SkeletalAugmentation:
    """Soft geometric augmentation for hand-landmark sequences.

    Applied on numpy arrays of shape ``(T, 21, 3)`` before reshaping.
    """

    def __init__(
        self,
        rotation: float = 10.0,
        scale: tuple[float, float] = (0.9, 1.1),
        translation: float = 0.1,
        noise: float = 0.01,
        p: float = 0.5,
    ):
        self.rotation = rotation
        self.scale = scale
        self.translation = translation
        self.noise = noise
        self.p = p

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        landmarks = landmarks.copy()

        if random.random() < self.p:
            angle = np.radians(random.uniform(-self.rotation, self.rotation))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x = landmarks[:, :, 0].copy()
            y = landmarks[:, :, 1].copy()
            landmarks[:, :, 0] = x * cos_a - y * sin_a
            landmarks[:, :, 1] = x * sin_a + y * cos_a

        if random.random() < self.p:
            landmarks *= random.uniform(*self.scale)

        if random.random() < self.p:
            landmarks[:, :, 0] += random.uniform(-self.translation, self.translation)
            landmarks[:, :, 1] += random.uniform(-self.translation, self.translation)

        if random.random() < self.p:
            landmarks += np.random.normal(0, self.noise, landmarks.shape)

        if random.random() < self.p * 0.5:
            landmarks[:, :, 0] = -landmarks[:, :, 0]

        return landmarks


def pad_or_truncate(landmarks: np.ndarray, max_frames: int) -> np.ndarray:
    """Resample or zero-pad a landmark sequence to exactly *max_frames*.

    - If the sequence is longer, uniformly sample *max_frames* indices.
    - If shorter, zero-pad at the end.
    """
    T = landmarks.shape[0]
    if T > max_frames:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        return landmarks[indices]
    if T < max_frames:
        pad_shape = (max_frames - T,) + landmarks.shape[1:]
        padding = np.zeros(pad_shape, dtype=landmarks.dtype)
        return np.concatenate([landmarks, padding], axis=0)
    return landmarks
