"""MediaPipe landmark extraction from video files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """Read RGB frames from a video file, optionally subsampling.

    Returns (frames, fps).  Requires ``opencv-python``.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if total == 0:
        cap.release()
        return [], fps

    if max_frames is not None and total > max_frames:
        selected = set(np.linspace(0, total - 1, max_frames, dtype=int))
    else:
        selected = set(range(total))

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in selected:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1

    cap.release()
    return frames, fps


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Wrist-center and scale by wrist-to-middle-finger distance.

    Parameters
    ----------
    landmarks : array of shape ``(T, 21, 3)``

    Returns
    -------
    Normalized array of same shape.
    """
    normalized = np.copy(landmarks)
    for i in range(landmarks.shape[0]):
        frame = landmarks[i]
        centered = frame - frame[0]
        scale = np.linalg.norm(frame[12] - frame[0])
        normalized[i] = centered / scale if scale > 1e-6 else centered
    return normalized


def interpolate_missing_frames(
    landmarks_list: list[Optional[np.ndarray]],
) -> np.ndarray:
    """Fill None entries with linear interpolation or edge values.

    Parameters
    ----------
    landmarks_list : list of arrays ``(21, 3)`` or None

    Returns
    -------
    Filled array of shape ``(T, 21, 3)``.
    """
    T = len(landmarks_list)
    result = np.zeros((T, 21, 3), dtype=np.float32)
    valid = [i for i, lm in enumerate(landmarks_list) if lm is not None]

    if not valid:
        return result

    for i in valid:
        result[i] = landmarks_list[i]

    for i in range(T):
        if landmarks_list[i] is not None:
            continue
        prev_idx = next((j for j in range(i - 1, -1, -1) if landmarks_list[j] is not None), None)
        next_idx = next((j for j in range(i + 1, T) if landmarks_list[j] is not None), None)

        if prev_idx is not None and next_idx is not None:
            alpha = (i - prev_idx) / (next_idx - prev_idx)
            result[i] = (1 - alpha) * landmarks_list[prev_idx] + alpha * landmarks_list[next_idx]
        elif prev_idx is not None:
            result[i] = landmarks_list[prev_idx]
        elif next_idx is not None:
            result[i] = landmarks_list[next_idx]

    return result
