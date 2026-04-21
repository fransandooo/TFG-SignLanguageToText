"""MediaPipe landmark extraction from video files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


HAND_LANDMARKER_CANDIDATES = (
    Path("04_server/hand_landmarker.task"),
    Path("01_notebooks/word_recognition/hand_landmarker.task"),
    Path("hand_landmarker.task"),
)
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class LandmarkExtractionStats:
    """Processing statistics for one video."""

    success: bool
    error: str | None
    sampled_frames: int
    detected_frames: int
    detection_rate: float


def find_hand_landmarker_model(model_path: str | Path | None = None) -> Path:
    """Return an existing MediaPipe HandLandmarker task file.

    The project already keeps the model in historical notebook/server folders.
    This helper does not download files; callers get a clear error if no local
    task model is available.
    """
    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"HandLandmarker task model not found: {path}")

    for candidate in HAND_LANDMARKER_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "HandLandmarker task model not found. Expected one of: "
        + ", ".join(str(path) for path in HAND_LANDMARKER_CANDIDATES)
        + f". Download source: {HAND_LANDMARKER_URL}"
    )


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


def create_hand_landmarker(
    model_path: str | Path | None = None,
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.3,
    min_hand_presence_confidence: float = 0.3,
    min_tracking_confidence: float = 0.3,
) -> Any:
    """Create a MediaPipe Tasks HandLandmarker instance."""
    import mediapipe as mp

    resolved_model_path = find_hand_landmarker_model(model_path)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_frame(frame: np.ndarray, detector: Any) -> Optional[np.ndarray]:
    """Extract the first detected hand from one RGB frame as ``(21, 3)``."""
    import mediapipe as mp

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
    except Exception:
        return None
    return None


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
    result = np.zeros((T, 21, 3))
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


def extract_landmarks_from_video(
    video_path: str | Path,
    detector: Any | None = None,
    max_frames: int = 64,
    min_detection_rate: float = 0.3,
    model_path: str | Path | None = None,
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.3,
    min_hand_presence_confidence: float = 0.3,
    min_tracking_confidence: float = 0.3,
) -> tuple[np.ndarray | None, LandmarkExtractionStats]:
    """Extract normalized MediaPipe hand landmarks from a video.

    Returns an array with shape ``(T, 21, 3)`` and the same normalization used
    for the WLASL landmark files: missing frames are linearly interpolated,
    edge gaps copy the nearest valid frame, and the sequence is wrist-centered
    and scaled by the wrist-to-landmark-12 distance. If the detection rate is
    below ``min_detection_rate``, returns ``None`` plus failure stats.
    """
    frames, _ = load_video_frames(video_path, max_frames=max_frames)
    if not frames:
        return None, LandmarkExtractionStats(
            success=False,
            error="no_frames_loaded",
            sampled_frames=0,
            detected_frames=0,
            detection_rate=0.0,
        )

    owns_detector = detector is None
    if detector is None:
        detector = create_hand_landmarker(
            model_path=model_path,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    try:
        landmarks_list: list[Optional[np.ndarray]] = []
        detected_frames = 0
        for frame in frames:
            landmarks = extract_landmarks_from_frame(frame, detector)
            landmarks_list.append(landmarks)
            if landmarks is not None:
                detected_frames += 1

        detection_rate = detected_frames / len(frames)
        if detection_rate < min_detection_rate:
            return None, LandmarkExtractionStats(
                success=False,
                error=f"low_detection_rate_{detection_rate:.2f}",
                sampled_frames=len(frames),
                detected_frames=detected_frames,
                detection_rate=detection_rate,
            )

        interpolated = interpolate_missing_frames(landmarks_list)
        normalized = normalize_landmarks(interpolated)
        return normalized, LandmarkExtractionStats(
            success=True,
            error=None,
            sampled_frames=len(frames),
            detected_frames=detected_frames,
            detection_rate=detection_rate,
        )
    finally:
        if owns_detector and hasattr(detector, "close"):
            detector.close()
