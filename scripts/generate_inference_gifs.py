"""
Generate two GIFs from a single sign video clip using MediaPipe Tasks API
(compatible with mediapipe >= 0.10.10).

If you prefer the legacy solutions API, downgrade to mediapipe==0.10.9
and use generate_inference_gifs.py instead.
"""
import cv2
import numpy as np
import imageio.v2 as imageio
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================================
# CONFIG
# ============================================================================
VIDEO_PATH = Path("/home/fransando/Dev/tfg-sign-language-recognition/06_datasets/wlasl/raw/kaggle_wlasl/videos/38999.mp4")
OUTPUT_DIR = Path.home() / "Desktop" / "tfg_gifs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES = 12
TARGET_WIDTH = 320
FPS = 8
LOOP = 0

# Hand landmarker model (downloaded once)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"

# ============================================================================
# Download model if needed
# ============================================================================
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ============================================================================
# Read video
# ============================================================================
assert VIDEO_PATH.exists(), f"Video not found: {VIDEO_PATH}"
print(f"Reading video: {VIDEO_PATH}")

cap = cv2.VideoCapture(str(VIDEO_PATH))
all_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    all_frames.append(frame)
cap.release()
print(f"Read {len(all_frames)} frames")

indices = np.linspace(0, len(all_frames) - 1, TARGET_FRAMES).astype(int)
sampled = [all_frames[i] for i in indices]
print(f"Sampled {len(sampled)} frames")

# ============================================================================
# Process frames with the new Tasks API
# ============================================================================
base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

# MediaPipe Hands connections (21 keypoints, standard skeleton)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                 # palm base
]

h, w = sampled[0].shape[:2]
target_height = int(TARGET_WIDTH * h / w)

raw_frames_rgb = []
landmark_frames_rgb = []

for idx, frame_bgr in enumerate(sampled):
    frame_bgr = cv2.resize(frame_bgr, (TARGET_WIDTH, target_height))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    raw_frames_rgb.append(frame_rgb)

    # Detect landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    annotated = frame_rgb.copy()
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            # Convert normalised coordinates to pixels
            pts = [(int(lm.x * TARGET_WIDTH), int(lm.y * target_height)) for lm in hand]

            # Líneas en azul primario UFV
            for a, b in HAND_CONNECTIONS:
                cv2.line(annotated, pts[a], pts[b], (220, 50, 50), 1)  # BGR de #1E3A8A

            # Puntos en ámbar de acento
            for x, y in pts:
                cv2.circle(annotated, (x, y), 2, (0, 170, 0), -1)  # BGR de #F59E0B

    landmark_frames_rgb.append(annotated)
    print(f"  Frame {idx + 1}/{len(sampled)} processed")

landmarker.close()

# ============================================================================
# Save GIFs
# ============================================================================
gif1_path = OUTPUT_DIR / "frames_no.gif"
gif2_path = OUTPUT_DIR / "landmarks_no.gif"
duration = 1.0 / FPS

print(f"\nSaving {gif1_path}")
imageio.mimsave(gif1_path, raw_frames_rgb, duration=duration, loop=LOOP)
print(f"Saving {gif2_path}")
imageio.mimsave(gif2_path, landmark_frames_rgb, duration=duration, loop=LOOP)

print("\nDone.")
print(f"  GIF 1 (raw frames):     {gif1_path.stat().st_size / 1024:.1f} KB")
print(f"  GIF 2 (with landmarks): {gif2_path.stat().st_size / 1024:.1f} KB")