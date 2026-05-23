"""
Generate a single image with MediaPipe Hands landmarks for the Goals slide.

Takes any photo of a hand making a sign and produces a clean version with
the 21 keypoints + skeleton overlaid, ready to drop into Canva.

Run from the project root with the venv activated:
    source models/bin/activate
    python scripts/generate_goal_slide_image.py
"""
import cv2
import numpy as np
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================================
# CONFIG — adjust these paths
# ============================================================================
# Path to your input image. Can be:
#   - A frame extracted from one of your WLASL videos
#   - A photo you take yourself making a sign
#   - Any clean photo of a hand
INPUT_IMAGE = Path("/home/fransando/Pictures/Screenshots/handimage.png")

# Where the output images will be saved
OUTPUT_DIR = Path.home() / "Desktop" / "tfg_goal_slide"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If INPUT_IMAGE is a video, which frame to extract (middle frame by default)
# Set to None to use the input as-is (when it's already a .jpg / .png)
EXTRACT_FRAME_INDEX = "middle"   # options: integer index, "middle", or None

# Output dimensions for the final image (Canva-friendly)
OUTPUT_WIDTH = 800

# Visual style of landmarks (tuned to match the GIF you already have)
KEYPOINT_RADIUS = 5             # bigger than the GIF (3) because it's a static image
KEYPOINT_COLOR_BGR = (50, 50, 220)   # red (BGR) — slightly softer than pure red
CONNECTION_THICKNESS = 2
CONNECTION_COLOR_BGR = (60, 200, 60)  # green (BGR)

# Hand landmarker model (downloaded once, cached)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"

# ============================================================================
# Download the MediaPipe model if not cached
# ============================================================================
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ============================================================================
# Load the input image (or extract a frame from a video)
# ============================================================================
assert INPUT_IMAGE.exists(), f"Input not found: {INPUT_IMAGE}"
print(f"Reading input: {INPUT_IMAGE}")

is_video = INPUT_IMAGE.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm"}

if is_video:
    cap = cv2.VideoCapture(str(INPUT_IMAGE))
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    print(f"Video has {len(all_frames)} frames")

    if EXTRACT_FRAME_INDEX == "middle":
        idx = len(all_frames) // 2
    elif isinstance(EXTRACT_FRAME_INDEX, int):
        idx = EXTRACT_FRAME_INDEX
    else:
        idx = len(all_frames) // 2
    print(f"Extracting frame at index {idx}")
    img_bgr = all_frames[idx]
else:
    img_bgr = cv2.imread(str(INPUT_IMAGE))
    if img_bgr is None:
        raise RuntimeError(f"Could not decode image: {INPUT_IMAGE}")

# ============================================================================
# Resize to target width (keep aspect ratio)
# ============================================================================
h, w = img_bgr.shape[:2]
target_height = int(OUTPUT_WIDTH * h / w)
img_bgr = cv2.resize(img_bgr, (OUTPUT_WIDTH, target_height))
print(f"Resized to {OUTPUT_WIDTH}x{target_height}")

# ============================================================================
# Run MediaPipe Hands
# ============================================================================
base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
result = landmarker.detect(mp_image)
landmarker.close()

if not result.hand_landmarks:
    print("WARNING: No hand detected. Try a different frame or check the image.")
else:
    print(f"Detected {len(result.hand_landmarks)} hand(s)")

# ============================================================================
# Draw landmarks on a copy of the original
# ============================================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

annotated_bgr = img_bgr.copy()

if result.hand_landmarks:
    for hand in result.hand_landmarks:
        pts = [(int(lm.x * OUTPUT_WIDTH), int(lm.y * target_height)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated_bgr, pts[a], pts[b], CONNECTION_COLOR_BGR, CONNECTION_THICKNESS)
        for x, y in pts:
            cv2.circle(annotated_bgr, (x, y), KEYPOINT_RADIUS, KEYPOINT_COLOR_BGR, -1)

# ============================================================================
# Save both versions: raw + annotated
# ============================================================================
raw_path = OUTPUT_DIR / "hand_raw.png"
annotated_path = OUTPUT_DIR / "hand_with_landmarks.png"

cv2.imwrite(str(raw_path), img_bgr)
cv2.imwrite(str(annotated_path), annotated_bgr)

print(f"\nDone.")
print(f"  Raw image:        {raw_path}  ({raw_path.stat().st_size / 1024:.1f} KB)")
print(f"  With landmarks:   {annotated_path}  ({annotated_path.stat().st_size / 1024:.1f} KB)")
print(f"\nUse 'hand_with_landmarks.png' in the Goals slide.")