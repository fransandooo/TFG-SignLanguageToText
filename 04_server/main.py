# main.py
import os
import uuid
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Optional: only if you want to debug by saving PNGs
from PIL import Image

from ultralytics import YOLO

# ---- Config ----
SESSIONS_ROOT = Path("./sessions")
SESSIONS_ROOT.mkdir(exist_ok=True, parents=True)

# Confidence threshold for keeping detections
MIN_CONFIDENCE = 0.5  # try 0.5 or 0.6; you can tune this later


# YOLO model configuration: we are training with imgsz=416
MODEL_IMG_SIZE = 416
MODEL_CHANNELS = 3  # YOLO expects RGB

# ✅ NEW: path to your trained YOLO checkpoint
# Change this to your actual best.pt path
MODEL_WEIGHTS_PATH = Path(
    "../hand_bbox_baseline/runs_yolo/yolov8n_mix_base_lr/weights/best.pt"
)

if not MODEL_WEIGHTS_PATH.exists():
    raise RuntimeError(f"YOLO weights not found at: {MODEL_WEIGHTS_PATH}")

# ✅ NEW: load YOLO model once at startup
hand_model = YOLO(str(MODEL_WEIGHTS_PATH))
CLASS_NAMES = hand_model.names  # e.g. {0: "hand"} if you trained 1-class model

app = FastAPI(
    title="ASL Hand Detection Session API",
    description=(
        "API that receives pixel traces (frames as flattened pixel arrays) "
        "grouped by session and runs a hand-detection YOLO model over them."
    ),
    version="0.2.0",
)


# ---- Pydantic models (request/response schemas) ----

class CreateSessionResponse(BaseModel):
    session_id: str


class PixelTrace(BaseModel):
    """
    Representation of a single video frame as a flattened pixel array.

    For this TFG setup, the client (mobile) should:
      - Resize each frame to 416x416 pixels
      - Use 3 channels (RGB)
      - Flatten to width * height * channels values (0-255)
    """
    width: int = Field(
        MODEL_IMG_SIZE,
        gt=0,
        description=f"Frame width in pixels (must be {MODEL_IMG_SIZE} for YOLO).",
    )
    height: int = Field(
        MODEL_IMG_SIZE,
        gt=0,
        description=f"Frame height in pixels (must be {MODEL_IMG_SIZE} for YOLO).",
    )
    channels: int = Field(
        MODEL_CHANNELS,
        description="Number of channels: 3 for RGB (recommended).",
    )
    format: Literal["uint8"] = "uint8"
    frame_index: int = Field(
        ...,
        ge=0,
        description="Index of the frame in the session (0-based, assigned by client).",
    )
    data: List[int] = Field(
        ...,
        description="Flattened pixel values 0-255, length must be width * height * channels.",
    )


class AddFrameResponse(BaseModel):
    session_id: str
    frame_index: int
    stored_path: str


class PredictionDet(BaseModel):
    frame_index: int
    label: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x_min, y_min, x_max, y_max] in pixel coordinates
    crop_path: Optional[str] = None     # path to cropped image on server (optional)



class SessionPredictionResponse(BaseModel):
    session_id: str
    num_frames: int
    predictions: List[PredictionDet]
    aggregated: Dict[str, Any]


# ---- Helper functions ----

def session_dir(session_id: str) -> Path:
    return SESSIONS_ROOT / session_id


def ensure_session_exists(session_id: str) -> Path:
    sdir = session_dir(session_id)
    if not sdir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return sdir


def save_pixel_trace(session_id: str, trace: PixelTrace) -> Path:
    """Convert the flattened trace data to a numpy array and save it.

    We enforce that the input resolution matches the YOLO training resolution (416x416).
    """
    sdir = ensure_session_exists(session_id)

    # Enforce 416x416 for now, to match YOLO training
    if trace.width != MODEL_IMG_SIZE or trace.height != MODEL_IMG_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Frame must be {MODEL_IMG_SIZE}x{MODEL_IMG_SIZE} pixels. "
                f"Got {trace.width}x{trace.height}."
            ),
        )

    if trace.channels not in (1, 3):
        raise HTTPException(
            status_code=400,
            detail="Only 1-channel (grayscale) or 3-channel (RGB) frames are supported.",
        )

    expected_len = trace.width * trace.height * trace.channels
    if len(trace.data) != expected_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Data length {len(trace.data)} does not match "
                f"width*height*channels = {expected_len}"
            ),
        )

    # Convert to numpy array
    arr = np.array(trace.data, dtype=np.uint8)

    if trace.channels == 1:
        arr = arr.reshape((trace.height, trace.width))
    else:
        arr = arr.reshape((trace.height, trace.width, trace.channels))

    # Save as .npy for quick loading later
    frame_name = f"frame_{trace.frame_index:04d}.npy"
    frame_path = sdir / frame_name
    np.save(frame_path, arr)

    # Optional: also save as PNG for debugging / visualization
    png_path = sdir / f"frame_{trace.frame_index:04d}.png"
    img = Image.fromarray(arr)
    img.save(png_path)

    return frame_path


def load_session_frames(session_id: str) -> List[np.ndarray]:
    """Load all frames (.npy) for a given session in sorted order."""
    sdir = ensure_session_exists(session_id)
    frame_paths = sorted(sdir.glob("frame_*.npy"))
    frames = [np.load(p) for p in frame_paths]
    return frames

def load_session_frame_pngs(session_id: str) -> List[Path]:
    """Load all frame PNG paths for a given session in sorted order."""
    sdir = ensure_session_exists(session_id)
    frame_paths = sorted(sdir.glob("frame_*.png"))
    return frame_paths

def yolo_model_predict_from_paths(frame_paths: List[Path]) -> (List[PredictionDet], Dict[str, Any]):
    """
    Run the loaded YOLO model on the list of frame PNG paths.

    For each detection with confidence >= MIN_CONFIDENCE:
      - Save a cropped hand image in a 'crops' subfolder next to the frames.
      - Attach the crop_path to the PredictionDet.
    """
    predictions: List[PredictionDet] = []

    print(f"[YOLO] Running prediction on {len(frame_paths)} frames with MIN_CONFIDENCE={MIN_CONFIDENCE}...")

    for frame_index, frame_path in enumerate(frame_paths):
        print(f"[YOLO] Frame {frame_index}: {frame_path}")

        # Load full frame once for cropping
        img = Image.open(frame_path).convert("RGB")
        img_w, img_h = img.size  # should be 416x416

        # Use YOLO with confidence threshold
        results = hand_model(
            str(frame_path),
            conf=MIN_CONFIDENCE,  # YOLO will discard lower-conf boxes
            verbose=False,
        )
        res = results[0]
        boxes = res.boxes

        num_boxes = 0 if boxes is None else len(boxes)
        print(f"[YOLO]   raw detections from model: {num_boxes}")

        if boxes is None or len(boxes) == 0:
            continue

        crops_dir = frame_path.parent / "crops"
        crops_dir.mkdir(exist_ok=True)

        kept_for_frame = 0

        for det_idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf.cpu().numpy().ravel()[0])
            cls_idx = int(box.cls.cpu().numpy().ravel()[0])

            # Extra safety: skip low-confidence boxes in our own logic too
            if conf < MIN_CONFIDENCE:
                print(f"[YOLO]   skipping box conf={conf:.3f} < {MIN_CONFIDENCE}")
                continue

            # Clamp coords
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(img_w, int(x1)))
            y1 = max(0, min(img_h, int(y1)))
            x2 = max(0, min(img_w, int(x2)))
            y2 = max(0, min(img_h, int(y2)))

            if x2 <= x1 or y2 <= y1:
                print(f"[YOLO]   skipped degenerate box: {(x1, y1, x2, y2)}")
                continue

            crop_img = img.crop((x1, y1, x2, y2))

            crop_filename = f"crop_{frame_index:04d}_{kept_for_frame:02d}.png"
            crop_path = crops_dir / crop_filename
            crop_img.save(crop_path)

            label = CLASS_NAMES.get(cls_idx, "hand") if isinstance(CLASS_NAMES, dict) else "hand"

            pred = PredictionDet(
                frame_index=frame_index,
                label=label,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                crop_path=str(crop_path),
            )
            predictions.append(pred)
            kept_for_frame += 1

        print(f"[YOLO]   kept {kept_for_frame} detections after thresholding")

    has_hand = len(predictions) > 0
    max_conf = max((p.confidence for p in predictions), default=0.0)

    aggregated = {
        "has_hand": has_hand,
        "max_confidence": max_conf,
        "num_frames": len(frame_paths),
        "num_detections": len(predictions),
        "min_confidence": MIN_CONFIDENCE,
    }

    print(f"[YOLO] Aggregated: {aggregated}")
    return predictions, aggregated


# ---- Endpoints ----

@app.post("/session", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    """Generate a new session ID and create a folder to store its frames."""
    session_id = str(uuid.uuid4())
    sdir = session_dir(session_id)
    sdir.mkdir(parents=True, exist_ok=False)
    return CreateSessionResponse(session_id=session_id)


@app.post("/session/{session_id}/add_frame", response_model=AddFrameResponse)
def add_frame(session_id: str, trace: PixelTrace) -> AddFrameResponse:
    """Add a pixel trace frame to a given session.

    The client must send the frame already resized to 416x416.
    """
    frame_path = save_pixel_trace(session_id, trace)
    return AddFrameResponse(
        session_id=session_id,
        frame_index=trace.frame_index,
        stored_path=str(frame_path),
    )


@app.post("/session/{session_id}/predict", response_model=SessionPredictionResponse)
def predict_session(session_id: str) -> SessionPredictionResponse:
    """Load all frames for a session and run YOLO model inference + cropping."""
    frame_paths = load_session_frame_pngs(session_id)
    if not frame_paths:
        raise HTTPException(status_code=400, detail="Session has no frames")

    predictions, aggregated = yolo_model_predict_from_paths(frame_paths)
    return SessionPredictionResponse(
        session_id=session_id,
        num_frames=len(frame_paths),
        predictions=predictions,
        aggregated=aggregated,
    )


