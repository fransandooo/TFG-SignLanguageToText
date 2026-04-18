"""
FastAPI server for ASL recognition.

Supports:
- Dynamic model selection through a curated model registry
- Isolated-word prediction (`/predict`)
- Heuristic continuous-sequence prediction (`/predict-sequence`)
"""

import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SERVER_DIR = Path(__file__).parent
BASE_DIR = SERVER_DIR.parent
HAND_LANDMARKER_TASK = SERVER_DIR / "hand_landmarker.task"
DEBUG_OUTPUT_DIR = BASE_DIR / "05_experiments" / "continuous_debug"

DEFAULT_MODEL_ID = "attention_30_prod"
MAX_FRAMES = 64
MIN_DETECTION_RATE = 0.3
CONTINUOUS_WINDOW_FRAMES = 48
CONTINUOUS_WINDOW_STRIDE = 8
SEQUENCE_CONFIDENCE_THRESHOLD = 0.40
SEQUENCE_RELAXED_SEGMENT_THRESHOLD = 0.35


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    label: str
    architecture: str
    checkpoint_path: Path
    metadata_path: Path
    num_classes: int
    dataset_name: str
    val_top1: float
    experimental: bool = False


MODEL_REGISTRY: Dict[str, ModelRegistryEntry] = {
    "attention_30_prod": ModelRegistryEntry(
        model_id="attention_30_prod",
        label="GRU+Attn 30 (Prod)",
        architecture="gru_attention",
        checkpoint_path=BASE_DIR / "05_experiments" / "final_assault" / "best_model_attention.pt",
        metadata_path=BASE_DIR / "06_datasets" / "wlasl" / "landmarks" / "wlasl30" / "metadata.json",
        num_classes=30,
        dataset_name="WLASL30",
        val_top1=64.29,
    ),
    "attention_30_kfold_best": ModelRegistryEntry(
        model_id="attention_30_kfold_best",
        label="GRU+Attn 30 (K-Fold Best)",
        architecture="gru_attention",
        checkpoint_path=BASE_DIR / "05_experiments" / "kfold_wlasl100" / "fold_0" / "best_model.pt",
        metadata_path=BASE_DIR / "06_datasets" / "wlasl" / "landmarks" / "wlasl100_filtered30" / "metadata.json",
        num_classes=30,
        dataset_name="WLASL100 Filtered30",
        val_top1=65.70,
    ),
    "transformer_30_exp": ModelRegistryEntry(
        model_id="transformer_30_exp",
        label="Transformer 30",
        architecture="transformer",
        checkpoint_path=BASE_DIR / "05_experiments" / "final_assault" / "best_model_transformer.pt",
        metadata_path=BASE_DIR / "06_datasets" / "wlasl" / "landmarks" / "wlasl30" / "metadata.json",
        num_classes=30,
        dataset_name="WLASL30",
        val_top1=50.00,
        experimental=True,
    ),
    "attention_100": ModelRegistryEntry(
        model_id="attention_100",
        label="GRU+Attn 100",
        architecture="gru_attention",
        checkpoint_path=BASE_DIR / "05_experiments" / "wlasl_100" / "best_model.pt",
        metadata_path=BASE_DIR / "06_datasets" / "wlasl" / "subsets" / "wlasl_100_subset.json",
        num_classes=100,
        dataset_name="WLASL100 subset",
        val_top1=51.61,
    ),
    "attention_300_exp": ModelRegistryEntry(
        model_id="attention_300_exp",
        label="GRU+Attn 300",
        architecture="gru_attention",
        checkpoint_path=BASE_DIR / "05_experiments" / "wlasl_300" / "best_model.pt",
        metadata_path=BASE_DIR / "06_datasets" / "wlasl" / "subsets" / "wlasl_300_subset.json",
        num_classes=300,
        dataset_name="WLASL300 subset",
        val_top1=32.07,
        experimental=True,
    ),
    "attention_1000_exp": ModelRegistryEntry(
        model_id="attention_1000_exp",
        label="GRU+Attn 1000",
        architecture="gru_attention",
        checkpoint_path=BASE_DIR / "05_experiments" / "wlasl_1000" / "best_model.pt",
        metadata_path=BASE_DIR / "06_datasets" / "wlasl" / "subsets" / "wlasl_1000_subset.json",
        num_classes=1000,
        dataset_name="WLASL1000 subset",
        val_top1=13.69,
        experimental=True,
    ),
}


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, rnn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(rnn_output).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), rnn_output).squeeze(1)
        return context, weights


class PoseClassifierAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        num_classes = config["num_classes"]
        dropout = config["dropout"]
        bidirectional = config.get("bidirectional", True)

        self.num_directions = 2 if bidirectional else 1
        rnn_hidden = hidden_size * self.num_directions

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.attention = TemporalAttention(rnn_hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(x)
        rnn_out, _ = self.gru(x)
        context, attn_weights = self.attention(rnn_out)
        logits = self.classifier(context)
        if return_attention:
            return logits, attn_weights
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 65, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoseTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        input_size = config["input_size"]
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_transformer_layers", 4)
        num_classes = config["num_classes"]
        dropout = config["dropout"]
        max_len = config.get("max_frames", 64)

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        cls_output = x[:, 0]
        return self.classifier(cls_output)


@dataclass
class LoadedModel:
    entry: ModelRegistryEntry
    model: nn.Module
    idx_to_gloss: Dict[int, str]
    config: Dict[str, Any]
    supports_attention: bool
    param_count: int


def build_model(architecture: str, config: Dict[str, Any]) -> nn.Module:
    if architecture == "gru_attention":
        return PoseClassifierAttention(config)
    if architecture == "transformer":
        return PoseTransformer(config)
    raise RuntimeError(f"Unsupported architecture: {architecture}")


def read_gloss_mapping(metadata_path: Path) -> Dict[int, str]:
    if not metadata_path.exists():
        raise RuntimeError(f"Metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    gloss_to_idx = metadata.get("gloss_to_idx")
    if not gloss_to_idx:
        glosses = metadata.get("glosses")
        if glosses:
            gloss_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}
        else:
            raise RuntimeError(f"No gloss mapping found in: {metadata_path}")

    return {int(idx): gloss for gloss, idx in gloss_to_idx.items()}


def load_video_frames(video_path: Path, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
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

    frames: List[np.ndarray] = []
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


def extract_landmarks_from_frame(frame: np.ndarray, detector: Any) -> Optional[np.ndarray]:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        return np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
    return None


def interpolate_missing_frames(landmarks_list: List[Optional[np.ndarray]]) -> np.ndarray:
    frame_count = len(landmarks_list)
    result = np.zeros((frame_count, 21, 3), dtype=np.float32)
    valid_indices = [idx for idx, lm in enumerate(landmarks_list) if lm is not None]

    if not valid_indices:
        return result

    for idx in valid_indices:
        result[idx] = landmarks_list[idx]

    for idx in range(frame_count):
        if landmarks_list[idx] is not None:
            continue
        prev_idx = next((j for j in range(idx - 1, -1, -1) if landmarks_list[j] is not None), None)
        next_idx = next((j for j in range(idx + 1, frame_count) if landmarks_list[j] is not None), None)

        if prev_idx is not None and next_idx is not None:
            alpha = (idx - prev_idx) / (next_idx - prev_idx)
            result[idx] = (1 - alpha) * landmarks_list[prev_idx] + alpha * landmarks_list[next_idx]
        elif prev_idx is not None:
            result[idx] = landmarks_list[prev_idx]
        elif next_idx is not None:
            result[idx] = landmarks_list[next_idx]

    return result


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    normalized = np.copy(landmarks)
    for frame_idx in range(landmarks.shape[0]):
        frame = landmarks[frame_idx]
        centered = frame - frame[0]
        scale = np.linalg.norm(frame[12] - frame[0])
        normalized[frame_idx] = centered / scale if scale > 1e-6 else centered
    return normalized


def pad_or_truncate_landmarks(landmarks: np.ndarray, max_frames: int) -> np.ndarray:
    frame_count = landmarks.shape[0]
    if frame_count > max_frames:
        indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        return landmarks[indices]
    if frame_count < max_frames:
        padding = np.zeros((max_frames - frame_count, 21, 3), dtype=np.float32)
        return np.concatenate([landmarks, padding], axis=0)
    return landmarks


def prepare_landmark_tensor(
    raw_landmarks: List[Optional[np.ndarray]], max_frames: int, min_detection_rate: float = MIN_DETECTION_RATE
) -> Tuple[np.ndarray, float]:
    if not raw_landmarks:
        raise HTTPException(status_code=422, detail="No frames available to process")

    detected = sum(1 for landmarks in raw_landmarks if landmarks is not None)
    detection_rate = detected / len(raw_landmarks)
    if detection_rate < min_detection_rate:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Hand detection rate too low ({detection_rate:.0%}). "
                "Keep the signing hand visible and centered."
            ),
        )

    interpolated = interpolate_missing_frames(raw_landmarks)
    normalized = normalize_landmarks(interpolated)
    normalized = pad_or_truncate_landmarks(normalized, max_frames)
    return normalized, detection_rate


app = FastAPI(
    title="ASL Recognition API",
    description="Dynamic model selection for isolated-word and continuous-sign demos.",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cpu")
detector: Any = None
model_cache: Dict[str, LoadedModel] = {}


def load_model(model_id: str) -> LoadedModel:
    if model_id in model_cache:
        return model_cache[model_id]

    entry = MODEL_REGISTRY.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
    if not entry.checkpoint_path.exists():
        raise HTTPException(status_code=500, detail=f"Checkpoint not found: {entry.checkpoint_path}")

    checkpoint = torch.load(entry.checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(entry.architecture, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_gloss = read_gloss_mapping(entry.metadata_path)
    if len(idx_to_gloss) != config["num_classes"]:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Label mapping size mismatch for {model_id}: "
                f"{len(idx_to_gloss)} labels vs {config['num_classes']} classes"
            ),
        )

    loaded = LoadedModel(
        entry=entry,
        model=model,
        idx_to_gloss=idx_to_gloss,
        config=config,
        supports_attention=entry.architecture == "gru_attention",
        param_count=sum(param.numel() for param in model.parameters()),
    )
    model_cache[model_id] = loaded
    return loaded


def get_model(model_id: Optional[str]) -> LoadedModel:
    return load_model(model_id or DEFAULT_MODEL_ID)


def load_detector() -> Any:
    if not HAND_LANDMARKER_TASK.exists():
        raise RuntimeError(f"hand_landmarker.task not found: {HAND_LANDMARKER_TASK}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_TASK)),
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


@app.on_event("startup")
def startup() -> None:
    global detector
    detector = load_detector()
    load_model(DEFAULT_MODEL_ID)


def run_prediction(
    loaded_model: LoadedModel,
    normalized_landmarks: np.ndarray,
    return_attention: bool,
) -> Tuple[List["Prediction"], Optional[List[float]]]:
    max_frames = loaded_model.config.get("max_frames", MAX_FRAMES)
    seq = torch.from_numpy(normalized_landmarks.reshape(max_frames, 63).astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        if return_attention and loaded_model.supports_attention:
            logits, attn_weights = loaded_model.model(seq, return_attention=True)
            attention_weights = attn_weights[0].tolist()
        else:
            logits = loaded_model.model(seq)
            attention_weights = None
        probs = F.softmax(logits, dim=1)[0]

    top_k = min(5, probs.shape[0])
    top_values, top_indices = torch.topk(probs, k=top_k)
    predictions = [
        Prediction(
            word=loaded_model.idx_to_gloss[index.item()],
            confidence=round(value.item(), 4),
            rank=rank + 1,
        )
        for rank, (value, index) in enumerate(zip(top_values, top_indices))
    ]
    if attention_weights is not None:
        attention_weights = [round(weight, 5) for weight in attention_weights]
    return predictions, attention_weights


def save_upload(video: UploadFile) -> Path:
    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video.file.read())
        return Path(tmp.name)


def build_sequence_response(
    loaded_model: LoadedModel,
    raw_landmarks: List[Optional[np.ndarray]],
    fps: float,
    frames_processed: int,
    processing_time_ms: float,
    debug: bool = False,
    save_debug: bool = False,
    video_filename: Optional[str] = None,
) -> "SequencePredictResponse":
    if not raw_landmarks:
        raise HTTPException(status_code=422, detail="No frames available to process")

    model_max_frames = loaded_model.config.get("max_frames", MAX_FRAMES)
    total_frames = len(raw_landmarks)
    window_frames = min(CONTINUOUS_WINDOW_FRAMES, total_frames)
    stride = CONTINUOUS_WINDOW_STRIDE if total_frames > window_frames else window_frames
    overall_detection_rate = sum(1 for landmarks in raw_landmarks if landmarks is not None) / total_frames

    if total_frames <= window_frames:
        starts = [0]
    else:
        starts = list(range(0, total_frames - window_frames + 1, stride))
        last_start = total_frames - window_frames
        if starts[-1] != last_start:
            starts.append(last_start)

    candidates: List[Dict[str, Any]] = []
    window_detection_rates: List[float] = []
    window_debug_rows: List[SequenceWindowDebug] = []

    for window_index, start_idx in enumerate(starts):
        end_idx = min(start_idx + window_frames, total_frames)
        window_landmarks = raw_landmarks[start_idx:end_idx]
        detected = sum(1 for landmarks in window_landmarks if landmarks is not None)
        window_detection_rate = detected / len(window_landmarks)
        window_detection_rates.append(window_detection_rate)
        top_predictions: List[Prediction] = []
        accepted = False
        rejection_reason: Optional[str] = None

        if window_detection_rate < MIN_DETECTION_RATE:
            rejection_reason = "low_detection"
        else:
            normalized, _ = prepare_landmark_tensor(window_landmarks, model_max_frames)
            predictions, _ = run_prediction(loaded_model, normalized, return_attention=False)
            top_predictions = predictions
            top_prediction = predictions[0]
            if top_prediction.confidence < SEQUENCE_CONFIDENCE_THRESHOLD:
                rejection_reason = "low_confidence"
            else:
                accepted = True
                candidates.append(
                    {
                        "word": top_prediction.word,
                        "confidence": top_prediction.confidence,
                        "start_frame": start_idx,
                        "end_frame": end_idx,
                    }
                )

        if debug:
            window_debug_rows.append(
                SequenceWindowDebug(
                    index=window_index,
                    start_frame=start_idx,
                    end_frame=end_idx,
                    start_ms=int(round(start_idx / fps * 1000)),
                    end_ms=int(round(end_idx / fps * 1000)),
                    detection_rate=round(window_detection_rate, 3),
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    top_predictions=top_predictions,
                )
            )

    merged: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not merged:
            merged.append(
                {
                    "word": candidate["word"],
                    "start_frame": candidate["start_frame"],
                    "end_frame": candidate["end_frame"],
                    "confidence_sum": candidate["confidence"],
                    "windows": 1,
                }
            )
            continue

        last = merged[-1]
        same_word = last["word"] == candidate["word"]
        close_enough = candidate["start_frame"] - last["end_frame"] <= stride

        if same_word and close_enough:
            last["end_frame"] = max(last["end_frame"], candidate["end_frame"])
            last["confidence_sum"] += candidate["confidence"]
            last["windows"] += 1
        else:
            merged.append(
                {
                    "word": candidate["word"],
                    "start_frame": candidate["start_frame"],
                    "end_frame": candidate["end_frame"],
                    "confidence_sum": candidate["confidence"],
                    "windows": 1,
                }
            )

    selection_strategy = "stable_segments"
    filtered = [segment for segment in merged if segment["windows"] >= 2]
    if not filtered and merged:
        relaxed_segments = [
            segment
            for segment in merged
            if (segment["confidence_sum"] / segment["windows"]) >= SEQUENCE_RELAXED_SEGMENT_THRESHOLD
        ]
        if relaxed_segments:
            selection_strategy = "relaxed_segments"
            filtered = relaxed_segments
        else:
            selection_strategy = "best_merged_fallback"
            filtered = [max(merged, key=lambda segment: segment["confidence_sum"] / segment["windows"])]

    fallback_prediction: Optional[Prediction] = None
    if not filtered:
        selection_strategy = "whole_video_fallback"
        normalized, _ = prepare_landmark_tensor(raw_landmarks, model_max_frames)
        predictions, _ = run_prediction(loaded_model, normalized, return_attention=False)
        fallback_prediction = predictions[0]
        filtered = [
            {
                "word": predictions[0].word,
                "start_frame": 0,
                "end_frame": total_frames,
                "confidence_sum": predictions[0].confidence,
                "windows": 1,
            }
        ]

    segments = [
        SequenceSegment(
            word=segment["word"],
            start_ms=int(round(segment["start_frame"] / fps * 1000)),
            end_ms=int(round(segment["end_frame"] / fps * 1000)),
            confidence=round(segment["confidence_sum"] / segment["windows"], 4),
            windows=segment["windows"],
        )
        for segment in filtered
    ]

    debug_payload: Optional[SequenceDebugInfo] = None
    if debug:
        selected_keys = {
            (segment["word"], segment["start_frame"], segment["end_frame"], segment["windows"])
            for segment in filtered
        }
        merged_debug = [
            SequenceMergeDebugSegment(
                word=segment["word"],
                start_frame=segment["start_frame"],
                end_frame=segment["end_frame"],
                start_ms=int(round(segment["start_frame"] / fps * 1000)),
                end_ms=int(round(segment["end_frame"] / fps * 1000)),
                avg_confidence=round(segment["confidence_sum"] / segment["windows"], 4),
                windows=segment["windows"],
                selected=(
                    segment["word"],
                    segment["start_frame"],
                    segment["end_frame"],
                    segment["windows"],
                ) in selected_keys,
            )
            for segment in merged
        ]

        debug_payload = SequenceDebugInfo(
            video_filename=video_filename,
            total_frames=total_frames,
            fps=round(fps, 3),
            window_size_frames=window_frames,
            model_input_frames=model_max_frames,
            stride_frames=stride,
            min_detection_rate=MIN_DETECTION_RATE,
            confidence_threshold=SEQUENCE_CONFIDENCE_THRESHOLD,
            relaxed_segment_threshold=SEQUENCE_RELAXED_SEGMENT_THRESHOLD,
            raw_window_count=len(starts),
            accepted_window_count=len(candidates),
            merged_segment_count=len(merged),
            final_segment_count=len(segments),
            overall_detection_rate=round(overall_detection_rate, 3),
            selection_strategy=selection_strategy,
            fallback_prediction=fallback_prediction,
            windows=window_debug_rows,
            merged_segments=merged_debug,
        )

    response = SequencePredictResponse(
        mode="sequence",
        model_id=loaded_model.entry.model_id,
        model_label=loaded_model.entry.label,
        architecture=loaded_model.entry.architecture,
        num_classes=loaded_model.entry.num_classes,
        sequence_text=" ".join(segment.word for segment in segments),
        segments=segments,
        frames_processed=frames_processed,
        window_count=len(starts),
        avg_detection_rate=round(
            sum(window_detection_rates) / len(window_detection_rates)
            if window_detection_rates
            else overall_detection_rate,
            3,
        ),
        processing_time_ms=round(processing_time_ms, 1),
        debug=debug_payload,
    )

    if debug and save_debug and debug_payload is not None:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = Path(video_filename or "video").stem.replace(" ", "_")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        debug_path = DEBUG_OUTPUT_DIR / f"{timestamp}__{loaded_model.entry.model_id}__{safe_stem}.json"
        debug_payload.saved_debug_path = str(debug_path)
        debug_path.write_text(json.dumps(response.model_dump(mode="json"), indent=2))
        print(f"[sequence-debug] saved: {debug_path}")

    return response


class Prediction(BaseModel):
    word: str
    confidence: float
    rank: int


class ModelSummary(BaseModel):
    model_id: str
    label: str
    architecture: str
    num_classes: int
    dataset_name: str
    val_top1: float
    experimental: bool
    available_attention: bool


class ModelsResponse(BaseModel):
    default_model_id: str
    models: List[ModelSummary]


class PredictResponse(BaseModel):
    mode: Literal["isolated"]
    model_id: str
    model_label: str
    architecture: str
    num_classes: int
    predictions: List[Prediction]
    detection_rate: float
    frames_processed: int
    attention_weights: Optional[List[float]]
    processing_time_ms: float


class SequenceSegment(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    confidence: float
    windows: int


class SequenceWindowDebug(BaseModel):
    index: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    detection_rate: float
    accepted: bool
    rejection_reason: Optional[str]
    top_predictions: List[Prediction]


class SequenceMergeDebugSegment(BaseModel):
    word: str
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    avg_confidence: float
    windows: int
    selected: bool


class SequenceDebugInfo(BaseModel):
    video_filename: Optional[str]
    total_frames: int
    fps: float
    window_size_frames: int
    model_input_frames: int
    stride_frames: int
    min_detection_rate: float
    confidence_threshold: float
    relaxed_segment_threshold: float
    raw_window_count: int
    accepted_window_count: int
    merged_segment_count: int
    final_segment_count: int
    overall_detection_rate: float
    selection_strategy: str
    fallback_prediction: Optional[Prediction]
    windows: List[SequenceWindowDebug]
    merged_segments: List[SequenceMergeDebugSegment]
    saved_debug_path: Optional[str] = None


class SequencePredictResponse(BaseModel):
    mode: Literal["sequence"]
    model_id: str
    model_label: str
    architecture: str
    num_classes: int
    sequence_text: str
    segments: List[SequenceSegment]
    frames_processed: int
    window_count: int
    avg_detection_rate: float
    processing_time_ms: float
    debug: Optional[SequenceDebugInfo] = None


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    default_model_id: str
    default_model_label: str
    loaded_models: int
    available_models: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    default_entry = MODEL_REGISTRY[DEFAULT_MODEL_ID]
    return HealthResponse(
        status="ok",
        detector_loaded=detector is not None,
        default_model_id=default_entry.model_id,
        default_model_label=default_entry.label,
        loaded_models=len(model_cache),
        available_models=len(MODEL_REGISTRY),
    )


@app.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    models = [
        ModelSummary(
            model_id=entry.model_id,
            label=entry.label,
            architecture=entry.architecture,
            num_classes=entry.num_classes,
            dataset_name=entry.dataset_name,
            val_top1=entry.val_top1,
            experimental=entry.experimental,
            available_attention=entry.architecture == "gru_attention",
        )
        for entry in MODEL_REGISTRY.values()
    ]
    return ModelsResponse(default_model_id=DEFAULT_MODEL_ID, models=models)


@app.get("/classes")
def get_classes(model_id: str = DEFAULT_MODEL_ID) -> Dict[str, Any]:
    loaded_model = get_model(model_id)
    return {
        "model_id": loaded_model.entry.model_id,
        "classes": list(loaded_model.idx_to_gloss.values()),
        "num_classes": len(loaded_model.idx_to_gloss),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    video: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
) -> PredictResponse:
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not loaded")

    loaded_model = get_model(model_id)
    tmp_path = save_upload(video)
    start_time = time.perf_counter()

    try:
        frames, _ = load_video_frames(tmp_path, max_frames=loaded_model.config.get("max_frames", MAX_FRAMES))
        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from the video")

        raw_landmarks = [extract_landmarks_from_frame(frame, detector) for frame in frames]
        normalized, detection_rate = prepare_landmark_tensor(
            raw_landmarks, loaded_model.config.get("max_frames", MAX_FRAMES)
        )
        predictions, attention_weights = run_prediction(
            loaded_model,
            normalized,
            return_attention=loaded_model.supports_attention,
        )
        processing_time_ms = (time.perf_counter() - start_time) * 1000
    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictResponse(
        mode="isolated",
        model_id=loaded_model.entry.model_id,
        model_label=loaded_model.entry.label,
        architecture=loaded_model.entry.architecture,
        num_classes=loaded_model.entry.num_classes,
        predictions=predictions,
        detection_rate=round(detection_rate, 3),
        frames_processed=len(frames),
        attention_weights=attention_weights,
        processing_time_ms=round(processing_time_ms, 1),
    )


@app.post("/predict-sequence", response_model=SequencePredictResponse)
async def predict_sequence(
    video: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    debug: bool = Form(False),
    save_debug: bool = Form(False),
) -> SequencePredictResponse:
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not loaded")

    loaded_model = get_model(model_id)
    tmp_path = save_upload(video)
    start_time = time.perf_counter()

    try:
        frames, fps = load_video_frames(tmp_path, max_frames=None)
        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from the video")

        raw_landmarks = [extract_landmarks_from_frame(frame, detector) for frame in frames]
        if sum(1 for landmarks in raw_landmarks if landmarks is not None) / len(raw_landmarks) < MIN_DETECTION_RATE:
            raise HTTPException(
                status_code=422,
                detail="Hand detection rate is too low for continuous prediction",
            )
        response = build_sequence_response(
            loaded_model=loaded_model,
            raw_landmarks=raw_landmarks,
            fps=fps,
            frames_processed=len(frames),
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            debug=debug or save_debug,
            save_debug=save_debug,
            video_filename=video.filename,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
