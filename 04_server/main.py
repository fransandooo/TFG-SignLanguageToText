"""
FastAPI server for ASL Word Recognition.
Pipeline: video upload → MediaPipe landmarks → GRU+Attention model → top-5 predictions.

Para cambiar de modelo basta con editar MODEL_WEIGHTS y METADATA_PATH.
El servidor lee el config del propio checkpoint y el orden de clases del metadata
del dataset — nada está hardcodeado.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Rutas — lo único que hay que cambiar para usar otro modelo ─────────────────
BASE_DIR = Path(__file__).parent.parent

MODEL_WEIGHTS = BASE_DIR / "05_experiments" / "final_assault" / "best_model_attention.pt"

# metadata.json del dataset con el que se entrenó ese modelo (contiene gloss_to_idx)
METADATA_PATH = BASE_DIR / "06_datasets" / "wlasl" / "landmarks" / "wlasl30" / "metadata.json"

HAND_LANDMARKER_TASK = BASE_DIR / "hand_landmarker.task"

# ── Arquitectura del modelo (copia exacta del notebook 11) ────────────────────
#
# PyTorch guarda en el .pt solo los PESOS (números), no la arquitectura.
# Necesitamos definir la clase aquí para darle a PyTorch el "molde vacío"
# antes de rellenar los pesos con load_state_dict().
# No se entrena nada — es solo la definición de la estructura.

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
        input_size  = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers  = config["num_layers"]
        num_classes = config["num_classes"]
        dropout     = config["dropout"]
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


# ── Pipeline de landmarks (igual que notebook 08) ─────────────────────────────

MAX_FRAMES = 64  # debe coincidir con el entrenamiento

def load_video_frames(video_path: Path, max_frames: int = MAX_FRAMES) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = set(np.linspace(0, total - 1, max_frames, dtype=int)) if total > max_frames else set(range(total))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def extract_landmarks_from_frame(frame: np.ndarray, detector) -> Optional[np.ndarray]:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in lm])
    return None


def interpolate_missing_frames(landmarks_list: List[Optional[np.ndarray]]) -> np.ndarray:
    n = len(landmarks_list)
    result = np.zeros((n, 21, 3))
    valid = [i for i, lm in enumerate(landmarks_list) if lm is not None]
    if not valid:
        return result
    for i in valid:
        result[i] = landmarks_list[i]
    for i in range(n):
        if landmarks_list[i] is not None:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if landmarks_list[j] is not None), None)
        nxt  = next((j for j in range(i + 1, n)     if landmarks_list[j] is not None), None)
        if prev is not None and nxt is not None:
            alpha = (i - prev) / (nxt - prev)
            result[i] = (1 - alpha) * landmarks_list[prev] + alpha * landmarks_list[nxt]
        elif prev is not None:
            result[i] = landmarks_list[prev]
        elif nxt is not None:
            result[i] = landmarks_list[nxt]
    return result


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Centra en muñeca (landmark 0) y escala por distancia muñeca→punta dedo medio (12)."""
    out = np.copy(landmarks)
    for t in range(landmarks.shape[0]):
        frame = landmarks[t]
        centered = frame - frame[0]
        scale = np.linalg.norm(frame[12] - frame[0])
        out[t] = centered / scale if scale > 1e-6 else centered
    return out


# ── Startup de la app ──────────────────────────────────────────────────────────

app = FastAPI(
    title="ASL Word Recognition API",
    description="GRU+Attention sobre landmarks MediaPipe.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cpu")

# Estos se rellenan en startup:
model: Optional[PoseClassifierAttention] = None
detector = None
idx_to_gloss: dict[int, str] = {}
model_config: dict = {}


@app.on_event("startup")
def load_resources():
    global model, detector, idx_to_gloss, model_config

    # 1. Cargar checkpoint — contiene pesos + config + métricas de entrenamiento
    if not MODEL_WEIGHTS.exists():
        raise RuntimeError(f"Pesos no encontrados: {MODEL_WEIGHTS}")

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)
    # El checkpoint tiene: epoch, model_state_dict, val_top1, config
    model_config = checkpoint["config"]

    print(f"Checkpoint: {MODEL_WEIGHTS.name}")
    print(f"  Epoch:    {checkpoint.get('epoch', '?')}")
    print(f"  Val top1: {checkpoint.get('val_top1', '?'):.2f}%")
    print(f"  Config:   {model_config}")

    # 2. Crear el molde vacío con el config del propio checkpoint
    model = PoseClassifierAttention(model_config)

    # 3. Cargar los pesos entrenados en el molde → modelo listo para inferencia
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Cargar el mapping de clases del metadata del dataset
    #    (mismo orden que durante el entrenamiento)
    if not METADATA_PATH.exists():
        raise RuntimeError(f"Metadata no encontrado: {METADATA_PATH}")

    metadata = json.loads(METADATA_PATH.read_text())
    gloss_to_idx: dict[str, int] = metadata["gloss_to_idx"]
    idx_to_gloss = {int(v): k for k, v in gloss_to_idx.items()}
    print(f"  Clases ({len(idx_to_gloss)}): {list(idx_to_gloss.values())[:5]} …")

    # 5. Cargar MediaPipe HandLandmarker
    if not HAND_LANDMARKER_TASK.exists():
        raise RuntimeError(f"hand_landmarker.task no encontrado: {HAND_LANDMARKER_TASK}")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_TASK)),
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    print("MediaPipe HandLandmarker cargado.")


# ── Modelos de respuesta ───────────────────────────────────────────────────────

class Prediction(BaseModel):
    word: str
    confidence: float
    rank: int

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    detection_rate: float
    frames_processed: int
    attention_weights: List[float]
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    val_top1: float
    num_classes: int
    model_params: int
    supported_words: List[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    params = sum(p.numel() for p in model.parameters()) if model else 0
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        checkpoint=MODEL_WEIGHTS.name,
        val_top1=round(model_config.get("val_top1", 0.0), 2) if model else 0.0,
        num_classes=len(idx_to_gloss),
        model_params=params,
        supported_words=list(idx_to_gloss.values()),
    )


@app.get("/classes")
def get_classes():
    return {"classes": list(idx_to_gloss.values()), "num_classes": len(idx_to_gloss)}


@app.post("/predict", response_model=PredictResponse)
async def predict(video: UploadFile = File(...)):
    if model is None or detector is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = Path(tmp.name)

    t_start = time.perf_counter()

    try:
        frames = load_video_frames(tmp_path)
        if not frames:
            raise HTTPException(status_code=422, detail="No se pudieron extraer frames del vídeo")

        raw_landmarks = [extract_landmarks_from_frame(f, detector) for f in frames]
        detected = sum(1 for lm in raw_landmarks if lm is not None)
        detection_rate = detected / len(raw_landmarks)

        if detection_rate < 0.3:
            raise HTTPException(
                status_code=422,
                detail=f"Tasa de detección de mano demasiado baja ({detection_rate:.0%}). Asegúrate de que la mano sea visible.",
            )

        interpolated = interpolate_missing_frames(raw_landmarks)
        normalized   = normalize_landmarks(interpolated)

        T = normalized.shape[0]
        max_frames = model_config.get("max_frames", MAX_FRAMES)
        if T < max_frames:
            pad = np.zeros((max_frames - T, 21, 3))
            normalized = np.concatenate([normalized, pad], axis=0)
        else:
            normalized = normalized[:max_frames]

        seq = torch.from_numpy(
            normalized.reshape(max_frames, 63).astype(np.float32)
        ).unsqueeze(0)  # (1, max_frames, 63)

        with torch.no_grad():
            logits, attn_weights = model(seq, return_attention=True)
            probs = F.softmax(logits, dim=1)[0]

        top5_vals, top5_idx = torch.topk(probs, k=5)
        predictions = [
            Prediction(
                word=idx_to_gloss[idx.item()],
                confidence=round(val.item(), 4),
                rank=rank + 1,
            )
            for rank, (val, idx) in enumerate(zip(top5_vals, top5_idx))
        ]

        attention_weights  = attn_weights[0].tolist()
        processing_time_ms = (time.perf_counter() - t_start) * 1000

    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictResponse(
        predictions=predictions,
        detection_rate=round(detection_rate, 3),
        frames_processed=len(frames),
        attention_weights=[round(w, 5) for w in attention_weights],
        processing_time_ms=round(processing_time_ms, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
