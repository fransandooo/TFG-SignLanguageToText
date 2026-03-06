# TFG — Sign Language to Text

**Trabajo de Fin de Grado** | Universidad Francisco de Vitoria
**Autor**: Francisco Sandoval Hernández
**Área**: Deep Learning · Computer Vision · Sign Language Recognition

---

## Overview

End-to-end system for real-time **American Sign Language (ASL) word recognition**. The system takes a video sequence of a sign and outputs the corresponding word directly, without intermediate letter classification.

The core pipeline combines **MediaPipe hand landmark extraction** with a **GRU + Temporal Attention** model, achieving **63.9% Top-1 / 86.5% Top-5** accuracy (5-fold cross-validation, 30 ASL words).

---

## Architecture

```
Camera / Video
      │
      ▼
MediaPipe Holistic
  (hand landmarks)
      │  21 keypoints × 3 coords = 63 features/frame
      ▼
Sequence (64 frames × 63 features)
      │
      ▼
BiGRU + Temporal Attention
  (2.3M parameters)
      │
      ▼
Word prediction (Top-1 / Top-5)
```

The landmark-based approach reduces input dimensionality by ~2389x compared to raw pixels while being invariant to lighting, skin tone, and background.

---

## Results

| Model | Classes | Test Top-1 | Test Top-5 | Notes |
|-------|---------|-----------|-----------|-------|
| CNN+LSTM (pixels) | 30 | 10.4% | 27.8% | Abandoned |
| LSTM (landmarks) | 30 | 41.2% | 67.6% | Baseline |
| GRU + soft aug | 30 | 38.2% | 76.5% | |
| GRU + Enriched | 30 | 47.1% | 79.4% | |
| **GRU + Attention** | **30** | **63.9% ±4.5%** | **86.5% ±2.0%** | **5-fold CV — champion** |
| GRU + Attention | 100 | 44.7% | 70.5% | Scaling experiment |
| GRU + Attention | 300 | 28.6% | 59.7% | Scaling experiment |
| GRU + Attention | 1000 | 11.2% | 31.2% | Scaling experiment |

Hand detection (YOLO, earlier phase):
- mAP50: ~98% | Mean IoU: ~87%

---

## Project Structure

```
.
├── 01_notebooks/
│   ├── dataset_preparation/   # Dataset creation and processing
│   ├── hand_detection/        # YOLO hand detector training
│   └── word_recognition/      # MediaPipe + GRU pipeline (main work)
│       ├── 07_mediapipe_exploration.ipynb
│       ├── 08_mediapipe_preprocessing.ipynb
│       ├── 09_lstm_landmarks_training.ipynb
│       ├── 11_tta_attention_transformer.ipynb
│       ├── 13_kfold_validation.ipynb
│       ├── 15_wlasl2000_landmark_extraction.ipynb
│       ├── 17a_train_wlasl100.ipynb
│       └── ...
├── 02_models/
│   ├── yolov8n.pt             # Pre-trained YOLOv8 Nano (hand detection)
│   └── yolo11n.pt             # Pre-trained YOLO11 Nano
├── 03_app/
│   └── signtotext/            # Flutter mobile app
├── 04_server/
│   └── main.py                # FastAPI backend (YOLO inference)
├── 05_experiments/            # Training runs, metrics, plots
├── 06_datasets/               # Processed datasets (not in git — see below)
├── 07_docs/
│   └── PFG_latex/             # LaTeX thesis source
└── docs/
    └── project_state.md       # Development progress log
```

---

## Datasets

| Dataset | Type | Classes | Videos |
|---------|------|---------|--------|
| [WLASL](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) | Word-level ASL | 2000 | ~21,000 |
| [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/) | Word-level ASL | 1000 | ~25,000 |
| [EgoHands](http://vision.soic.indiana.edu/projects/egohands/) | Hand detection | — | egocentric |
| [Kaggle Hand Dataset](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset) | Hand detection | — | multi-view |

Datasets are not included in the repository. See `DATASETS_REFERENCES.txt` for download instructions.

---

## Setup

### Requirements

- Python 3.10+
- Flutter 3.10+ (only for the mobile app)
- CUDA-capable GPU recommended for training

### Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install mediapipe ultralytics
pip install jupyter notebook
pip install opencv-python pandas numpy matplotlib scikit-learn
```

### Backend server

```bash
cd 04_server
pip install -r requirements.txt
python main.py
# API available at localhost:8000 — docs at /docs
```

### Flutter app

```bash
cd 03_app/signtotext
flutter pub get
flutter run
```

---

## Reproducing the Main Pipeline

The word recognition pipeline is fully documented in `01_notebooks/word_recognition/`. Run notebooks in order:

1. **`08_mediapipe_preprocessing.ipynb`** — extract hand landmarks from WLASL videos
2. **`09_lstm_landmarks_training.ipynb`** — LSTM baseline on landmarks
3. **`11_tta_attention_transformer.ipynb`** — GRU + Temporal Attention (champion model)
4. **`13_kfold_validation.ipynb`** — 5-fold cross-validation (final evaluation)
5. **`15_wlasl2000_landmark_extraction.ipynb`** — scale to full WLASL2000
6. **`17a_train_wlasl100.ipynb`** — scaling experiment (100 classes)

For dataset download and processing, start from `14_wlasl2000_video_audit.ipynb`.

---

## Model Architecture Details

```python
class PoseClassifierAttention(nn.Module):
    # Input:  (batch, 64, 63)  — 64 frames, 21 landmarks × 3 coords
    # GRU:    bidirectional, hidden=256, layers=2, dropout=0.4
    # Attention: temporal self-attention over GRU outputs
    # Output: (batch, num_classes)

    # Optimizer: AdamW, lr=1e-3
    # Scheduler: CosineAnnealingLR
    # Early stopping: patience=30
```

Parameters: **2.26M**

---

## Hand Detection (Earlier Phase)

The initial project phase focused on YOLO-based hand detection before pivoting to end-to-end word recognition:

- Trained YOLOv8n and YOLO11n on EgoHands + Kaggle Hand Dataset
- mAP50 ~98%, Mean IoU ~87%
- Notebooks in `01_notebooks/hand_detection/`
- Results in `05_experiments/hand_bbox_baseline/`
