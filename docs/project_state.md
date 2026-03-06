# Project State

## Current Focus

**Session**: `MS-ASL-Dataset-Preparation` (started 2026-03-06)
**Status**: Starting — prepare MS-ASL as additional dataset per tutor directive (download videos + extract landmarks)
**Previous Session**: `Big-TrainingWASL2000` (COMPLETED — WLASL scaling pipeline built, training runs completed for 100/300/1000 classes)

## Tutor Meeting (2026-02-11) - Key Directives

1. **Add more data** — Scale progressively: 100 words -> 255 words -> 1024 words -> full WLASL2000
2. **More videos per word** — Need 20+ videos per word class for reliable training
3. **SOTA chapter structure** — Hierarchical explanation: CNN -> RNN -> LSTM -> GRU, then landmarks. Focus on THEORY not usage.
4. **Datasets not needed in SOTA** — Only techniques and technologies, no dataset descriptions in SOTA chapter.

## Tutor Meeting (2026-02-06) - Critical Directives (Still Active)

1. **TTA REJECTED** — Tutor (Moises) considers TTA "fake biased information". Must NOT use TTA for final results.
2. **K-Fold / Larger Test Set** — COMPLETED: 5-Fold CV gives 63.9% +/- 4.5% Top-1
3. **Add More Datasets** — WLASL alone is not enough. Find and integrate additional sign language datasets.
4. **Landmarks Confirmed Excellent** — Tutor praised the MediaPipe landmarks approach.
5. **SOTA Chapter** — Write about subsymbolic AI: explain progression from basics to current techniques.
6. **Paper Reference** — "The Three Magician Judge" — to be investigated and referenced.

## Key Direction Change (2026-01-21)

**Tutor meeting outcome**: Shift from letter-by-letter classification to **word-level sequence recognition**.

### Active Approach
- Input: Sequence of frames showing a complete word gesture
- Output: The word directly
- Model learns temporal patterns end-to-end
- MediaPipe landmarks as input (21 x 3 = 63 features/frame)

## Best Results Achieved

```
Model: GRU + Temporal Attention (PoseClassifierAttention)
Input: MediaPipe hand landmarks (21 x 3 = 63 features/frame, 64 frames)
Parameters: 2.26M

K-FOLD VALIDATED (332 samples, 30 classes, 5 folds):
  Top-1: 63.9% +/- 4.5%
  Top-5: 86.5% +/- 2.0%
  Per-fold: [65.7%, 62.7%, 69.7%, 56.1%, 65.2%]

Single split (WLASL30 test, 34 samples):
  Without TTA: Val 64.3% | Test 58.8% | Top-5 91.2%
  With TTA (REJECTED): Test 61.8% | Top-5 91.2%

EFFECTIVE BASELINE: 63.9% K-Fold mean (tutor-approved method)
NEXT GOAL: Scale to more words while maintaining accuracy
```

## Tutor Meeting (2026-01-28) - Target Accuracy

**Target Accuracy (Baseline for good model)**:
- Top-1: 60-70%  -> **ACHIEVED: 63.9% (K-Fold)**
- Top-5: 80-88%  -> **ACHIEVED: 86.5% (K-Fold)**

**Improved targets (stretch)**:
- Top-1: 70-80%  -> Not yet reached
- Top-5: 85-91%  -> **ACHIEVED: 86.5%**

## Recent Decisions

- **2026-02-18**: NEW SESSION `Big-TrainingWASL2000` — scale up training to full WLASL
- **2026-02-11**: Tutor directive: more data (100->255->1024 words), SOTA chapter structure
- **2026-02-10**: K-Fold validation completed: 63.9% +/- 4.5% Top-1 (robust, no TTA)
- **2026-02-10**: Previous session COMPLETED. K-Fold replaces single-split evaluation.
- **2026-02-06**: TUTOR REJECTED TTA — must use K-Fold cross-validation instead
- **2026-02-06**: TUTOR REQUESTED more datasets beyond WLASL
- **2026-02-03**: Pivoted from pixel-based CNN+LSTM to MediaPipe landmarks (2389x data reduction)
- **2026-02-03**: GRU + Temporal Attention identified as champion architecture
- **2026-02-03**: Transformer underperformed (50% vs 58.8%) due to insufficient data
- **2026-01-22**: Use 30 words (WLASL30) instead of 100 for initial experiments

## Model Evolution

| Model | Test Top-1 | Test Top-5 | Params | Notes |
|-------|-----------|------------|--------|-------|
| CNN+LSTM (pixels) | 10.4% | 27.8% | 14.3M | Abandoned |
| LSTM (landmarks) | 41.2% | 67.6% | 2.8M | Baseline |
| GRU soft aug | 38.2% | 76.5% | 2.1M | |
| GRU Enriched | 47.1% | 79.4% | 2.1M | |
| **GRU Attention** | **58.8%** | **91.2%** | **2.3M** | Single split, 30 classes |
| Transformer | 50.0% | 73.5% | 3.2M | Insufficient data |
| **GRU Attention K-Fold** | **63.9% +/- 4.5%** | **86.5% +/- 2.0%** | **2.3M** | **5-fold, 332 samples, 30 classes** |
| GRU Attention (WLASL100) | 44.7% | 70.5% | 2.3M | 100 classes, 1113 samples |
| GRU Attention (WLASL300) | 28.6% | 59.7% | 2.3M | 300 classes, 2808 samples |
| GRU Attention (WLASL1000) | 11.2% | 31.2% | 2.3M | 1000 classes, 7389 samples |

## Next Steps

### This Session (MS-ASL-Dataset-Preparation)
1. **Download MS-ASL videos** — Use URLs from MSASL_train/val/test.json (YouTube clips with timestamps)
2. **Extract MediaPipe landmarks for MS-ASL** — Same pipeline as WLASL (Nb 15 pattern)
3. **Create MS-ASL subset JSONs** — MS-ASL100/200/500/1000 subsets (matching official paper splits)
4. **Train on MS-ASL** — Use same GRU+Attention architecture for cross-dataset comparison
5. **Combine WLASL + MS-ASL** — Merge datasets on shared glosses for richer training

### Thesis Writing
6. Write SOTA chapter: CNN -> RNN -> LSTM -> GRU hierarchy (theory focus, no dataset descriptions)
7. Write experiments chapter: scaling results table (30->100->300->1000 classes)
8. Investigate "Three Magician Judge" paper

### Later (if accuracy improves sufficiently)
9. Integrate GRU_Attention into FastAPI backend
10. Replace YOLO pipeline with MediaPipe landmarks
11. Update Flutter app for landmark-based inference

## Dataset Status

### WLASL30 Landmarks (Previous experiments)
```
06_datasets/wlasl/landmarks/wlasl30/
├── train/      242 videos (30 words)
├── val/        56 videos
├── test/       34 videos
└── metadata.json
```

### WLASL100 Landmarks (Processed, available)
```
06_datasets/wlasl/landmarks/wlasl100/
├── train/      741 videos (100 words)
├── val/        162 videos
├── test/       99 videos
└── metadata.json
```

### WLASL100 Filtered to 30 classes (Used for K-Fold)
```
06_datasets/wlasl/landmarks/wlasl100_filtered30/
├── train/      242 videos (same 30 words, more clips)
├── val/        56 videos
└── test/       (uses wlasl30 test)
Total: 332 videos used in K-Fold
```

### WLASL2000 (TO BE PROCESSED)
```
Target: Full WLASL dataset
Expected: ~21,000 videos across 2000 ASL word glosses
Progressive targets: 100 -> 255 -> 1024 -> 2000 classes
```

## K-Fold Validation Results (Notebook 13)

```
Stratified 5-Fold Cross-Validation
Dataset: WLASL100_filtered30 (332 samples, 30 classes)
Model: GRU + Temporal Attention (2.26M params)

Fold  Top-1    Top-5    Best Epoch
1     65.7%    85.1%    73
2     62.7%    83.6%    22
3     69.7%    87.9%    31
4     56.1%    86.4%    37
5     65.2%    89.4%    60

MEAN  63.9%    86.5%
STD    4.5%     2.0%

Best per-class: orange (100%), table (100%), help (92.9%)
Worst per-class: what (8.3%), cool (12.5%), fine (22.2%)
```

## Artifacts Location

| Artifact | Location |
|----------|----------|
| Champion model (GRU_Attention) | `05_experiments/final_assault/best_model_attention.pt` |
| K-Fold models (5 folds) | `05_experiments/kfold_wlasl100/fold_{0-4}/best_model.pt` |
| K-Fold results | `05_experiments/kfold_wlasl100/kfold_results.json` |
| Transformer model | `05_experiments/final_assault/best_model_transformer.pt` |
| Self-ensemble models | `05_experiments/final_push_60/gru_attention_seed{123,999}.pt` |
| K-Fold validation | `01_notebooks/word_recognition/13_kfold_validation.ipynb` |
| TTA + Attention + Transformer | `01_notebooks/word_recognition/11_tta_attention_transformer.ipynb` |
| Final push (all strategies) | `01_notebooks/word_recognition/12_final_push_60.ipynb` |
| MediaPipe preprocessing | `01_notebooks/word_recognition/08_mediapipe_preprocessing.ipynb` |

## Risks & Open Questions

- **Scaling to 2000 words**: Accuracy will likely drop significantly with more classes — need enough data per class
- **WLASL video availability**: Many WLASL videos are from YouTube and may be taken down
- **Finding compatible datasets**: Need word-level sign language datasets with video (not just images)
- **"Three Magician Judge" paper**: Unknown reference from tutor — needs research
- **MediaPipe detection rate ~69%**: Frames without detection are interpolated
- **Transformer viability**: May become viable with more data (underperformed at 30 classes)

## Completed Tasks

- [x] YOLO hand detection training (YOLOv8n)
- [x] SSDLite-MobileNetV3 training for comparison
- [x] Documentation structure setup
- [x] WLASL dataset research and selection
- [x] WLASL dataset download from Kaggle
- [x] WLASL100 processing (frames extracted)
- [x] WLASL30 filtering and preparation
- [x] DataLoader implementation (WLASLDataset, collate_fn)
- [x] CNN+LSTM baseline model implementation
- [x] Word accuracy metrics (Top-1, Top-5, confusion matrix)
- [x] First training run completed
- [x] Memory optimization: Chunking + AMP (2026-01-30)
- [x] Variability analysis notebook (2026-01-30)
- [x] ASL-safe augmentation pipeline (2026-01-30)
- [x] **MediaPipe landmark extraction pipeline** (2026-02-03)
- [x] **WLASL30 + WLASL100 landmark preprocessing** (2026-02-03)
- [x] **LSTM baseline on landmarks (41.2% Test)** (2026-02-03)
- [x] **GRU with soft augmentation (53.6% Val)** (2026-02-03)
- [x] **GRU + Temporal Attention (58.8% Test, 91.2% Top-5)** (2026-02-03)
- [x] **Transformer comparison (50% - confirmed insufficient data)** (2026-02-03)
- [x] **Weighted Ensemble + TTA -> 61.8% Test Top-1** (2026-02-10)
- [x] **K-Fold validation: 63.9% +/- 4.5% Top-1** (2026-02-10)
- [x] **OBJECTIVE ACHIEVED: 63.9% K-Fold Top-1, 86.5% Top-5** (2026-02-10)
- [x] **WLASL2000 video audit** — 11,980 downloaded, ~56% coverage (2026-02-19)
- [x] **WLASL2000 landmark extraction** — 11,844 .npy files, 330 MB, 98.8% success (2026-02-19)
- [x] **Dataset subset JSONs** — wlasl_100/300/1000/clean_full created (2026-02-19)
- [x] **Progressive training notebooks 17a-17d** — WLASL 100/300/1000/1544 classes (2026-02-19)
- [x] **Training runs completed**: WLASL100: 44.7% test, WLASL300: 28.6%, WLASL1000: 11.2% (2026-02-19)
- [x] **MS-ASL JSON annotations downloaded** — 1000 classes, 16K train + 5K val + 4K test (available)

## Parked Tasks

- Warning margin ablation study (old approach)
- Frame difference heatmaps (may revisit)
- Letter-level classification (abandoned)
- Transformer architecture (revisit with more data at larger scale)
