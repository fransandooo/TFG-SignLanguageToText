# Experiment Report: Production GRU Attention Baseline for WLASL30

## Metadata

| Field | Value |
|---|---|
| Experiment ID | `attention_30_prod` |
| Date | 2026-02-10 |
| Owner | Francisco Sandoval Hernandez |
| Git commit | `36de93f` |
| Status | Completed |

## Research Question

Can a bidirectional GRU with temporal attention classify isolated WLASL30 signs from one-hand MediaPipe landmarks with thesis-ready accuracy?

## Dataset and Split

| Field | Value |
|---|---|
| Dataset | WLASL30 |
| Dataset version or path | `06_datasets/wlasl/landmarks/wlasl30/` |
| Split strategy | Standard train/validation/test split |
| Number of classes | 30 |
| Train samples | 242 |
| Validation samples | 56 |
| Test samples | 34 |

## Input and Preprocessing

| Field | Value |
|---|---|
| Input representation | MediaPipe hand landmarks flattened per frame |
| Maximum frames | 64 |
| Number of hands | 1 |
| Landmark features | 21 landmarks x 3 coordinates = 63 features/frame |
| Normalization strategy | Wrist-centered coordinates scaled by wrist-to-middle-finger distance |
| Interpolation strategy | Linear interpolation between valid frames with edge fill at sequence boundaries |
| Mean detection rate | Not recorded for this run |
| Detection rate by split | Not recorded for this run |

## Model and Training

| Field | Value |
|---|---|
| Model architecture | `PoseClassifierAttention` (bidirectional GRU + temporal attention) |
| Hidden size | 256 |
| GRU layers | 2 |
| Dropout | 0.4 |
| Bidirectional | true |
| Batch size | 32 |
| Max epochs | 150 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0005 |
| Label smoothing | 0.1 |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 30 |
| Seed | 42 |
| Hardware | NVIDIA RTX 3050 Ti local workstation |
| Runtime | Not recorded in `final_results.json` |

## Metrics

| Metric | Validation | Test | Notes |
|---|---:|---:|---|
| Top-1 accuracy | 64.29 | 58.82 | Single split |
| Top-5 accuracy | 83.93 | 91.18 | Single split |
| Macro F1 | n/a | n/a | Not computed in this run |
| Balanced accuracy | n/a | n/a | Not computed in this run |

The stronger thesis reference is the later five-fold validation on WLASL100 filtered to the same 30 classes:

| Metric | Mean | Std |
|---|---:|---:|
| Top-1 accuracy | 63.85 | 4.50 |
| Top-5 accuracy | 86.46 | 2.04 |

## Artifacts

| Artifact | Path |
|---|---|
| Manifest | `05_experiments/final_assault/experiment_manifest.json` |
| Report | `05_experiments/final_assault/experiment_report.md` |
| Checkpoint | `05_experiments/final_assault/best_model_attention.pt` |
| Raw results | `05_experiments/final_assault/final_results.json` |
| Confusion matrix | n/a |
| Per-class accuracy | n/a |
| Training curves | n/a |

## Main Findings

- GRU + Temporal Attention is the strongest individual model in the final assault comparison.
- The model reaches 58.82% Top-1 and 91.18% Top-5 on the WLASL30 test split.
- The five-fold evaluation confirms that the architecture is a robust baseline for the 30-class setting, with 63.85% Top-1 mean accuracy.

## Limitations

- The single test split contains only 34 samples, so the test result has high variance.
- Macro F1 and balanced accuracy were not computed for this run.
- Detection-rate statistics were not saved in the original experiment artifacts.
- The representation uses one hand only, which can miss two-hand and non-manual sign information.

## Next Actions

- Use this report and manifest structure for the 32/64/96 frame-length sweep.
- Add normalized confusion matrices and per-class accuracy files for the final models.
- Record detection-rate statistics in future landmark extraction and evaluation runs.
