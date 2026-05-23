# SignToText FDP - Sign Language Recognition from Video

**Final Degree Project (FDP)** - Universidad Francisco de Vitoria
**Author:** Francisco Sandoval Hernandez
**Area:** Deep Learning, Computer Vision, Sign Language Recognition

This repository contains the research code, experiments, mobile prototype and written
documentation for an FDP on isolated **American Sign Language (ASL) word recognition** from short
video clips.

The project does not aim to deliver a production-ready sign language translator. Its goal is to
study whether compact hand-landmark sequences can support reproducible word-level ASL recognition
with neural temporal models. The final pipeline extracts 3D hand landmarks with MediaPipe Hands,
normalises them into fixed-length sequences and classifies each clip with a bidirectional GRU plus
temporal attention.

## What This Repository Contains

The repository is organised around the full project lifecycle:

- exploratory notebooks for dataset preparation, hand detection and word recognition;
- a reusable Python package under `src/slr/` for datasets, models, training, evaluation and
  experiment tracking;
- YAML configurations for reproducible experiment runs;
- stored experiment outputs, metrics, plots and manifests;
- a FastAPI inference server and an Expo/React Native mobile proof of concept;
- the LaTeX source of the FDP report.

Large datasets, downloaded videos and most trained checkpoints are intentionally not included in
Git. The repository keeps curated metrics and figures so that the results can still be inspected
without the full local data directory.

## Final Recognition Pipeline

```text
Short ASL video clip
        |
        v
Frame decoding with OpenCV
        |
        v
MediaPipe Hands landmark extraction
21 landmarks x 3 coordinates per valid frame
        |
        v
Normalisation and missing-frame handling
64 frames x 63 features
        |
        v
Bidirectional GRU + Temporal Attention
        |
        v
Top-K word-gloss predictions
```

The landmark representation reduces a `224 x 224 x 3` RGB frame to 63 floating-point coordinates,
which is roughly a 2,389x dimensionality reduction. This removes most background, lighting and
skin-tone information before training, at the cost of depending on reliable landmark extraction.

## Headline Results

| Experiment | Dataset | Classes | Protocol | Top-1 | Top-5 |
| --- | --- | ---: | --- | ---: | ---: |
| GRU + Attention | WLASL30 | 30 | Stratified 5-fold CV | **63.85 +/- 4.50%** | **86.46 +/- 2.04%** |
| GRU + Attention | WLASL30 | 30 | 3 seeds, fixed split | 58.82 +/- 2.94% | 78.43 +/- 4.49% |
| GRU + Attention | MS-ASL50 | 50 | 3 seeds, fixed split | 70.20 +/- 2.89% | 90.54 +/- 0.89% |

Architecture selection on the fixed WLASL30 split:

| Model | Classes | Top-1 | Top-5 | Notes |
| --- | ---: | ---: | ---: | --- |
| LSTM landmarks | 30 | 41.18% | 67.65% | Bidirectional recurrent baseline |
| GRU landmarks | 30 | 38.24% | 76.47% | Bidirectional recurrent baseline |
| GRU + enriched features | 30 | 47.06% | 79.41% | Bone vectors and joint angles |
| **GRU + Attention** | **30** | **58.82%** | **91.18%** | Selected architecture |
| Transformer | 30 | 50.00% | 73.53% | Self-attention baseline |

Vocabulary scaling on WLASL shows the main limitation of the approach: performance decreases as
the number of classes grows and the number of samples per class falls.

| WLASL vocabulary | Top-1 | Top-5 |
| ---: | ---: | ---: |
| 100 | 44.70% | 70.45% |
| 300 | 28.61% | 59.72% |
| 500 | 23.71% | 49.55% |
| 1,000 | 11.19% | 31.24% |
| 1,500 | 9.76% | 28.23% |

Earlier exploratory work also trained a YOLOv8n hand detector on EgoHands + Kaggle Hand Dataset,
reaching approximately 98% mAP@0.5 and 87% mean IoU. That phase is documented because it motivated
the methodological pivot from hand bounding boxes and static letters to landmark-based word
recognition.

## Repository Layout

```text
.
|-- 01_notebooks/
|   |-- dataset_preparation/       # Dataset source notes and preparation notebooks
|   |-- hand_detection/            # Early YOLO hand-localisation experiments
|   `-- word_recognition/          # Main exploratory ASL recognition notebooks
|-- 02_models/                     # Small pretrained YOLO base weights kept in Git
|-- 03_app/
|   `-- signtotext-expo/           # Expo / React Native mobile prototype
|-- 04_server/                     # FastAPI inference server
|-- 05_experiments/                # Curated metrics, plots, manifests and reports
|-- 06_datasets/                   # Local datasets and landmarks, ignored by Git
|-- 07_docs/
|   `-- PFG_latex/                 # FDP LaTeX report source and generated PDF
|-- configs/                       # Base and experiment-specific YAML configs
|-- docs/templates/                # Manifest schema and experiment report template
|-- scripts/                       # Training, evaluation, extraction and audit utilities
|-- src/slr/                       # Reusable Python package
`-- tests/                         # Unit tests for data, manifests and metrics
```

Key Python modules:

- `src/slr/data/`: landmark datasets, padding/truncation, splits and preprocessing utilities.
- `src/slr/models/`: `gru_baseline`, `gru_attention` and `transformer` model definitions.
- `src/slr/training/`: trainer, losses, checkpointing and early stopping.
- `src/slr/evaluation/`: Top-K metrics, Macro-F1, balanced accuracy and confusion matrices.
- `src/slr/tracking/`: experiment manifests with git, hardware, config and metric metadata.

## Datasets

| Dataset | Task | Used for | Notes |
| --- | --- | --- | --- |
| WLASL | Word-level ASL video recognition | Main benchmark, WLASL30 and scaling subsets | 2,000 glosses, about 21k clips |
| MS-ASL | Word-level ASL video recognition | Cross-dataset validation, MS-ASL50 | YouTube attrition affects reproducibility |
| EgoHands | Hand detection | Early YOLO phase | Egocentric hand images |
| Kaggle Hand Dataset | Hand detection | Early YOLO phase | Third-person hand images |
| Sign Language MNIST | Static ASL letters | Early pilot | Superseded by word-level recognition |
| ASL Alphabet | Static ASL letters | Early pilot | Superseded by word-level recognition |

Datasets are not committed because of size and licensing constraints. Local expected paths follow
the configs, for example:

```text
06_datasets/wlasl/landmarks/wlasl100_filtered30/
06_datasets/MS-ASL/subsets/msasl50/
```

Dataset source notes are in `01_notebooks/dataset_preparation/dataset_references.md`.

## Installation

Python 3.10+ is required. A CUDA-capable GPU is recommended for training, although evaluation and
small tests can run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[landmarks,dev]"
```

If you only need to inspect stored results, the datasets are not required. To run landmark
extraction, install the `landmarks` extra as shown above. To run the FastAPI server, also install
its runtime dependencies:

```bash
pip install -r 04_server/requirements.txt
```

## Running Tests

```bash
pytest
```

The tests cover dataset loading, manifest generation and metric computation. They do not require
the full video datasets.

## Reproducing Training Runs

Training is driven by YAML files under `configs/`. Each run resolves a base config, trains a model,
evaluates the best checkpoint and writes artifacts to `05_experiments/`.

```bash
python scripts/train.py --config configs/experiments/wlasl30_baseline.yaml
```

Examples:

```bash
python scripts/train.py --config configs/experiments/msasl50_baseline/seed42.yaml
python scripts/train.py --config configs/experiments/msasl50_baseline/seed123.yaml
python scripts/train.py --config configs/experiments/msasl50_baseline/seed2026.yaml
```

Sequence-length sweep configs are available in:

```text
configs/experiments/sweep_maxframes/
```

Every completed run writes a self-contained experiment directory with:

- `best_model.pt` when the checkpoint is available locally;
- `training_history.json`;
- `results.json`;
- `per_class_metrics.csv`;
- confusion matrices;
- `experiment_manifest.json`.

The manifest is the main audit artifact. It records the resolved configuration, dataset metadata,
hardware, git state, runtime and final metrics.

## Evaluating a Checkpoint

Use `scripts/evaluate.py` to evaluate a trained checkpoint on a split directory:

```bash
python scripts/evaluate.py \
  --checkpoint 05_experiments/final_assault/best_model_attention.pt \
  --metadata 06_datasets/wlasl/landmarks/wlasl30/metadata.json \
  --dataset-split 06_datasets/wlasl/landmarks/wlasl30/test \
  --output-dir 05_experiments/final_assault/eval_test \
  --architecture gru_attention
```

This produces Top-1, Top-5, Macro-F1, balanced accuracy, per-class metrics and confusion matrices.

## Inspecting Existing Results

Useful stored outputs:

- `05_experiments/final_assault/experiment_report.md`: compact report for the selected WLASL30
  comparison.
- `05_experiments/sequence_length_sweep/sweep_summary.md`: sequence-length sweep summary.
- `05_experiments/msasl50_baseline/summary.md`: MS-ASL50 cross-dataset validation.
- `05_experiments/gap_bottleneck_analysis/`: scaling and data-starvation analysis figures.
- `07_docs/PFG_latex/chapters/experiments.tex`: full experimental discussion in the FDP report.

The main FDP document is:

```text
07_docs/PFG_latex/main.pdf
```

## Running the Inference Prototype

The runtime proof of concept has two parts: a FastAPI server and an Expo mobile app.

### FastAPI server

```bash
cd 04_server
pip install -r requirements.txt
python main.py
```

The server exposes:

- `GET /`: basic server status;
- `GET /models`: available model registry entries;
- `POST /predict`: isolated-word prediction from an uploaded video;
- `POST /predict-sequence`: heuristic continuous-sequence prediction.

The server expects local checkpoint and metadata paths declared in `04_server/main.py`. Some
checkpoints and the `06_datasets/` directory are ignored by Git, so a fresh clone may need those
artifacts restored locally before inference works.

### Expo mobile app

```bash
cd 03_app/signtotext-expo
pnpm install
pnpm start
```

The app records or selects a short video, sends it to the FastAPI server, displays Top-K
predictions and can read the Top-1 result through text-to-speech.

## FDP Documentation

The written report is kept in `07_docs/PFG_latex/`. The most relevant chapters for repository
review are:

- `chapters/introduction.tex`: project scope and research question;
- `chapters/objectives.tex`: main and specific objectives;
- `chapters/development.tex`: system architecture, requirements and datasets;
- `chapters/experiments.tex`: methodology, results, scaling and limitations;
- `chapters/ethical_evaluation.tex`: accessibility, privacy, bias and responsible deployment;
- `chapters/conclusions.tex`: final conclusions and future work.

## Known Limitations

- The system recognises isolated ASL word glosses, not continuous sign language translation.
- MediaPipe failures on blurred, occluded or out-of-frame hands reduce the quality of landmark
  sequences.
- WLASL30 has a small fixed test split, so single errors move Top-1 by several percentage points.
- MS-ASL reconstruction depends on YouTube availability; the local subset may differ from the
  original publication state.
- Scaling beyond small vocabularies is limited by the number of clips per class.
- The mobile/server prototype is a proof of concept and should not be treated as an assistive
  product ready for deployment.

## Suggested Review Path

For a quick assessment of the project:

1. Read this README to understand the repository.
2. Open `07_docs/PFG_latex/main.pdf` for the full FDP report.
3. Inspect `05_experiments/*/experiment_manifest.json` files to audit reproducibility.
4. Review `src/slr/` for the reusable implementation.
5. Use `configs/experiments/` and `scripts/train.py` to reproduce selected runs if the local
   datasets are available.
