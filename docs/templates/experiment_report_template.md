# Experiment Report: `<title>`

## Metadata

| Field | Value |
|---|---|
| Experiment ID | `<experiment_id>` |
| Date | `<YYYY-MM-DD or ISO-8601 timestamp>` |
| Owner | `<name>` |
| Git commit | `<short or full commit hash>` |
| Status | `<planned / running / completed / archived>` |

## Research Question

`<State the experimental question in one sentence.>`

Example:

Does changing sequence length from 64 to 32 or 96 frames improve WLASL30 recognition accuracy under the same GRU + Temporal Attention architecture?

## Dataset and Split

| Field | Value |
|---|---|
| Dataset | `<WLASL30 / WLASL100 / MS-ASL100 / ...>` |
| Dataset version or path | `<path or version>` |
| Split strategy | `<train/val/test, k-fold, signer-independent, ...>` |
| Number of classes | `<int>` |
| Train samples | `<int or n/a>` |
| Validation samples | `<int or n/a>` |
| Test samples | `<int or n/a>` |

## Input and Preprocessing

| Field | Value |
|---|---|
| Input representation | `<pixels / landmarks / multimodal>` |
| Maximum frames | `<int>` |
| Number of hands | `<1 / 2 / variable>` |
| Landmark features | `<e.g. 21 landmarks x 3 coordinates = 63 features/frame>` |
| Normalization strategy | `<e.g. wrist-centered, wrist-to-middle-finger scale>` |
| Interpolation strategy | `<e.g. linear interpolation, edge fill>` |
| Mean detection rate | `<float or n/a>` |
| Detection rate by split | `<train / val / test values or n/a>` |

## Model and Training

| Field | Value |
|---|---|
| Model architecture | `<architecture name>` |
| Model config | `<short summary or manifest reference>` |
| Training config | `<optimizer, scheduler, batch size, epochs, patience>` |
| Seed | `<int>` |
| Hardware | `<CPU/GPU and memory if known>` |
| Runtime | `<training and/or evaluation runtime>` |

## Metrics

| Metric | Validation | Test | Notes |
|---|---:|---:|---|
| Top-1 accuracy | `<float>` | `<float>` | `<single split / fold mean / ...>` |
| Top-5 accuracy | `<float>` | `<float>` | `<single split / fold mean / ...>` |
| Macro F1 | `<float or n/a>` | `<float or n/a>` | `<path if computed separately>` |
| Balanced accuracy | `<float or n/a>` | `<float or n/a>` | `<path if computed separately>` |

## Artifacts

| Artifact | Path |
|---|---|
| Manifest | `05_experiments/<experiment_name>/experiment_manifest.json` |
| Report | `05_experiments/<experiment_name>/experiment_report.md` |
| Checkpoint | `<path or n/a>` |
| Confusion matrix | `<path or n/a>` |
| Per-class accuracy | `<path or n/a>` |
| Training curves | `<path or n/a>` |
| Raw results | `<path or n/a>` |

## Main Findings

- `<Finding 1>`
- `<Finding 2>`
- `<Finding 3>`

## Limitations

- `<Limitation 1>`
- `<Limitation 2>`

## Next Actions

- `<Next action 1>`
- `<Next action 2>`

## Comparison Notes

Use the same field names for related runs so tables can be built directly from the manifests. For a sequence-length sweep, create one run per length:

- `sequence_length_32`
- `sequence_length_64`
- `sequence_length_96`

Only change `experiment_id`, `title`, `research_question`, `max_frames`, derived runtime, metrics, and artifact paths unless the experimental design requires another controlled change.
