# Sequence Length Sweep Summary

Aggregation across all completed runs in the sweep directory.

## Aggregated by sequence length

| Frames | N seeds | Top-1 mean ± std | Top-5 mean ± std | Macro-F1 mean ± std | Balanced-Acc mean ± std |
|---|---|---|---|---|---|
| 32 | 3 | 56.86 ± 6.12 | 85.29 ± 2.95 | 50.52 ± 8.27 | 64.65 ± 7.48 |
| 64 | 3 | 58.82 ± 2.94 | 78.43 ± 4.49 | 54.88 ± 0.93 | 67.42 ± 0.76 |
| 96 | 3 | 54.90 ± 4.49 | 81.37 ± 4.49 | 49.63 ± 6.33 | 61.36 ± 6.56 |

## Individual runs

| Run | Frames | Seed | Top-1 | Top-5 | Macro-F1 | Bal-Acc | Val Top-1 | Best epoch | Total epochs |
|---|---|---|---|---|---|---|---|---|---|
| wlasl30_seed42_frames32 | 32 | 42 | 61.76 | 88.24 | 55.77 | 69.70 | 66.07 | 58 | 88 |
| wlasl30_seed123_frames32 | 32 | 123 | 50.00 | 85.29 | 40.99 | 56.06 | 64.29 | 29 | 59 |
| wlasl30_seed2026_frames32 | 32 | 2026 | 58.82 | 82.35 | 54.80 | 68.18 | 67.86 | 45 | 75 |
| wlasl30_seed42_frames64 | 64 | 42 | 55.88 | 73.53 | 55.94 | 68.18 | 66.07 | 40 | 70 |
| wlasl30_seed123_frames64 | 64 | 123 | 61.76 | 79.41 | 54.20 | 66.67 | 67.86 | 29 | 59 |
| wlasl30_seed2026_frames64 | 64 | 2026 | 58.82 | 82.35 | 54.49 | 67.42 | 64.29 | 27 | 57 |
| wlasl30_seed42_frames96 | 96 | 42 | 58.82 | 85.29 | 54.86 | 65.15 | 67.86 | 29 | 59 |
| wlasl30_seed123_frames96 | 96 | 123 | 50.00 | 76.47 | 42.59 | 53.79 | 67.86 | 56 | 86 |
| wlasl30_seed2026_frames96 | 96 | 2026 | 55.88 | 82.35 | 51.43 | 65.15 | 69.64 | 84 | 114 |


## Notes

The multi-seed sweep across three sequence lengths (32/64/96) on WLASL30 shows that 64 frames provides the best trade-off between accuracy and stability. While mean Top-1 differences (54.90% - 58.82%) fall within statistical noise given the 34-sample test set, 64 frames achieves the lowest variance across seeds on Macro-F1 (±0.93 vs ±6.33-8.27) and Balanced Accuracy (±0.76 vs ±6.56-7.48), indicating more reproducible performance. This validates the choice of max_frames=64 for downstream experiments.