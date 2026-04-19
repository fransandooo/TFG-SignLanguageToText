#!/usr/bin/env python3
"""Aggregate experiment manifests across a sweep directory and produce
a comparative summary (CSV + Markdown + PNG).

Intended for the sequence-length sweep, where each run directory follows
the pattern ``wlasl30_seed{SEED}_frames{FRAMES}`` and contains an
``experiment_manifest.json``. Robust to missing or partial runs: any
manifest that cannot be parsed is skipped with a warning, and summary
statistics use only the seeds that actually completed.

Usage:
    python scripts/summarize_sweep.py \\
        --sweep-dir 05_experiments/sequence_length_sweep \\
        --output-dir 05_experiments/sequence_length_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


RUN_PATTERN = re.compile(r"^wlasl30_seed(?P<seed>\d+)_frames(?P<frames>\d+)$")
METRIC_KEYS = ("top1", "top5", "macro_f1", "balanced_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True,
                        help="Directory containing one subdir per run")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write summary artifacts (defaults to --sweep-dir)")
    parser.add_argument("--run-pattern", default=r"^wlasl30_seed(?P<seed>\d+)_frames(?P<frames>\d+)$",
                        help="Regex with named groups 'seed' and 'frames'")
    return parser.parse_args()


def collect_runs(sweep_dir: Path, pattern: re.Pattern) -> list[dict[str, Any]]:
    """Walk *sweep_dir*, parse manifests, return one record per valid run.

    Missing manifests or runs whose directory name does not match
    *pattern* are skipped with a warning.
    """
    records: list[dict[str, Any]] = []
    for entry in sorted(sweep_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if not match:
            print(f"[skip] {entry.name}: does not match run pattern")
            continue
        manifest_path = entry / "experiment_manifest.json"
        if not manifest_path.exists():
            print(f"[skip] {entry.name}: experiment_manifest.json missing")
            continue
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"[skip] {entry.name}: invalid JSON ({exc})")
            continue

        record = {
            "run_id": entry.name,
            "seed": int(match.group("seed")),
            "frames": int(match.group("frames")),
            "top1": manifest.get("top1"),
            "top5": manifest.get("top5"),
            "macro_f1": manifest.get("macro_f1"),
            "balanced_accuracy": manifest.get("balanced_accuracy"),
            "val_top1": manifest.get("metrics", {}).get("val_top1"),
            "best_epoch": manifest.get("runtime", {}).get("best_epoch"),
            "total_epochs": manifest.get("runtime", {}).get("total_epochs"),
            "training_time_s": manifest.get("runtime", {}).get("training_time_s"),
        }
        records.append(record)
    return records


def aggregate_by_frames(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Group records by frame length and compute mean/std/count for each metric."""
    by_frames: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_frames.setdefault(rec["frames"], []).append(rec)

    aggregated: dict[int, dict[str, Any]] = {}
    for frames, recs in sorted(by_frames.items()):
        agg: dict[str, Any] = {"frames": frames, "n_seeds": len(recs)}
        agg["seeds"] = sorted(r["seed"] for r in recs)
        for metric in METRIC_KEYS:
            values = [r[metric] for r in recs if r[metric] is not None]
            if values:
                agg[f"{metric}_mean"] = mean(values)
                agg[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
                agg[f"{metric}_n"] = len(values)
            else:
                agg[f"{metric}_mean"] = None
                agg[f"{metric}_std"] = None
                agg[f"{metric}_n"] = 0
        aggregated[frames] = agg
    return aggregated


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_id", "frames", "seed",
        "top1", "top5", "macro_f1", "balanced_accuracy",
        "val_top1", "best_epoch", "total_epochs", "training_time_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["frames"], r["seed"])):
            writer.writerow({k: rec.get(k) for k in fieldnames})


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_markdown(records: list[dict[str, Any]],
                   aggregated: dict[int, dict[str, Any]],
                   path: Path) -> None:
    lines: list[str] = []
    lines.append("# Sequence Length Sweep Summary\n")
    lines.append("Aggregation across all completed runs in the sweep directory.\n")

    lines.append("## Aggregated by sequence length\n")
    lines.append("| Frames | N seeds | Top-1 mean ± std | Top-5 mean ± std | Macro-F1 mean ± std | Balanced-Acc mean ± std |")
    lines.append("|---|---|---|---|---|---|")
    for frames, agg in aggregated.items():
        def cell(metric: str) -> str:
            m = agg[f"{metric}_mean"]
            s = agg[f"{metric}_std"]
            if m is None:
                return "-"
            return f"{m:.2f} ± {s:.2f}"
        lines.append(
            f"| {frames} | {agg['n_seeds']} | "
            f"{cell('top1')} | {cell('top5')} | "
            f"{cell('macro_f1')} | {cell('balanced_accuracy')} |"
        )
    lines.append("")

    lines.append("## Individual runs\n")
    lines.append("| Run | Frames | Seed | Top-1 | Top-5 | Macro-F1 | Bal-Acc | Val Top-1 | Best epoch | Total epochs |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for rec in sorted(records, key=lambda r: (r["frames"], r["seed"])):
        lines.append(
            f"| {rec['run_id']} | {rec['frames']} | {rec['seed']} | "
            f"{_fmt(rec['top1'])} | {_fmt(rec['top5'])} | "
            f"{_fmt(rec['macro_f1'])} | {_fmt(rec['balanced_accuracy'])} | "
            f"{_fmt(rec['val_top1'])} | "
            f"{rec['best_epoch'] if rec['best_epoch'] is not None else '-'} | "
            f"{rec['total_epochs'] if rec['total_epochs'] is not None else '-'} |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def plot_comparison(aggregated: dict[int, dict[str, Any]], path: Path) -> None:
    if not aggregated:
        print("[warn] no runs to plot")
        return

    frames = sorted(aggregated.keys())
    top1_mean = [aggregated[f]["top1_mean"] for f in frames]
    top1_std = [aggregated[f]["top1_std"] or 0 for f in frames]
    top5_mean = [aggregated[f]["top5_mean"] for f in frames]
    top5_std = [aggregated[f]["top5_std"] or 0 for f in frames]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(frames, top1_mean, yerr=top1_std, marker="o", capsize=4, label="Top-1")
    ax.errorbar(frames, top5_mean, yerr=top5_std, marker="s", capsize=4, label="Top-5")
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("WLASL30 sequence length sweep (mean ± std over seeds)")
    ax.set_xticks(frames)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir
    if not sweep_dir.is_dir():
        raise SystemExit(f"Sweep dir not found: {sweep_dir}")
    output_dir = args.output_dir or sweep_dir
    pattern = re.compile(args.run_pattern)

    records = collect_runs(sweep_dir, pattern)
    if not records:
        raise SystemExit("No valid runs found under sweep dir")
    aggregated = aggregate_by_frames(records)

    csv_path = output_dir / "sweep_summary.csv"
    md_path = output_dir / "sweep_summary.md"
    png_path = output_dir / "sequence_length_comparison.png"

    write_csv(records, csv_path)
    write_markdown(records, aggregated, md_path)
    plot_comparison(aggregated, png_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {png_path}")
    print(f"Runs aggregated: {len(records)}  Frame lengths: {sorted(aggregated)}")


if __name__ == "__main__":
    main()
