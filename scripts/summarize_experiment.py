#!/usr/bin/env python3
"""Aggregate repeated experiment runs across seed directories.

Usage:
    python scripts/summarize_experiment.py \
        --experiment-dir 05_experiments/msasl50_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_KEYS = ("top1", "top5", "macro_f1", "balanced_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="Directory containing one subdirectory per run")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write summary artifacts (defaults to --experiment-dir)")
    parser.add_argument("--run-pattern", default=r"^seed(?P<seed>\d+)$",
                        help="Regex for run directories; must expose a 'seed' group")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        print(f"[skip] {path.parent.name}: experiment_manifest.json missing")
    except json.JSONDecodeError as exc:
        print(f"[skip] {path.parent.name}: invalid JSON ({exc})")
    return None


def collect_runs(experiment_dir: Path, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in sorted(experiment_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if not match:
            print(f"[skip] {entry.name}: does not match run pattern")
            continue
        manifest = load_manifest(entry / "experiment_manifest.json")
        if manifest is None:
            continue

        records.append(
            {
                "run_id": entry.name,
                "seed": int(match.group("seed")),
                "experiment_id": manifest.get("experiment_id"),
                "top1": manifest.get("top1"),
                "top5": manifest.get("top5"),
                "macro_f1": manifest.get("macro_f1"),
                "balanced_accuracy": manifest.get("balanced_accuracy"),
                "val_top1": manifest.get("metrics", {}).get("val_top1"),
                "best_epoch": manifest.get("runtime", {}).get("best_epoch"),
                "total_epochs": manifest.get("runtime", {}).get("total_epochs"),
                "training_time_s": manifest.get("runtime", {}).get("training_time_s"),
            }
        )
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for metric in METRIC_KEYS:
        values = [record[metric] for record in records if record.get(metric) is not None]
        summary[metric] = {
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else (0.0 if values else None),
            "n": len(values),
        }
    return summary


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "run_id", "seed", "experiment_id",
        "top1", "top5", "macro_f1", "balanced_accuracy",
        "val_top1", "best_epoch", "total_epochs", "training_time_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["seed"]):
            writer.writerow({key: record.get(key) for key in fieldnames})


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(records: list[dict[str, Any]], summary: dict[str, dict[str, Any]], path: Path) -> None:
    lines: list[str] = [
        "# Experiment Summary",
        "",
        f"Runs aggregated: {len(records)}",
        "",
        "## Individual Runs",
        "",
        "| Run | Seed | Top-1 | Top-5 | Macro-F1 | Balanced Accuracy | Val Top-1 | Best epoch | Total epochs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda item: item["seed"]):
        lines.append(
            f"| {record['run_id']} | {record['seed']} | "
            f"{fmt(record['top1'])} | {fmt(record['top5'])} | "
            f"{fmt(record['macro_f1'])} | {fmt(record['balanced_accuracy'])} | "
            f"{fmt(record['val_top1'])} | {fmt(record['best_epoch'])} | {fmt(record['total_epochs'])} |"
        )

    lines.extend(
        [
            "",
            "## Aggregated Metrics",
            "",
            "| Metric | Mean | Std | N |",
            "|---|---|---|---|",
        ]
    )
    for metric in METRIC_KEYS:
        item = summary[metric]
        lines.append(f"| {metric} | {fmt(item['mean'])} | {fmt(item['std'])} | {item['n']} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metrics(summary: dict[str, dict[str, Any]], path: Path) -> None:
    metrics = [metric for metric in METRIC_KEYS if summary[metric]["mean"] is not None]
    if not metrics:
        print("[warn] no metrics to plot")
        return

    labels = ["Top-1", "Top-5", "Macro-F1", "Balanced Accuracy"]
    label_by_metric = dict(zip(METRIC_KEYS, labels))
    means = [summary[metric]["mean"] for metric in metrics]
    stds = [summary[metric]["std"] or 0.0 for metric in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_positions = range(len(metrics))
    ax.bar(x_positions, means, yerr=stds, capsize=5, color="#4C78A8")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([label_by_metric[metric] for metric in metrics])
    ax.set_ylabel("Score (%)")
    ax.set_title("MS-ASL50 baseline metrics (mean ± std over seeds)")
    ax.set_ylim(0, max(100, max(means) + max(stds or [0]) + 5))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.experiment_dir.is_dir():
        raise SystemExit(f"Experiment dir not found: {args.experiment_dir}")

    pattern = re.compile(args.run_pattern)
    records = collect_runs(args.experiment_dir, pattern)
    if not records:
        raise SystemExit("No valid runs found")

    output_dir = args.output_dir or args.experiment_dir
    summary = aggregate(records)
    write_csv(records, output_dir / "summary.csv")
    write_markdown(records, summary, output_dir / "summary.md")
    plot_metrics(summary, output_dir / "metrics_comparison.png")

    print(f"Wrote: {output_dir / 'summary.csv'}")
    print(f"Wrote: {output_dir / 'summary.md'}")
    print(f"Wrote: {output_dir / 'metrics_comparison.png'}")
    print(f"Runs aggregated: {len(records)}")


if __name__ == "__main__":
    main()
