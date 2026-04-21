#!/usr/bin/env python3
"""Audit downloaded MS-ASL video clips and summarize training viability.

The audit inspects the videos that are physically present on disk. It does not
download clips, modify videos, or extract landmarks.

Usage:
    python scripts/audit_msasl.py \
        --msasl-dir 06_datasets/MS-ASL \
        --videos-dir 06_datasets/MS-ASL/videos \
        --output-dir 06_datasets/MS-ASL/audit \
        --wlasl-metadata 06_datasets/wlasl/landmarks/wlasl100_filtered30/metadata.json \
        --max-workers 6 \
        --top-k 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLITS = ("train", "val", "test")
MIN_DURATION_S = 0.5
MIN_DIMENSION_PX = 64
MIN_SIZE_BYTES = 10 * 1024
FFPROBE_TIMEOUT_S = 15
EXTRACTION_FPS_ASSUMPTION = 30.0
EXTRACTION_SECONDS_PER_FRAME = 0.04

VIDEO_FIELDS = [
    "split",
    "label",
    "class_name",
    "clip_id",
    "filepath",
    "size_bytes",
    "duration_s",
    "num_frames",
    "width",
    "height",
    "fps",
    "codec",
    "status",
]

CLASS_FIELDS = [
    "label",
    "class_name",
    "n_train_valid",
    "n_val_valid",
    "n_test_valid",
    "duration_train_s",
    "duration_val_s",
    "duration_test_s",
    "pct_corrupt",
    "viable_for_msasl30",
    "viable_for_msasl50",
    "viable_for_msasl100",
]


@dataclass(frozen=True)
class VideoTask:
    split: str
    label: int
    class_name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--msasl-dir",
        type=Path,
        default=Path("06_datasets/MS-ASL"),
        help="MS-ASL root directory containing MSASL_classes.json",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("06_datasets/MS-ASL/videos"),
        help="Directory with videos organized as split/label/clip_id.mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("06_datasets/MS-ASL/audit"),
        help="Directory where audit artifacts will be written",
    )
    parser.add_argument(
        "--wlasl-metadata",
        type=Path,
        default=Path("06_datasets/wlasl/landmarks/wlasl100_filtered30/metadata.json"),
        help="WLASL metadata JSON used to cross-reference glosses",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel ffprobe workers",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Only audit labels from 0 to top-k - 1",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def load_msasl_classes(msasl_dir: Path) -> list[str]:
    classes_path = msasl_dir / "MSASL_classes.json"
    classes = load_json(classes_path)
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise SystemExit(f"Expected a list of class names in {classes_path}")
    return classes


def load_wlasl_glosses(metadata_path: Path) -> set[str]:
    metadata = load_json(metadata_path)
    glosses: set[str] = set()

    if isinstance(metadata, dict):
        raw_glosses = metadata.get("glosses")
        if isinstance(raw_glosses, list):
            glosses.update(str(item) for item in raw_glosses)

        gloss_to_idx = metadata.get("gloss_to_idx")
        if isinstance(gloss_to_idx, dict):
            glosses.update(str(key) for key in gloss_to_idx.keys())

        idx_to_gloss = metadata.get("idx_to_gloss")
        if isinstance(idx_to_gloss, dict):
            glosses.update(str(value) for value in idx_to_gloss.values())

    if not glosses:
        raise SystemExit(f"No WLASL glosses found in {metadata_path}")

    return glosses


def normalize_gloss(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_rate(value: Any) -> float | None:
    if value in (None, "", "0/0"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            numerator_f = float(numerator)
            denominator_f = float(denominator)
        except ValueError:
            return None
        if denominator_f == 0:
            return None
        return numerator_f / denominator_f
    try:
        return float(text)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffprobe_timeout"}
    except FileNotFoundError:
        raise SystemExit("ffprobe was not found. Install ffmpeg before running this audit.")

    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip() or "ffprobe_failed"}

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "ffprobe_invalid_json"}

    streams = payload.get("streams") or []
    if not streams:
        return {"ok": False, "error": "no_video_stream"}

    stream = streams[0]
    fmt = payload.get("format") or {}

    duration_s = parse_float(stream.get("duration"))
    if duration_s is None:
        duration_s = parse_float(fmt.get("duration"))

    fps = parse_rate(stream.get("avg_frame_rate"))
    if fps is None:
        fps = parse_rate(stream.get("r_frame_rate"))

    num_frames = parse_int(stream.get("nb_frames"))
    if num_frames is None and duration_s is not None and fps is not None:
        num_frames = int(round(duration_s * fps))

    return {
        "ok": True,
        "duration_s": duration_s,
        "num_frames": num_frames,
        "width": parse_int(stream.get("width")),
        "height": parse_int(stream.get("height")),
        "fps": fps,
        "codec": stream.get("codec_name"),
    }


def classify_video(size_bytes: int, probe: dict[str, Any]) -> str:
    if size_bytes < MIN_SIZE_BYTES:
        return "empty"
    if not probe.get("ok"):
        return "corrupt"

    duration_s = probe.get("duration_s")
    width = probe.get("width")
    height = probe.get("height")

    if duration_s is None or duration_s < MIN_DURATION_S:
        return "too_short"
    if width is None or height is None or width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        return "too_small"
    return "valid"


def audit_one_video(task: VideoTask) -> dict[str, Any]:
    try:
        size_bytes = task.path.stat().st_size
    except FileNotFoundError:
        size_bytes = 0
        probe = {"ok": False, "error": "missing_file"}
    else:
        probe = probe_video(task.path)

    status = classify_video(size_bytes, probe)
    return {
        "split": task.split,
        "label": task.label,
        "class_name": task.class_name,
        "clip_id": task.path.stem,
        "filepath": task.path.as_posix(),
        "size_bytes": size_bytes,
        "duration_s": probe.get("duration_s"),
        "num_frames": probe.get("num_frames"),
        "width": probe.get("width"),
        "height": probe.get("height"),
        "fps": probe.get("fps"),
        "codec": probe.get("codec"),
        "status": status,
    }


def collect_tasks(videos_dir: Path, classes: list[str], top_k: int) -> list[VideoTask]:
    if top_k <= 0:
        raise SystemExit("--top-k must be positive")

    max_label = min(top_k, len(classes))
    tasks: list[VideoTask] = []
    for split in SPLITS:
        for label in range(max_label):
            label_dir = videos_dir / split / f"{label:04d}"
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.glob("*.mp4")):
                tasks.append(VideoTask(split, label, classes[label], path))
    return tasks


def audit_videos(tasks: list[VideoTask], max_workers: int) -> list[dict[str, Any]]:
    if max_workers <= 0:
        raise SystemExit("--max-workers must be positive")

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(audit_one_video, task): task for task in tasks}
        completed = 0
        total = len(future_to_task)
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                records.append(future.result())
            except Exception as exc:
                records.append(
                    {
                        "split": task.split,
                        "label": task.label,
                        "class_name": task.class_name,
                        "clip_id": task.path.stem,
                        "filepath": task.path.as_posix(),
                        "size_bytes": None,
                        "duration_s": None,
                        "num_frames": None,
                        "width": None,
                        "height": None,
                        "fps": None,
                        "codec": None,
                        "status": "corrupt",
                    }
                )
                print(f"[warn] Failed to audit {task.path}: {exc}")
            if completed % 500 == 0 or completed == total:
                print(f"[progress] Probed {completed}/{total} videos")

    return sorted(records, key=lambda row: (row["label"], row["split"], row["clip_id"]))


def aggregate_per_class(
    video_records: list[dict[str, Any]],
    classes: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    max_label = min(top_k, len(classes))
    by_label: dict[int, list[dict[str, Any]]] = {label: [] for label in range(max_label)}
    for record in video_records:
        by_label.setdefault(int(record["label"]), []).append(record)

    class_records: list[dict[str, Any]] = []
    for label in range(max_label):
        rows = by_label.get(label, [])
        valid_by_split = {
            split: [row for row in rows if row["split"] == split and row["status"] == "valid"]
            for split in SPLITS
        }
        durations = {
            split: sum(float(row.get("duration_s") or 0.0) for row in split_rows)
            for split, split_rows in valid_by_split.items()
        }
        corrupt_count = sum(1 for row in rows if row["status"] == "corrupt")
        pct_corrupt = (100.0 * corrupt_count / len(rows)) if rows else 0.0

        n_train = len(valid_by_split["train"])
        n_val = len(valid_by_split["val"])
        n_test = len(valid_by_split["test"])

        class_records.append(
            {
                "label": label,
                "class_name": classes[label],
                "n_train_valid": n_train,
                "n_val_valid": n_val,
                "n_test_valid": n_test,
                "duration_train_s": durations["train"],
                "duration_val_s": durations["val"],
                "duration_test_s": durations["test"],
                "pct_corrupt": pct_corrupt,
                "viable_for_msasl30": label < 30 and n_train >= 30 and n_val >= 5 and n_test >= 5,
                "viable_for_msasl50": label < 50 and n_train >= 20 and n_val >= 5 and n_test >= 5,
                "viable_for_msasl100": label < 100 and n_train >= 15 and n_val >= 3 and n_test >= 3,
            }
        )

    return class_records


def viable_count(class_records: list[dict[str, Any]], top_n: int, train: int, val: int, test: int) -> int:
    return sum(
        1
        for row in class_records
        if int(row["label"]) < top_n
        and int(row["n_train_valid"]) >= train
        and int(row["n_val_valid"]) >= val
        and int(row["n_test_valid"]) >= test
    )


def build_threshold_summary(class_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = [
        ("MS-ASL100 strict", 100, 20, 3, 3),
        ("MS-ASL100 baseline", 100, 15, 3, 3),
        ("MS-ASL50 strict", 50, 20, 5, 5),
        ("MS-ASL30 strict", 30, 30, 5, 5),
    ]
    return [
        {
            "subset": name,
            "top_n": top_n,
            "min_train": train,
            "min_val": val,
            "min_test": test,
            "viable_classes": viable_count(class_records, top_n, train, val, test),
        }
        for name, top_n, train, val, test in definitions
    ]


def build_shared_glosses(
    classes: list[str],
    wlasl_glosses: set[str],
    class_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wlasl_by_normalized = {normalize_gloss(gloss): gloss for gloss in sorted(wlasl_glosses)}
    class_by_label = {int(row["label"]): row for row in class_records}
    shared: list[dict[str, Any]] = []

    for label, class_name in enumerate(classes):
        normalized = normalize_gloss(class_name)
        if normalized not in wlasl_by_normalized:
            continue
        row = class_by_label.get(label)
        if row is None:
            msasl_valid_total = None
            in_msasl100_viable = False
            n_train_valid = None
            n_val_valid = None
            n_test_valid = None
        else:
            msasl_valid_total = (
                int(row["n_train_valid"]) + int(row["n_val_valid"]) + int(row["n_test_valid"])
            )
            in_msasl100_viable = bool(row["viable_for_msasl100"])
            n_train_valid = int(row["n_train_valid"])
            n_val_valid = int(row["n_val_valid"])
            n_test_valid = int(row["n_test_valid"])
        shared.append(
            {
                "gloss": class_name,
                "normalized_gloss": normalized,
                "wlasl_gloss": wlasl_by_normalized[normalized],
                "msasl_label": label,
                "audited": row is not None,
                "in_wlasl100": True,
                "in_msasl_top100": label < 100,
                "in_msasl100_viable": in_msasl100_viable,
                "n_viable_msasl": 1 if in_msasl100_viable else 0,
                "n_train_valid": n_train_valid,
                "n_val_valid": n_val_valid,
                "n_test_valid": n_test_valid,
                "n_valid_msasl": msasl_valid_total,
            }
        )

    return sorted(shared, key=lambda item: (not item["in_msasl_top100"], item["msasl_label"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key)) for key in fieldnames})


def format_csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return value


def fmt_number(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def total_valid_duration(class_records: list[dict[str, Any]], top_n: int) -> float:
    total = 0.0
    for row in class_records:
        if int(row["label"]) >= top_n:
            continue
        total += float(row["duration_train_s"])
        total += float(row["duration_val_s"])
        total += float(row["duration_test_s"])
    return total


def estimate_extraction_hours(duration_s: float) -> float:
    return duration_s * EXTRACTION_FPS_ASSUMPTION * EXTRACTION_SECONDS_PER_FRAME / 3600.0


def split_summary(video_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_rows = [row for row in video_records if row["split"] == split]
        valid_rows = [row for row in split_rows if row["status"] == "valid"]
        rows.append(
            {
                "split": split,
                "total": len(split_rows),
                "valid": len(valid_rows),
                "corrupt": sum(1 for row in split_rows if row["status"] == "corrupt"),
                "too_short": sum(1 for row in split_rows if row["status"] == "too_short"),
                "too_small": sum(1 for row in split_rows if row["status"] == "too_small"),
                "empty": sum(1 for row in split_rows if row["status"] == "empty"),
                "duration_valid_s": sum(float(row.get("duration_s") or 0.0) for row in valid_rows),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_report(
    path: Path,
    video_records: list[dict[str, Any]],
    class_records: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    shared_glosses: list[dict[str, Any]],
    top_k: int,
) -> None:
    total_clips = len(video_records)
    valid_clips = sum(1 for row in video_records if row["status"] == "valid")
    corrupt_clips = sum(1 for row in video_records if row["status"] == "corrupt")
    empty_clips = sum(1 for row in video_records if row["status"] == "empty")
    too_short_clips = sum(1 for row in video_records if row["status"] == "too_short")
    too_small_clips = sum(1 for row in video_records if row["status"] == "too_small")

    top_classes = sorted(
        class_records,
        key=lambda row: (
            int(row["n_train_valid"]) + int(row["n_val_valid"]) + int(row["n_test_valid"]),
            -int(row["label"]),
        ),
        reverse=True,
    )[:30]
    worst_classes = sorted(
        class_records,
        key=lambda row: (
            int(row["n_train_valid"]) + int(row["n_val_valid"]) + int(row["n_test_valid"]),
            -float(row["pct_corrupt"]),
            int(row["label"]),
        ),
    )[:30]

    top100_duration_s = total_valid_duration(class_records, min(100, top_k))
    top30_duration_s = total_valid_duration(class_records, min(30, top_k))
    top100_hours = estimate_extraction_hours(top100_duration_s)
    top30_hours = estimate_extraction_hours(top30_duration_s)

    viable_100 = viable_count(class_records, 100, 15, 3, 3)
    viable_50 = viable_count(class_records, 50, 20, 5, 5)
    viable_30 = viable_count(class_records, 30, 30, 5, 5)

    if viable_100 >= 100:
        recommendation = "Use MS-ASL100 with the baseline 15/3/3 split threshold."
    elif viable_50 >= 30:
        recommendation = "Use a reduced MS-ASL50-style subset, then select the strongest 30 classes for controlled experiments."
    elif viable_30 >= 20:
        recommendation = "Use an MS-ASL30-style subset and keep the class list fixed before landmark extraction."
    else:
        recommendation = "Use the strongest available classes only; the current top-k set does not fully satisfy the planned thresholds."

    lines: list[str] = []
    lines.append("# MS-ASL Video Audit")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Audited labels: 0-{top_k - 1}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"The audit found {valid_clips:,} valid clips out of {total_clips:,} clips on disk "
        f"({(100.0 * valid_clips / total_clips) if total_clips else 0.0:.1f}%). "
        f"Invalid clips include {corrupt_clips:,} corrupt, {empty_clips:,} empty, "
        f"{too_short_clips:,} too short, and {too_small_clips:,} too small clips."
    )
    lines.append("")
    lines.append(f"Recommendation: {recommendation}")
    lines.append("")
    lines.append("## Valid Clips by Split")
    lines.append("")
    lines.extend(
        markdown_table(
            ["Split", "Total", "Valid", "Corrupt", "Empty", "Too short", "Too small", "Valid duration (s)"],
            [
                [
                    row["split"],
                    f"{row['total']:,}",
                    f"{row['valid']:,}",
                    f"{row['corrupt']:,}",
                    f"{row['empty']:,}",
                    f"{row['too_short']:,}",
                    f"{row['too_small']:,}",
                    fmt_number(row["duration_valid_s"], 1),
                ]
                for row in split_summary(video_records)
            ],
        )
    )
    lines.append("")
    lines.append("## Viability Thresholds")
    lines.append("")
    lines.extend(
        markdown_table(
            ["Subset", "Top N", "Min train", "Min val", "Min test", "Viable classes"],
            [
                [
                    row["subset"],
                    row["top_n"],
                    row["min_train"],
                    row["min_val"],
                    row["min_test"],
                    row["viable_classes"],
                ]
                for row in threshold_rows
            ],
        )
    )
    lines.append("")
    lines.append("## Top 30 Best Classes")
    lines.append("")
    lines.extend(
        markdown_table(
            ["Label", "Class", "Train", "Val", "Test", "Total valid", "Corrupt %"],
            [
                [
                    row["label"],
                    row["class_name"],
                    row["n_train_valid"],
                    row["n_val_valid"],
                    row["n_test_valid"],
                    int(row["n_train_valid"]) + int(row["n_val_valid"]) + int(row["n_test_valid"]),
                    f"{float(row['pct_corrupt']):.1f}",
                ]
                for row in top_classes
            ],
        )
    )
    lines.append("")
    lines.append("## Top 30 Weakest Classes")
    lines.append("")
    lines.extend(
        markdown_table(
            ["Label", "Class", "Train", "Val", "Test", "Total valid", "Corrupt %"],
            [
                [
                    row["label"],
                    row["class_name"],
                    row["n_train_valid"],
                    row["n_val_valid"],
                    row["n_test_valid"],
                    int(row["n_train_valid"]) + int(row["n_val_valid"]) + int(row["n_test_valid"]),
                    f"{float(row['pct_corrupt']):.1f}",
                ]
                for row in worst_classes
            ],
        )
    )
    lines.append("")
    lines.append("## Shared Glosses with WLASL100")
    lines.append("")
    shared_top100 = [item for item in shared_glosses if item["in_msasl_top100"]]
    viable_shared_top100 = [item for item in shared_top100 if item["in_msasl100_viable"]]
    lines.append(
        f"Shared glosses in MS-ASL top-100: {len(shared_top100):,}. "
        f"Shared glosses viable under the MS-ASL100 15/3/3 threshold: {len(viable_shared_top100):,}."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            ["Gloss", "In WLASL100", "In MS-ASL100", "MS-ASL viable count", "Valid clips"],
            [
                [
                    item["gloss"],
                    "yes" if item["in_wlasl100"] else "no",
                    "yes" if item["in_msasl_top100"] else "no",
                    item["n_viable_msasl"],
                    item["n_valid_msasl"],
                ]
                for item in shared_top100[:40]
            ],
        )
    )
    lines.append("")
    lines.append("## Landmark Extraction Time Estimate")
    lines.append("")
    lines.append(
        "Estimate formula: valid video duration * 30 frames/s * 0.04 s/frame / 3600."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            ["Scope", "Valid duration (s)", "Estimated hours"],
            [
                ["Top-100 audited labels", fmt_number(top100_duration_s, 1), fmt_number(top100_hours, 2)],
                ["Top-30 audited labels", fmt_number(top30_duration_s, 1), fmt_number(top30_hours, 2)],
            ],
        )
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `audit_videos.csv`: one row per audited video.")
    lines.append("- `audit_per_class.csv`: one row per class with split counts and viability flags.")
    lines.append("- `shared_glosses_wlasl_msasl.json`: structured cross-dataset gloss overlap.")
    lines.append("- `audit_plot_class_viability.png`: valid clips by split for audited classes.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_shared_glosses(path: Path, shared_glosses: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "normalization": "lowercase, trim whitespace, replace hyphen/underscore with spaces, collapse whitespace",
        "shared_glosses": shared_glosses,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_viability_plot(path: Path, class_records: list[dict[str, Any]], top_k: int) -> None:
    rows = sorted(
        [row for row in class_records if int(row["label"]) < top_k],
        key=lambda row: int(row["n_train_valid"]) + int(row["n_val_valid"]) + int(row["n_test_valid"]),
        reverse=True,
    )
    if not rows:
        print("[warn] No class records available for plot")
        return

    labels = [f"{int(row['label']):04d} {row['class_name']}" for row in rows]
    train = [int(row["n_train_valid"]) for row in rows]
    val = [int(row["n_val_valid"]) for row in rows]
    test = [int(row["n_test_valid"]) for row in rows]

    fig_width = max(14.0, min(36.0, len(rows) * 0.28))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    x_positions = list(range(len(rows)))
    ax.bar(x_positions, train, label="train", color="#4C78A8")
    ax.bar(x_positions, val, bottom=train, label="val", color="#F58518")
    bottoms = [train_value + val_value for train_value, val_value in zip(train, val)]
    ax.bar(x_positions, test, bottom=bottoms, label="test", color="#54A24B")

    ax.set_title(f"MS-ASL Top-{top_k} Valid Clips by Class")
    ax.set_xlabel("Class, ordered by total valid clips")
    ax.set_ylabel("Valid video clips")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_msasl_classes(args.msasl_dir)
    wlasl_glosses = load_wlasl_glosses(args.wlasl_metadata)
    tasks = collect_tasks(args.videos_dir, classes, args.top_k)
    print(f"[info] Found {len(tasks):,} video files for labels 0-{min(args.top_k, len(classes)) - 1}")

    video_records = audit_videos(tasks, args.max_workers)
    class_records = aggregate_per_class(video_records, classes, args.top_k)
    threshold_rows = build_threshold_summary(class_records)
    shared_glosses = build_shared_glosses(classes, wlasl_glosses, class_records)

    write_csv(output_dir / "audit_videos.csv", video_records, VIDEO_FIELDS)
    write_csv(output_dir / "audit_per_class.csv", class_records, CLASS_FIELDS)
    write_report(
        output_dir / "audit_report.md",
        video_records,
        class_records,
        threshold_rows,
        shared_glosses,
        args.top_k,
    )
    write_shared_glosses(output_dir / "shared_glosses_wlasl_msasl.json", shared_glosses)
    write_viability_plot(output_dir / "audit_plot_class_viability.png", class_records, min(args.top_k, len(classes)))

    print(f"[done] Wrote audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()
