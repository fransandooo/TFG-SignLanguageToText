#!/usr/bin/env python3
"""Build a contiguous-label MS-ASL subset from the video audit artifacts.

Usage:
    python scripts/build_msasl_subset.py \
        --audit-csv 06_datasets/MS-ASL/audit/audit_per_class.csv \
        --msasl-classes 06_datasets/MS-ASL/MSASL_classes.json \
        --videos-dir 06_datasets/MS-ASL/videos \
        --output-dir 06_datasets/MS-ASL/subsets/msasl50 \
        --n-classes 50 \
        --min-train 20 \
        --min-val 3 \
        --min-test 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=Path("06_datasets/MS-ASL/audit/audit_per_class.csv"))
    parser.add_argument("--audit-videos-csv", type=Path, default=None,
                        help="Per-video audit CSV (defaults to audit_videos.csv next to --audit-csv)")
    parser.add_argument("--msasl-classes", type=Path, default=Path("06_datasets/MS-ASL/MSASL_classes.json"))
    parser.add_argument("--videos-dir", type=Path, default=Path("06_datasets/MS-ASL/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("06_datasets/MS-ASL/subsets/msasl50"))
    parser.add_argument("--n-classes", type=int, default=50)
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-val", type=int, default=3)
    parser.add_argument("--min-test", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required CSV file: {path}") from exc


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "class"


def select_classes(
    class_rows: list[dict[str, str]],
    n_classes: int,
    min_train: int,
    min_val: int,
    min_test: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    strict = [
        row for row in class_rows
        if as_int(row, "n_train_valid") >= min_train
        and as_int(row, "n_val_valid") >= min_val
        and as_int(row, "n_test_valid") >= min_test
    ]
    criterion = {
        "min_train": min_train,
        "min_val": min_val,
        "min_test": min_test,
        "relaxed": False,
        "reason": None,
    }

    candidates = strict
    if len(candidates) < n_classes:
        relaxed_min_train = min(min_train, 15)
        candidates = [
            row for row in class_rows
            if as_int(row, "n_train_valid") >= relaxed_min_train
            and as_int(row, "n_val_valid") >= min_val
            and as_int(row, "n_test_valid") >= min_test
        ]
        criterion.update(
            {
                "min_train": relaxed_min_train,
                "relaxed": True,
                "reason": f"Strict threshold produced {len(strict)} classes; train minimum relaxed to {relaxed_min_train}.",
            }
        )

    candidates = sorted(
        candidates,
        key=lambda row: (
            as_int(row, "n_train_valid"),
            as_int(row, "n_val_valid"),
            as_int(row, "n_test_valid"),
            -as_int(row, "label"),
        ),
        reverse=True,
    )
    return candidates[:n_classes], criterion


def build_samples(
    selected: list[dict[str, str]],
    video_rows: list[dict[str, str]],
    videos_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    selected_labels = {as_int(row, "label") for row in selected}
    new_by_original = {as_int(row, "label"): new_label for new_label, row in enumerate(selected)}
    class_info = {
        as_int(row, "label"): {
            "new_label": new_by_original[as_int(row, "label")],
            "class_name": row["class_name"],
            "folder_name": safe_name(row["class_name"]),
            "counts": {
                "train": as_int(row, "n_train_valid"),
                "val": as_int(row, "n_val_valid"),
                "test": as_int(row, "n_test_valid"),
            },
            "duration_s": {
                "train": float(row["duration_train_s"]),
                "val": float(row["duration_val_s"]),
                "test": float(row["duration_test_s"]),
            },
        }
        for row in selected
    }

    samples: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in sorted(video_rows, key=lambda r: (r["split"], as_int(r, "label"), r["clip_id"])):
        original_label = as_int(row, "label")
        if original_label not in selected_labels or row["status"] != "valid":
            continue
        split = row["split"]
        if split not in samples:
            continue
        info = class_info[original_label]
        relative_video_path = Path(row["filepath"])
        try:
            relative_video_path = relative_video_path.relative_to(videos_dir)
        except ValueError:
            relative_video_path = Path(split) / f"{original_label:04d}" / f"{row['clip_id']}.mp4"

        landmark_rel = Path(split) / info["folder_name"] / f"{row['clip_id']}.npy"
        samples[split].append(
            {
                "clip_id": row["clip_id"],
                "path": landmark_rel.as_posix(),
                "gloss": info["class_name"],
                "gloss_idx": info["new_label"],
                "label": info["new_label"],
                "class_name": info["class_name"],
                "original_label": original_label,
                "video_path": relative_video_path.as_posix(),
                "landmarks_path": (Path("landmarks") / landmark_rel).as_posix(),
                "duration_s": float(row["duration_s"]) if row.get("duration_s") else None,
            }
        )
    return samples, class_info


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_subset_report(
    path: Path,
    selected: list[dict[str, str]],
    class_info: dict[int, dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
    criterion: dict[str, Any],
) -> None:
    total_duration_s = sum(
        sample["duration_s"] or 0.0
        for split_samples in samples.values()
        for sample in split_samples
    )
    lines = [
        "# MS-ASL50 Subset Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Selection Criterion",
        "",
        f"- Requested classes: {len(selected)}",
        f"- Minimum train samples: {criterion['min_train']}",
        f"- Minimum validation samples: {criterion['min_val']}",
        f"- Minimum test samples: {criterion['min_test']}",
        f"- Relaxed threshold: {'yes' if criterion['relaxed'] else 'no'}",
    ]
    if criterion["reason"]:
        lines.append(f"- Relaxation reason: {criterion['reason']}")
    lines.extend(
        [
            "",
            "## Sample Counts",
            "",
            "| Split | Samples |",
            "|---|---|",
        ]
    )
    for split in SPLITS:
        lines.append(f"| {split} | {len(samples[split])} |")
    lines.extend(
        [
            f"| total | {sum(len(samples[split]) for split in SPLITS)} |",
            "",
            f"Total video duration to process: {total_duration_s / 3600:.2f} hours.",
            "",
            "## Selected Classes",
            "",
            "| New label | Original label | Class | Train | Val | Test |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in selected:
        original_label = as_int(row, "label")
        info = class_info[original_label]
        counts = info["counts"]
        lines.append(
            f"| {info['new_label']} | {original_label} | {info['class_name']} | "
            f"{counts['train']} | {counts['val']} | {counts['test']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_classes <= 0:
        raise SystemExit("--n-classes must be positive")

    classes = load_json(args.msasl_classes)
    if not isinstance(classes, list):
        raise SystemExit(f"Expected class list in {args.msasl_classes}")

    audit_videos_csv = args.audit_videos_csv or args.audit_csv.with_name("audit_videos.csv")
    class_rows = read_csv(args.audit_csv)
    video_rows = read_csv(audit_videos_csv)

    selected, criterion = select_classes(
        class_rows,
        args.n_classes,
        args.min_train,
        args.min_val,
        args.min_test,
    )
    if len(selected) < args.n_classes:
        print(f"[warn] Only {len(selected)} classes satisfy the relaxed threshold.")

    samples, class_info = build_samples(selected, video_rows, args.videos_dir)
    class_names = [row["class_name"] for row in selected]
    gloss_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_gloss = {str(idx): name for idx, name in enumerate(class_names)}
    original_to_new = {str(as_int(row, "label")): idx for idx, row in enumerate(selected)}
    new_to_original = {str(idx): as_int(row, "label") for idx, row in enumerate(selected)}

    metadata = {
        "name": f"msasl{len(selected)}",
        "dataset_name": f"msasl{len(selected)}",
        "description": "MS-ASL subset built from disk-valid audited clips.",
        "source": "MS-ASL",
        "source_dataset": "MS-ASL",
        "num_classes": len(selected),
        "n_classes": len(selected),
        "glosses": class_names,
        "class_names": class_names,
        "gloss_to_idx": gloss_to_idx,
        "label_mapping": gloss_to_idx,
        "idx_to_gloss": idx_to_gloss,
        "original_label_mapping": original_to_new,
        "new_to_original_label": new_to_original,
        "samples": samples,
        "splits": samples,
        "statistics": {
            **{split: len(samples[split]) for split in SPLITS},
            "total": sum(len(samples[split]) for split in SPLITS),
        },
        "n_samples": {split: len(samples[split]) for split in SPLITS},
        "selection_criterion": criterion,
        "build_date": datetime.now().isoformat(timespec="seconds"),
    }

    remapping = {
        "dataset_name": metadata["dataset_name"],
        "original_to_new": original_to_new,
        "new_to_original": new_to_original,
        "classes": [
            {
                "new_label": idx,
                "original_label": as_int(row, "label"),
                "class_name": row["class_name"],
                "folder_name": class_info[as_int(row, "label")]["folder_name"],
            }
            for idx, row in enumerate(selected)
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "label_remapping.json", remapping)
    write_subset_report(args.output_dir / "subset_report.md", selected, class_info, samples, criterion)
    print(f"[done] Wrote subset metadata to {args.output_dir}")
    print(
        "[info] Samples: "
        + ", ".join(f"{split}={len(samples[split])}" for split in SPLITS)
        + f", total={sum(len(samples[split]) for split in SPLITS)}"
    )


if __name__ == "__main__":
    main()
