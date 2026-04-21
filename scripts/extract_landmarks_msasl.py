#!/usr/bin/env python3
"""Extract MediaPipe hand landmarks for an MS-ASL subset.

Usage:
    python scripts/extract_landmarks_msasl.py \
        --subset-metadata 06_datasets/MS-ASL/subsets/msasl50/metadata.json \
        --videos-dir 06_datasets/MS-ASL/videos \
        --output-dir 06_datasets/MS-ASL/subsets/msasl50/landmarks \
        --num-hands 1 \
        --max-workers 4 \
        --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, total=None, desc=None):
        if desc:
            print(f"{desc}: {total if total is not None else ''}")
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from slr.data.landmarks import create_hand_landmarker, extract_landmarks_from_video


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-metadata", type=Path, default=Path("06_datasets/MS-ASL/subsets/msasl50/metadata.json"))
    parser.add_argument("--videos-dir", type=Path, default=Path("06_datasets/MS-ASL/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("06_datasets/MS-ASL/subsets/msasl50/landmarks"))
    parser.add_argument("--model-path", type=Path, default=None,
                        help="MediaPipe hand_landmarker.task path; defaults to known project locations")
    parser.add_argument("--num-hands", type=int, default=2,
                        help="MediaPipe max hands. WLASL extraction used 2 and saved the first detected hand.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--min-detection-rate", type=float, default=0.3)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.3)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.3)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.3)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing subset metadata: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if "samples" not in metadata:
        raise SystemExit(f"Subset metadata does not contain samples: {path}")
    return metadata


def flatten_samples(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in SPLITS:
        for sample in metadata["samples"].get(split, []):
            item = dict(sample)
            item["split"] = split
            samples.append(item)
    return samples


def resolve_output_path(output_dir: Path, sample: dict[str, Any]) -> Path:
    landmarks_path = Path(sample.get("landmarks_path") or sample["path"])
    parts = landmarks_path.parts
    if parts and parts[0] == "landmarks":
        return output_dir.joinpath(*parts[1:])
    return output_dir / landmarks_path


def process_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_path = resolve_output_path(args.output_dir, sample)
    video_path = args.videos_dir / sample["video_path"]

    if args.skip_existing and output_path.exists():
        return {
            **sample,
            "output_path": output_path.as_posix(),
            "success": True,
            "skipped": True,
            "error": None,
            "sampled_frames": None,
            "detected_frames": None,
            "detection_rate": None,
        }

    detector = create_hand_landmarker(
        model_path=args.model_path,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    try:
        landmarks, stats = extract_landmarks_from_video(
            video_path,
            detector=detector,
            max_frames=args.max_frames,
            min_detection_rate=args.min_detection_rate,
        )
    finally:
        if hasattr(detector, "close"):
            detector.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if landmarks is not None:
        np.save(output_path, landmarks)

    stats_dict = asdict(stats)
    return {
        **sample,
        "output_path": output_path.as_posix(),
        "success": stats.success,
        "skipped": False,
        **stats_dict,
    }


def write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "clip_id",
        "class_name",
        "gloss_idx",
        "original_label",
        "video_path",
        "output_path",
        "success",
        "skipped",
        "error",
        "sampled_frames",
        "detected_frames",
        "detection_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(path: Path, rows: list[dict[str, Any]], elapsed_s: float, args: argparse.Namespace) -> None:
    processed_rows = [row for row in rows if not row.get("skipped")]
    successful_rows = [row for row in rows if row.get("success")]
    failed_rows = [row for row in rows if not row.get("success")]
    zero_detection = [
        row for row in processed_rows
        if row.get("detected_frames") == 0 or row.get("detection_rate") == 0
    ]
    measured_rates = [
        float(row["detection_rate"])
        for row in processed_rows
        if row.get("detection_rate") is not None
    ]

    lines = [
        "# MS-ASL Landmark Extraction Report",
        "",
        f"Total samples: {len(rows)}",
        f"Processed samples: {len(processed_rows)}",
        f"Successful samples: {len(successful_rows)}",
        f"Failed samples: {len(failed_rows)}",
        f"Skipped existing samples: {sum(1 for row in rows if row.get('skipped'))}",
        f"Zero-detection samples: {len(zero_detection)}",
        f"Mean detection rate: {mean(measured_rates):.3f}" if measured_rates else "Mean detection rate: n/a",
        f"Elapsed time: {elapsed_s / 3600:.2f} hours",
        "",
        "## MediaPipe Parameters",
        "",
        f"- `num_hands`: {args.num_hands}",
        f"- `max_frames`: {args.max_frames}",
        f"- `min_detection_rate`: {args.min_detection_rate}",
        f"- `min_hand_detection_confidence`: {args.min_hand_detection_confidence}",
        f"- `min_hand_presence_confidence`: {args.min_hand_presence_confidence}",
        f"- `min_tracking_confidence`: {args.min_tracking_confidence}",
        "",
        "## Detection Rate by Split",
        "",
        "| Split | Samples | Mean detection rate | Failed |",
        "|---|---|---|---|",
    ]
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        split_rates = [
            float(row["detection_rate"])
            for row in split_rows
            if row.get("detection_rate") is not None
        ]
        failed = sum(1 for row in split_rows if not row.get("success"))
        rate_text = f"{mean(split_rates):.3f}" if split_rates else "n/a"
        lines.append(f"| {split} | {len(split_rows)} | {rate_text} | {failed} |")

    lines.extend(
        [
            "",
            "## Detection Rate by Class",
            "",
            "| Class | Samples | Mean detection rate | Failed | Flag |",
            "|---|---|---|---|---|",
        ]
    )
    classes = sorted({row["class_name"] for row in rows})
    for class_name in classes:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        class_rates = [
            float(row["detection_rate"])
            for row in class_rows
            if row.get("detection_rate") is not None
        ]
        failed = sum(1 for row in class_rows if not row.get("success"))
        class_mean = mean(class_rates) if class_rates else None
        flag = "low detection" if class_mean is not None and class_mean < 0.5 else ""
        rate_text = f"{class_mean:.3f}" if class_mean is not None else "n/a"
        lines.append(f"| {class_name} | {len(class_rows)} | {rate_text} | {failed} | {flag} |")

    if zero_detection:
        lines.extend(["", "## Zero-Detection Samples", "", "| Split | Class | Clip ID | Error |", "|---|---|---|---|"])
        for row in zero_detection:
            lines.append(f"| {row['split']} | {row['class_name']} | {row['clip_id']} | {row.get('error') or ''} |")

    lines.extend(
        [
            "",
            "## Format Validation",
            "",
            "Compare one generated `.npy` with one WLASL `.npy` after extraction:",
            "",
            "```bash",
            "python - <<'PY'",
            "import numpy as np",
            "from pathlib import Path",
            "msasl = next(Path('06_datasets/MS-ASL/subsets/msasl50/landmarks').glob('*/*/*.npy'))",
            "wlasl = next(Path('06_datasets/wlasl/landmarks/wlasl100_filtered30').glob('*/*/*.npy'))",
            "for name, path in [('MS-ASL', msasl), ('WLASL', wlasl)]:",
            "    arr = np.load(path)",
            "    print(name, path, arr.shape, arr.dtype, arr.min(), arr.max())",
            "PY",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")

    metadata = load_metadata(args.subset_metadata)
    samples = flatten_samples(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_sample, sample, args) for sample in samples]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting landmarks"):
            rows.append(future.result())

    rows.sort(key=lambda row: (row["split"], int(row["gloss_idx"]), row["clip_id"]))
    elapsed_s = time.time() - start
    write_stats_csv(args.output_dir.parent / "extraction_stats.csv", rows)
    write_report(args.output_dir.parent / "extraction_report.md", rows, elapsed_s, args)

    failed = sum(1 for row in rows if not row.get("success"))
    print(f"[done] Processed {len(rows)} samples in {elapsed_s / 3600:.2f} hours")
    if failed:
        print(f"[warn] {failed} samples failed; see {args.output_dir.parent / 'extraction_report.md'}")


if __name__ == "__main__":
    main()
