#!/usr/bin/env python3
"""Extract MediaPipe hand landmarks for an MS-ASL subset.

Usage:
    python scripts/extract_landmarks_msasl.py \
        --subset-metadata 06_datasets/MS-ASL/subsets/msasl50/metadata.json \
        --videos-dir 06_datasets/MS-ASL/videos \
        --output-dir 06_datasets/MS-ASL/subsets/msasl50/landmarks \
        --num-hands 1 \
        --max-workers 4 \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles
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
    parser.add_argument("--force-max-frames", action="store_true",
                        help="Legacy mode: uniformly sample videos to --max-frames before saving. Default saves full-length sequences and leaves padding/truncation to the Dataset.")
    parser.add_argument("--min-detection-rate", type=float, default=0.3)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.3)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.3)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.3)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .npy files. Takes precedence over --skip-existing.")
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

    if args.overwrite and output_path.exists():
        output_path.unlink()

    if args.skip_existing and not args.overwrite and output_path.exists():
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
            max_frames=args.max_frames if args.force_max_frames else None,
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


def compute_length_stats(output_dir: Path) -> dict[str, Any]:
    paths = sorted(output_dir.glob("*/*/*.npy"))
    lengths = [int(np.load(path, mmap_mode="r").shape[0]) for path in paths]
    if not lengths:
        return {
            "n_files": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "exact_64": 0,
        }

    if len(lengths) >= 4:
        quartile_values = quantiles(lengths, n=4)
        p25 = quartile_values[0]
        p75 = quartile_values[2]
    else:
        p25 = min(lengths)
        p75 = max(lengths)

    return {
        "n_files": len(paths),
        "min": min(lengths),
        "p25": p25,
        "median": median(lengths),
        "p75": p75,
        "max": max(lengths),
        "exact_64": sum(1 for length in lengths if length == 64),
    }


def sample_has_landmarks(sample: dict[str, Any], output_dir: Path) -> bool:
    return resolve_output_path(output_dir, sample).exists()


def filter_metadata_by_outputs(metadata_path: Path, metadata: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    backup_path = metadata_path.with_suffix(metadata_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(metadata_path, backup_path)

    original_counts = {
        split: len(metadata.get("samples", {}).get(split, []))
        for split in SPLITS
    }
    filtered_samples: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        filtered_samples[split] = [
            sample for sample in metadata.get("samples", {}).get(split, [])
            if sample_has_landmarks(sample, output_dir)
        ]

    filtered = dict(metadata)
    filtered["samples"] = filtered_samples
    if isinstance(metadata.get("splits"), dict):
        filtered["splits"] = {
            split: [
                sample for sample in metadata.get("splits", {}).get(split, [])
                if sample_has_landmarks(sample, output_dir)
            ]
            for split in SPLITS
        }

    n_samples = {split: len(filtered_samples[split]) for split in SPLITS}
    filtered["n_samples"] = n_samples
    if isinstance(filtered.get("statistics"), dict):
        statistics = dict(filtered["statistics"])
        for split in SPLITS:
            statistics[split] = n_samples[split]
        statistics["total"] = sum(n_samples.values())
        filtered["statistics"] = statistics

    filtered["extraction_filter_applied"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metadata_backup": backup_path.name,
        "samples_before": original_counts,
        "samples_after": n_samples,
        "samples_filtered": sum(original_counts.values()) - sum(n_samples.values()),
    }
    metadata_path.write_text(json.dumps(filtered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return filtered


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    elapsed_s: float,
    args: argparse.Namespace,
    length_stats: dict[str, Any],
    metadata_filter: dict[str, Any],
    command: str,
) -> None:
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
        f"- `force_max_frames`: {args.force_max_frames}",
        f"- `max_frames`: {args.max_frames if args.force_max_frames else 'not applied at extraction time'}",
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
            "## Format Update",
            "",
            "Saved `.npy` files are no longer forced to `max_frames` at extraction time. Sequences keep the decoded video length after per-frame detection, interpolation, edge-fill at missing-detection boundaries, and normalization. Padding or truncation is handled later by the Dataset.",
            "",
            f"- Regeneration command: `{command}`",
            f"- Metadata backup: `{args.subset_metadata.with_suffix(args.subset_metadata.suffix + '.bak').name}`",
            f"- Samples filtered from metadata: {metadata_filter.get('samples_filtered', 0)}",
            f"- Landmark files: {length_stats['n_files']}",
            f"- Length min: {length_stats['min']}",
            f"- Length p25: {length_stats['p25']}",
            f"- Length median: {length_stats['median']}",
            f"- Length p75: {length_stats['p75']}",
            f"- Length max: {length_stats['max']}",
            f"- Files with exactly 64 frames: {length_stats['exact_64']}",
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
    if args.overwrite and args.skip_existing:
        print("[warn] --overwrite and --skip-existing were both set; --overwrite takes precedence.")

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
    filtered_metadata = filter_metadata_by_outputs(args.subset_metadata, metadata, args.output_dir)
    length_stats = compute_length_stats(args.output_dir)
    metadata_filter = filtered_metadata.get("extraction_filter_applied", {})
    command = " ".join(sys.argv)
    write_report(
        args.output_dir.parent / "extraction_report.md",
        rows,
        elapsed_s,
        args,
        length_stats,
        metadata_filter,
        command,
    )

    failed = sum(1 for row in rows if not row.get("success"))
    print(f"[done] Processed {len(rows)} samples in {elapsed_s / 3600:.2f} hours")
    print(
        "[info] Lengths: "
        f"min={length_stats['min']} p25={length_stats['p25']} median={length_stats['median']} "
        f"p75={length_stats['p75']} max={length_stats['max']} exact64={length_stats['exact_64']}"
    )
    print(
        "[info] Metadata samples: "
        + ", ".join(f"{split}={filtered_metadata['n_samples'][split]}" for split in SPLITS)
        + f", total={sum(filtered_metadata['n_samples'].values())}"
    )
    if length_stats["n_files"] != sum(filtered_metadata["n_samples"].values()):
        print("[warn] Number of .npy files does not match filtered metadata sample count.")
    if failed:
        print(f"[warn] {failed} samples failed; see {args.output_dir.parent / 'extraction_report.md'}")


if __name__ == "__main__":
    main()
