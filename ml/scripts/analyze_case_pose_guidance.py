#!/usr/bin/env python3
"""Extract pose guidance metrics from the original success/error photographs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .case_classifier_utils import CLASSES, DEFAULT_RATIOS, SPLITS, grouped_split, scan_case_dataset, split_summary
    from .pose_guidance_metrics import (
        KEYPOINT_NAMES,
        alignment_metrics,
        default_guidance,
        range_overlap,
        success_reference,
        summarize_rows,
    )
    from .training_utils import choose_device, write_json
except ImportError:  # Direct execution: python ml/scripts/analyze_case_pose_guidance.py
    from case_classifier_utils import CLASSES, DEFAULT_RATIOS, SPLITS, grouped_split, scan_case_dataset, split_summary
    from pose_guidance_metrics import (
        KEYPOINT_NAMES,
        alignment_metrics,
        default_guidance,
        range_overlap,
        success_reference,
        summarize_rows,
    )
    from training_utils import choose_device, write_json

CSV_FIELDS = (
    "source_path", "filename", "source_id", "label", "split", "status",
    "image_width", "image_height", "detection_confidence",
    "box_x1", "box_y1", "box_x2", "box_y2",
    "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "center_x", "center_y", "frame_area", "roll_degrees",
    "side_perspective", "vertical_perspective", "perspective_difference",
    "top_width", "bottom_width", "left_height", "right_height",
    "min_corner_confidence", "corners_reliable",
    "default_guidance",
    *(
        f"{name}_{field}"
        for name in KEYPOINT_NAMES
        for field in ("x", "y", "x_norm", "y_norm", "confidence")
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/case"))
    parser.add_argument("--weights", type=Path, default=Path("runs/phase-2/baseline/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/pose-guidance"))
    parser.add_argument("--ratios", type=float, nargs=3, default=DEFAULT_RATIOS, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    return parser.parse_args()


def _tensor_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _empty_row(record: Any, split: str, width: int, height: int, status: str) -> dict[str, Any]:
    return {
        "source_path": str(record.path),
        "filename": record.path.name,
        "source_id": record.source_id,
        "label": record.label,
        "split": split,
        "status": status,
        "image_width": width,
        "image_height": height,
    }


def row_from_result(
    record: Any,
    split: str,
    result: Any,
    keypoint_confidence: float,
) -> dict[str, Any]:
    height, width = (int(value) for value in result.orig_shape)
    if result.boxes is None or len(result.boxes) == 0:
        return _empty_row(record, split, width, height, "no_detection")

    confidences = _tensor_list(result.boxes.conf)
    best_index = max(range(len(confidences)), key=lambda index: confidences[index])
    box = [float(value) for value in _tensor_list(result.boxes.xyxy)[best_index]]
    if result.keypoints is None or len(result.keypoints) <= best_index:
        return _empty_row(record, split, width, height, "missing_keypoints")
    coordinates = _tensor_list(result.keypoints.xy)[best_index]
    confidence_values = _tensor_list(getattr(result.keypoints, "conf", None))
    point_confidences = confidence_values[best_index] if confidence_values else [1.0] * 4
    if len(coordinates) != 4 or len(point_confidences) != 4:
        return _empty_row(record, split, width, height, "invalid_keypoints")

    keypoints = [
        {"x": float(point[0]), "y": float(point[1]), "confidence": float(confidence)}
        for point, confidence in zip(coordinates, point_confidences)
    ]
    row = _empty_row(record, split, width, height, "detected")
    row.update({
        "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
        "box_x1_norm": box[0] / width, "box_y1_norm": box[1] / height,
        "box_x2_norm": box[2] / width, "box_y2_norm": box[3] / height,
    })
    row.update(alignment_metrics(
        box,
        keypoints,
        width,
        height,
        float(confidences[best_index]),
        keypoint_confidence,
    ))
    for name, point in zip(KEYPOINT_NAMES, keypoints):
        row.update({
            f"{name}_x": point["x"],
            f"{name}_y": point["y"],
            f"{name}_x_norm": point["x"] / width,
            f"{name}_y_norm": point["y"] / height,
            f"{name}_confidence": point["confidence"],
        })
    row["default_guidance"] = default_guidance(row)
    return row


def grouped_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": summarize_rows(rows), "by_split_and_label": {}}
    for split in SPLITS:
        output["by_split_and_label"][split] = {}
        for label in CLASSES:
            selected = [row for row in rows if row["split"] == split and row["label"] == label]
            output["by_split_and_label"][split][label] = summarize_rows(selected)
    return output


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Case pose-guidance analysis",
        "",
        "This report uses the existing slot pose detector on the original full photographs. "
        "It does not train or integrate the success/error classifier.",
        "",
        "## Coverage",
        "",
        "| Split | Label | Images | Detected | Reliable corners |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        for label in CLASSES:
            group = report["summary"]["by_split_and_label"][split][label]
            lines.append(
                f"| {split} | {label} | {group['images']} | "
                f"{group['detected']} ({_percentage(group['detection_rate'])}) | "
                f"{group['reliable_corners']} ({_percentage(group['reliable_corner_rate_all_images'])}) |"
            )

    lines.extend([
        "",
        "## Train-success reference ranges",
        "",
        "These are observed p05-p95 ranges, not thresholds applied to CupDetector.",
        "",
        "| Metric | Median | p05 | p95 |",
        "| --- | ---: | ---: | ---: |",
    ])
    features = report["train_success_reference"]["features"]
    for name, values in features.items():
        if values:
            lines.append(f"| {name} | {values['median']:.4f} | {values['p05']:.4f} | {values['p95']:.4f} |")

    lines.extend([
        "",
        "## Validation overlap with train-success ranges",
        "",
        "A useful guidance metric should usually retain many success images while excluding "
        "some error images. High overlap for both labels means that metric alone cannot tell "
        "good and bad poses apart.",
        "",
        "| Metric | Success inside | Error inside |",
        "| --- | ---: | ---: |",
    ])
    overlap = report["validation_range_overlap"]
    for name in features:
        success = overlap["success"].get(name)
        error = overlap["error"].get(name)
        if success and error:
            lines.append(f"| {name} | {_percentage(success['rate'])} | {_percentage(error['rate'])} |")

    lines.extend([
        "",
        "## Current default guidance on validation images",
        "",
        "This is a single-frame replay of the current default rules. Live smoothing and "
        "instruction hold timing are intentionally not simulated.",
        "",
        "| Label | Aligned | Most common instruction |",
        "| --- | ---: | --- |",
    ])
    for label in CLASSES:
        group = report["summary"]["by_split_and_label"]["val"][label]
        guidance = group["default_guidance"]
        aligned = guidance.get("aligned", 0)
        most_common = max(guidance.items(), key=lambda item: item[1])
        lines.append(
            f"| {label} | {aligned}/{group['images']} ({_percentage(aligned / max(group['images'], 1))}) | "
            f"{most_common[0]} ({most_common[1]}) |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `no_detection` means the slot detector itself needs more representative detection/pose labels.",
        "- `detected` with unreliable corners means the four-corner prediction needs improvement before angle guidance can be trusted.",
        "- Reliable detections whose metrics overlap heavily across success and error show the limit of geometry-only guidance.",
        "- Only train-success images establish the ranges above. Validation measures them; test remains available for the final chosen rules.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        print(f"Pose checkpoint not found: {args.weights}", file=sys.stderr)
        return 2
    if not 0 <= args.confidence <= 1 or not 0 <= args.keypoint_confidence <= 1:
        print("Confidence thresholds must be between zero and one", file=sys.stderr)
        return 2
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    records = scan_case_dataset(args.dataset)
    assignment = grouped_split(records, args.ratios, args.seed)
    ordered = sorted(records, key=lambda item: (assignment[item.source_id], item.label, item.path.name))
    device = choose_device(args.device)
    model = YOLO(str(args.weights.resolve()))
    rows: list[dict[str, Any]] = []
    # Passing thousands of source paths to one Ultralytics loader can cause a
    # large memory spike even with stream=True. Small explicit batches keep the
    # run stable while producing the same per-image predictions.
    for start in range(0, len(ordered), args.batch):
        batch_records = ordered[start:start + args.batch]
        predictions = model.predict(
            source=[str(record.path) for record in batch_records],
            imgsz=args.imgsz,
            batch=len(batch_records),
            device=device,
            conf=args.confidence,
            stream=False,
            verbose=False,
        )
        if len(predictions) != len(batch_records):
            print(
                f"Expected {len(batch_records)} predictions for batch at {start}, "
                f"received {len(predictions)}",
                file=sys.stderr,
            )
            return 2
        for record, result in zip(batch_records, predictions):
            rows.append(row_from_result(record, assignment[record.source_id], result, args.keypoint_confidence))
        if len(rows) % 100 < args.batch or len(rows) == len(ordered):
            print(f"Analyzed {len(rows)}/{len(ordered)} images", flush=True)
    if len(rows) != len(ordered):
        print(f"Expected {len(ordered)} predictions but received {len(rows)}", file=sys.stderr)
        return 2

    reference = success_reference(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ultralytics": ultralytics.__version__,
        "dataset": str(args.dataset.resolve()),
        "weights": str(args.weights.resolve()),
        "device": device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "detection_confidence_threshold": args.confidence,
        "keypoint_confidence_threshold": args.keypoint_confidence,
        "seed": args.seed,
        "split": split_summary(records, assignment),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "summary": grouped_summaries(rows),
        "train_success_reference": reference,
        "validation_range_overlap": range_overlap(rows, reference),
    }
    output = args.output.resolve()
    write_csv(output / "case-pose-metrics.csv", rows)
    write_json(output / "case-pose-summary.json", report)
    write_markdown(output / "case-pose-summary.md", report)
    print(json.dumps({
        "images": len(rows),
        "status_counts": report["status_counts"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
