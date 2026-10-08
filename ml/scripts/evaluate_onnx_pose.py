#!/usr/bin/env python3
"""Evaluate the fixed-batch browser ONNX model one test image at a time."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from PIL import Image

from training_utils import write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def ground_truth(label: Path, width: int, height: int) -> tuple[list[float], list[list[float]]]:
    rows = [row for row in label.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"{label}: expected one annotation row")
    values = [float(value) for value in rows[0].split()]
    if len(values) != 17:
        raise ValueError(f"{label}: expected 17 values")
    cx, cy, box_width, box_height = values[1:5]
    box = [
        (cx - box_width / 2) * width,
        (cy - box_height / 2) * height,
        (cx + box_width / 2) * width,
        (cy + box_height / 2) * height,
    ]
    keypoints = [
        [values[5 + offset * 3] * width, values[6 + offset * 3] * height]
        for offset in range(4)
    ]
    return box, keypoints


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    images = sorted(
        path
        for path in (args.dataset / "images" / args.split).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    model = YOLO(str(args.model), task="pose")
    cases: list[dict[str, object]] = []
    confidences: list[float] = []
    ious: list[float] = []
    corner_errors: list[float] = []

    for image_path in images:
        label_path = args.dataset / "labels" / args.split / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size
        expected_box, expected_keypoints = ground_truth(label_path, width, height)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.7,
            max_det=1,
            device="cpu",
            verbose=False,
        )[0]
        case: dict[str, object] = {"image": str(image_path), "detected": False}
        if result.boxes is not None and len(result.boxes):
            predicted_box = result.boxes.xyxy[0].cpu().tolist()
            confidence = float(result.boxes.conf[0].cpu())
            predicted_keypoints = (
                [] if result.keypoints is None else result.keypoints.xy[0].cpu().tolist()
            )
            iou = box_iou(expected_box, predicted_box)
            errors = [
                math.dist(expected, predicted)
                for expected, predicted in zip(expected_keypoints, predicted_keypoints)
            ]
            case.update(
                {
                    "detected": True,
                    "confidence": confidence,
                    "box_iou": iou,
                    "corner_errors_pixels": errors,
                }
            )
            confidences.append(confidence)
            ious.append(iou)
            corner_errors.extend(errors)
        cases.append(case)

    detected = len(confidences)
    summary = {
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "confidence_threshold": args.confidence,
        "images": len(images),
        "detected": detected,
        "detection_rate": detected / len(images) if images else 0.0,
        "confidence_median": statistics.median(confidences) if confidences else None,
        "confidence_p05": percentile(confidences, 0.05),
        "box_iou_median": statistics.median(ious) if ious else None,
        "box_iou_p05": percentile(ious, 0.05),
        "corner_error_pixels_median": statistics.median(corner_errors) if corner_errors else None,
        "corner_error_pixels_p95": percentile(corner_errors, 0.95),
        "cases": cases,
    }
    write_json(args.output, summary)
    print(
        {
            key: value
            for key, value in summary.items()
            if key not in {"model", "dataset", "cases"}
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
