#!/usr/bin/env python3
"""Evaluate the case classifier with safety-focused test-set metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

try:
    from .case_classifier_utils import CLASSES, IMAGE_SUFFIXES
    from .training_utils import choose_device, write_json
    from .validate_case_classifier import validate
except ImportError:  # Direct execution: python ml/scripts/evaluate_case_classifier.py
    from case_classifier_utils import CLASSES, IMAGE_SUFFIXES
    from training_utils import choose_device, write_json
    from validate_case_classifier import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/case-classifier"))
    parser.add_argument("--weights", type=Path, default=Path("runs/case-classifier/baseline/weights/best.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--success-threshold", type=float, default=0.80)
    parser.add_argument("--name", default="baseline-evaluation")
    parser.add_argument("--examples", type=int, default=24)
    return parser.parse_args()


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    true_success = sum(row["actual"] == "success" for row in rows)
    true_error = len(rows) - true_success
    accepted_success = sum(
        row["actual"] == "success" and row["success_probability"] >= threshold
        for row in rows
    )
    false_accepts = sum(
        row["actual"] == "error" and row["success_probability"] >= threshold
        for row in rows
    )
    rejected_success = true_success - accepted_success
    rejected_error = true_error - false_accepts
    return {
        "threshold": threshold,
        "images": len(rows),
        "confusion": {
            "success_accepted": accepted_success,
            "success_rejected": rejected_success,
            "error_rejected": rejected_error,
            "error_false_accepted": false_accepts,
        },
        "success_acceptance_rate": accepted_success / max(true_success, 1),
        "success_rejection_rate": rejected_success / max(true_success, 1),
        "error_rejection_rate": rejected_error / max(true_error, 1),
        "false_accept_rate": false_accepts / max(true_error, 1),
    }


def render_review_sheet(rows: list[dict[str, Any]], output: Path, limit: int) -> None:
    review = sorted(
        rows,
        key=lambda row: (
            row["predicted"] == row["actual"],
            -abs(row["success_probability"] - 0.5),
        ),
    )[:limit]
    if not review:
        return
    tile_width, tile_height, caption_height = 240, 320, 48
    columns = 4
    rows_count = (len(review) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows_count * (tile_height + caption_height)), "#101512")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(review):
        with Image.open(item["path"]) as source:
            image = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (tile_width, tile_height))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + caption_height)
        sheet.paste(image, (x + (tile_width - image.width) // 2, y + (tile_height - image.height) // 2))
        colour = "#b7ff14" if item["predicted"] == item["actual"] else "#ff6b6b"
        draw.text((x + 6, y + tile_height + 5), f"actual: {item['actual']}", fill=colour)
        draw.text((x + 6, y + tile_height + 23), f"success: {item['success_probability']:.3f}", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> int:
    args = parse_args()
    if not 0 <= args.success_threshold <= 1:
        print("--success-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"Classifier checkpoint not found: {args.weights}", file=sys.stderr)
        return 2
    report = validate(args.dataset)
    if report["summary"]["status"] != "PASS":
        print("Prepared dataset validation failed; run validate_case_classifier.py", file=sys.stderr)
        return 2
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    project = Path("runs/case-classifier").resolve()
    output = project / args.name
    device = choose_device(args.device)
    model = YOLO(
        str(args.weights.resolve()),
        task="classify" if args.weights.suffix.lower() == ".onnx" else None,
    )
    validation = model.val(
        data=str(args.dataset.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=True,
    )
    success_indices = [index for index, name in model.names.items() if name == "success"]
    if len(success_indices) != 1:
        print(f"Expected exactly one 'success' class, model names are {model.names}", file=sys.stderr)
        return 2
    success_index = success_indices[0]
    test_paths = sorted(
        path
        for label in CLASSES
        for path in (args.dataset / "test" / label).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    # Static ONNX exports in this project have a fixed batch dimension of 1.
    # Ultralytics' validation path handles that contract automatically, while
    # predict() otherwise attempts to assemble the requested batch itself.
    if args.weights.suffix.lower() == ".onnx":
        # ClassificationPredictor currently combines a list source into one
        # tensor even when batch=1, which violates a static ONNX batch shape.
        predictions = (
            model.predict(
                source=str(path),
                imgsz=args.imgsz,
                batch=1,
                device=device,
                verbose=False,
            )[0]
            for path in test_paths
        )
    else:
        predictions = model.predict(
            source=[str(path) for path in test_paths],
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            stream=True,
            verbose=False,
        )
    rows: list[dict[str, Any]] = []
    for path, result in zip(test_paths, predictions):
        probability = float(result.probs.data[success_index].detach().cpu().item())
        rows.append({
            "path": str(path.resolve()),
            "actual": path.parent.name,
            "success_probability": probability,
            "predicted": "success" if probability >= args.success_threshold else "error",
        })

    metrics = threshold_metrics(rows, args.success_threshold)
    write_json(output / "test-metrics.json", {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "ultralytics": ultralytics.__version__,
        "weights": str(args.weights.resolve()),
        "device": device,
        "imgsz": args.imgsz,
        "classifier_top1": getattr(validation, "top1", None),
        "classifier_top5": getattr(validation, "top5", None),
        **metrics,
    })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "test-predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "actual", "success_probability", "predicted"))
        writer.writeheader()
        writer.writerows(rows)
    render_review_sheet(rows, output / "review-sheet.jpg", args.examples)
    print(json.dumps(metrics, indent=2))
    print(f"Evaluation saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
