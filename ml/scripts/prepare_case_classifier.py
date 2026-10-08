#!/usr/bin/env python3
"""Create grouped 70/15/15 slot crops for the success/error classifier."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

try:
    from .case_classifier_utils import (
        CLASSES,
        DEFAULT_RATIOS,
        SPLITS,
        expanded_crop_box,
        grouped_split,
        letterbox_square,
        scan_case_dataset,
        split_summary,
        validate_ratios,
        write_json,
    )
    from .training_utils import choose_device
except ImportError:  # Direct execution: python ml/scripts/prepare_case_classifier.py
    from case_classifier_utils import (
        CLASSES,
        DEFAULT_RATIOS,
        SPLITS,
        expanded_crop_box,
        grouped_split,
        letterbox_square,
        scan_case_dataset,
        split_summary,
        validate_ratios,
        write_json,
    )
    from training_utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset/case"))
    parser.add_argument("--output", type=Path, default=Path("dataset/case-classifier"))
    parser.add_argument("--weights", type=Path, default=Path("runs/phase-2/baseline/weights/best.pt"))
    parser.add_argument("--ratios", nargs=3, type=float, default=DEFAULT_RATIOS, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--classifier-size",
        type=int,
        default=320,
        help="Square output size; the complete crop is resized and padded, never center-cropped",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ratios = validate_ratios(args.ratios)
    if not args.weights.is_file():
        print(f"Pose checkpoint not found: {args.weights}", file=sys.stderr)
        return 2
    if not 0 <= args.confidence <= 1:
        print("--confidence must be between 0 and 1", file=sys.stderr)
        return 2
    if args.classifier_size <= 0:
        print("--classifier-size must be greater than zero", file=sys.stderr)
        return 2

    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            print(f"Output already exists: {output}", file=sys.stderr)
            print("Choose a new --output or explicitly pass --overwrite.", file=sys.stderr)
            return 2
        shutil.rmtree(output)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    records = scan_case_dataset(args.source)
    assignment = grouped_split(records, ratios=ratios, seed=args.seed)
    for split in SPLITS:
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = YOLO(str(args.weights.resolve()))
    manifest_images: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()

    print(f"Preparing {len(records)} images on {device} using grouped 70/15/15 assignment")
    for start in range(0, len(records), args.batch):
        batch_records = records[start : start + args.batch]
        results = model.predict(
            source=[str(record.path) for record in batch_records],
            imgsz=args.imgsz,
            conf=args.confidence,
            max_det=1,
            device=device,
            verbose=False,
        )
        for record, result in zip(batch_records, results):
            split = assignment[record.source_id]
            entry: dict[str, object] = {
                "source_path": str(record.path),
                "filename": record.path.name,
                "source_id": record.source_id,
                "label": record.label,
                "split": split,
            }
            if result.boxes is None or len(result.boxes) == 0:
                entry["status"] = "no_detection"
                entry["reason"] = "The pose detector did not find a slot at the configured confidence."
                manifest_images.append(entry)
                status_counts["no_detection"] += 1
                continue

            confidences = result.boxes.conf.detach().cpu().tolist()
            best_index = max(range(len(confidences)), key=confidences.__getitem__)
            xyxy = result.boxes.xyxy[best_index].detach().cpu().tolist()
            with Image.open(record.path) as source_image:
                source_image = ImageOps.exif_transpose(source_image).convert("RGB")
                crop_box = expanded_crop_box(
                    xyxy,
                    image_width=source_image.width,
                    image_height=source_image.height,
                    margin=args.margin,
                )
                relative_path = Path(split) / record.label / record.path.name
                destination = output / relative_path
                crop = source_image.crop(crop_box)
                prepared = letterbox_square(crop, args.classifier_size)
                prepared.save(destination, format="JPEG", quality=92, optimize=True)

            entry.update({
                "status": "prepared",
                "prepared_path": relative_path.as_posix(),
                "detector_confidence": round(float(confidences[best_index]), 6),
                "crop_xyxy": list(crop_box),
                "source_crop_size": [crop.width, crop.height],
                "prepared_size": [args.classifier_size, args.classifier_size],
            })
            manifest_images.append(entry)
            status_counts["prepared"] += 1

        completed = min(start + len(batch_records), len(records))
        if completed % 160 == 0 or completed == len(records):
            print(f"Prepared {completed}/{len(records)} source images")

    prepared_by_split = {
        split: {
            label: sum(
                item["status"] == "prepared" and item["split"] == split and item["label"] == label
                for item in manifest_images
            )
            for label in CLASSES
        }
        for split in SPLITS
    }
    missed_by_class = {
        label: sum(item["status"] == "no_detection" and item["label"] == label for item in manifest_images)
        for label in CLASSES
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source.resolve()),
        "output_root": str(output),
        "pose_weights": str(args.weights.resolve()),
        "ratios": dict(zip(SPLITS, ratios)),
        "seed": args.seed,
        "crop_margin": args.margin,
        "detector_confidence": args.confidence,
        "detector_imgsz": args.imgsz,
        "classifier_input_size": args.classifier_size,
        "classifier_preprocessing": "letterbox_full_crop",
        "letterbox_fill_rgb": [114, 114, 114],
        "assignment_summary": split_summary(records, assignment),
        "prepared_by_split": prepared_by_split,
        "status_counts": dict(status_counts),
        "missed_by_class": missed_by_class,
        "images": manifest_images,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Manifest: {output / 'manifest.json'}")
    print(f"Prepared crops: {status_counts['prepared']}")
    print(f"No pose detection: {status_counts['no_detection']} ({missed_by_class})")
    if any(prepared_by_split[split][label] == 0 for split in SPLITS for label in CLASSES):
        print("At least one split/class is empty; classifier training cannot proceed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
