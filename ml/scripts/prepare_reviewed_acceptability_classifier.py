#!/usr/bin/env python3
"""Prepare the manually reviewed centered images for acceptability training."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

try:
    from .case_classifier_utils import (
        CLASSES,
        IMAGE_SUFFIXES,
        SPLITS,
        CaseImage,
        grouped_split,
        letterbox_square,
        source_id_from_filename,
        split_summary,
        validate_ratios,
        write_json,
    )
except ImportError:
    from case_classifier_utils import (
        CLASSES,
        IMAGE_SUFFIXES,
        SPLITS,
        CaseImage,
        grouped_split,
        letterbox_square,
        source_id_from_filename,
        split_summary,
        validate_ratios,
        write_json,
    )


SOURCE_FOLDERS = {"success": "acceptable", "error": "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/Categorize/slot-angle-pose-v3-full-984/images/centered"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/reviewed-acceptability-classifier"),
    )
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--classifier-size", type=int, default=320)
    parser.add_argument("--no-balance-train", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scan_source(root: Path) -> list[CaseImage]:
    records: list[CaseImage] = []
    for label, folder in SOURCE_FOLDERS.items():
        class_dir = root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing reviewed class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(CaseImage(path.resolve(), label, source_id_from_filename(path)))
    if not records:
        raise ValueError(f"No reviewed images found under {root}")
    return records


def prepared_name(path: Path) -> str:
    return f"{path.stem}.jpg"


def main() -> int:
    args = parse_args()
    ratios = validate_ratios(args.ratios)
    if args.classifier_size <= 0:
        raise ValueError("--classifier-size must be greater than zero")

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)

    records = scan_source(source)
    assignment = grouped_split(records, ratios=ratios, seed=args.seed)
    for split in SPLITS:
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)

    manifest_images: list[dict[str, object]] = []
    prepared_by_split: dict[str, dict[str, list[Path]]] = {
        split: {label: [] for label in CLASSES} for split in SPLITS
    }
    for index, record in enumerate(records, start=1):
        split = assignment[record.source_id]
        relative = Path(split) / record.label / prepared_name(record.path)
        destination = output / relative
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            prepared = letterbox_square(image, args.classifier_size)
            prepared.save(destination, format="JPEG", quality=92, optimize=True)
        prepared_by_split[split][record.label].append(destination)
        manifest_images.append({
            "source_path": str(record.path),
            "filename": record.path.name,
            "source_id": record.source_id,
            "label": record.label,
            "split": split,
            "status": "prepared",
            "prepared_path": relative.as_posix(),
            "prepared_size": [args.classifier_size, args.classifier_size],
            "replica": False,
        })
        if index % 200 == 0 or index == len(records):
            print(f"Prepared {index}/{len(records)} reviewed images")

    replicas = 0
    if not args.no_balance_train:
        train_counts = {label: len(prepared_by_split["train"][label]) for label in CLASSES}
        minority = min(CLASSES, key=train_counts.__getitem__)
        majority = max(CLASSES, key=train_counts.__getitem__)
        minority_paths = prepared_by_split["train"][minority]
        for replica_index in range(train_counts[majority] - train_counts[minority]):
            original = minority_paths[replica_index % len(minority_paths)]
            destination = original.with_name(
                f"{original.stem}__replica_{replica_index + 1:04d}{original.suffix}"
            )
            shutil.copyfile(original, destination)
            source_item = next(
                item for item in manifest_images if output / str(item["prepared_path"]) == original
            )
            relative = destination.relative_to(output)
            manifest_images.append({
                **source_item,
                "source_path": f"{source_item['source_path']}#replica={replica_index + 1}",
                "filename": destination.name,
                "prepared_path": relative.as_posix(),
                "replica": True,
                "replica_of": original.relative_to(output).as_posix(),
            })
            replicas += 1

    disk_counts = {
        split: {
            label: len(list((output / split / label).iterdir()))
            for label in CLASSES
        }
        for split in SPLITS
    }
    original_counts = Counter(record.label for record in records)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "output_root": str(output),
        "ratios": dict(zip(SPLITS, ratios)),
        "seed": args.seed,
        "classifier_input_size": args.classifier_size,
        "classifier_preprocessing": "letterbox_full_frame",
        "letterbox_fill_rgb": [114, 114, 114],
        "original_counts": dict(original_counts),
        "assignment_summary": split_summary(records, assignment),
        "balanced_training_replicas": replicas,
        "prepared_by_split": disk_counts,
        "status_counts": {"prepared": len(manifest_images)},
        "images": manifest_images,
    }
    write_json(output / "manifest.json", manifest)
    print(f"Manifest: {output / 'manifest.json'}")
    print(f"Original images: {len(records)} ({dict(original_counts)})")
    print(f"Training replicas: {replicas}")
    print(f"Prepared counts: {disk_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
