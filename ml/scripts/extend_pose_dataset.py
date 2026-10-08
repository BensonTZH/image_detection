#!/usr/bin/env python3
"""Extend a validated YOLO pose dataset with another CVAT label export."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from case_classifier_utils import CaseImage, grouped_split, source_id_from_filename

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class NewRecord:
    image: Path
    label: Path
    source_id: str
    label_text: str
    corrections: list[dict[str, float | int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new-labels", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--boundary-tolerance", type=float, default=0.002)
    return parser.parse_args()


def normalized_label(path: Path, tolerance: float) -> tuple[str, list[dict[str, float | int]]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty annotation row")
    values = [float(value) for value in lines[0].split()]
    if len(values) != 17:
        raise ValueError(f"{path}: expected 17 values, found {len(values)}")
    if values[0] != 0:
        raise ValueError(f"{path}: class must be 0")
    for index in (7, 10, 13, 16):
        if values[index] not in (0, 1, 2):
            raise ValueError(f"{path}: invalid visibility at field {index}")

    corrections: list[dict[str, float | int]] = []
    coordinate_indices = (5, 6, 8, 9, 11, 12, 14, 15)
    for index in coordinate_indices:
        original = values[index]
        if 0 <= original <= 1:
            continue
        if original < -tolerance or original > 1 + tolerance:
            raise ValueError(f"{path}: coordinate {index}={original} exceeds boundary tolerance")
        corrected = min(1.0, max(0.0, original))
        values[index] = corrected
        corrections.append({"field": index, "from": original, "to": corrected})

    points = [values[5 + index * 3 : 7 + index * 3] for index in range(4)]
    tl, tr, br, bl = points
    ordered = (
        tl[0] < tr[0]
        and bl[0] < br[0]
        and max(tl[1], tr[1]) < min(bl[1], br[1])
    )
    if not ordered:
        raise ValueError(f"{path}: expected keypoints in TL, TR, BR, BL order")

    if corrections:
        left = min(point[0] for point in points)
        right = max(point[0] for point in points)
        top = min(point[1] for point in points)
        bottom = max(point[1] for point in points)
        values[1:5] = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)

    return (
        " ".join(
            str(int(value)) if index in (0, 7, 10, 13, 16) else f"{value:.6f}"
            for index, value in enumerate(values)
        )
        + "\n",
        corrections,
    )


def find_image(source: Path, stem: str) -> Path:
    matches = sorted(
        path for path in source.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one source image for {stem}, found {len(matches)}")
    return matches[0]


def load_new_records(args: argparse.Namespace) -> list[NewRecord]:
    records: list[NewRecord] = []
    for label in sorted(args.new_labels.glob("*.txt")):
        image = find_image(args.source_images, label.stem)
        label_text, corrections = normalized_label(label, args.boundary_tolerance)
        records.append(
            NewRecord(
                image=image,
                label=label,
                source_id=source_id_from_filename(image),
                label_text=label_text,
                corrections=corrections,
            )
        )
    if not records:
        raise ValueError(f"No label files found under {args.new_labels}")
    return records


def prepare_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    for kind in ("images", "labels"):
        for split in SPLITS:
            (output / kind / split).mkdir(parents=True, exist_ok=False)


def main() -> int:
    args = parse_args()
    args.base = args.base.resolve()
    args.new_labels = args.new_labels.resolve()
    args.source_images = args.source_images.resolve()
    args.output = args.output.resolve()

    base_manifest = json.loads((args.base / "manifest.json").read_text(encoding="utf-8"))
    base_rows = base_manifest.get("images", [])
    new_records = load_new_records(args)

    base_names = {row["filename"] for row in base_rows}
    new_names = {record.image.name for record in new_records}
    if len(new_names) != len(new_records):
        raise ValueError("New export contains duplicate filenames")
    if overlap := base_names & new_names:
        raise ValueError(f"New filenames overlap base dataset: {sorted(overlap)[:5]}")
    base_sources = {row["source_id"] for row in base_rows}
    new_sources = {record.source_id for record in new_records}
    if overlap := base_sources & new_sources:
        raise ValueError(f"New source UUIDs overlap base dataset: {sorted(overlap)[:5]}")

    assignment = grouped_split(
        [CaseImage(record.image, "success", record.source_id) for record in new_records],
        seed=args.seed,
    )
    prepare_output(args.output)

    manifest_rows: list[dict[str, object]] = []
    for row in base_rows:
        split = row["split"]
        image_source = args.base / row["image"]
        label_source = args.base / row["label"]
        image_target = args.output / "images" / split / row["filename"]
        label_target = args.output / "labels" / split / f"{Path(row['filename']).stem}.txt"
        shutil.copy2(image_source, image_target)
        shutil.copy2(label_source, label_target)
        manifest_rows.append(
            {
                "filename": row["filename"],
                "source_id": row["source_id"],
                "split": split,
                "provenance": "slot-v2",
                "image": str(image_target.relative_to(args.output)),
                "label": str(label_target.relative_to(args.output)),
                "corrections": [],
            }
        )

    for record in new_records:
        split = assignment[record.source_id]
        image_target = args.output / "images" / split / record.image.name
        label_target = args.output / "labels" / split / f"{record.image.stem}.txt"
        shutil.copy2(record.image, image_target)
        label_target.write_text(record.label_text, encoding="utf-8")
        manifest_rows.append(
            {
                "filename": record.image.name,
                "source_id": record.source_id,
                "split": split,
                "provenance": "new-export",
                "source_image": str(record.image),
                "source_label": str(record.label),
                "image": str(image_target.relative_to(args.output)),
                "label": str(label_target.relative_to(args.output)),
                "corrections": record.corrections,
            }
        )

    manifest_rows.sort(key=lambda row: str(row["filename"]))
    split_counts = Counter(str(row["split"]) for row in manifest_rows)
    source_counts = {
        split: len({str(row["source_id"]) for row in manifest_rows if row["split"] == split})
        for split in SPLITS
    }
    manifest = {
        "base_dataset": str(args.base),
        "new_labels": str(args.new_labels),
        "source_images": str(args.source_images),
        "seed": args.seed,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "preserved_base_splits": True,
        "split_summary": {
            split: {"images": split_counts[split], "sources": source_counts[split]}
            for split in SPLITS
        },
        "correction_count": sum(len(row["corrections"]) for row in manifest_rows),
        "images": manifest_rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for split in SPLITS:
        rows = [row for row in manifest_rows if row["split"] == split]
        (args.output / f"{split}.txt").write_text(
            "".join(f"images/{split}/{row['filename']}\n" for row in rows),
            encoding="utf-8",
        )
    (args.output / "data.yaml").write_text(
        "\n".join(
            (
                "path: .",
                "train: train.txt",
                "val: val.txt",
                "test: test.txt",
                "kpt_shape: [4, 3]",
                "flip_idx: [1, 0, 3, 2]",
                "names:",
                "  0: slot",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps({"images": len(manifest_rows), "splits": dict(split_counts), "corrections": manifest["correction_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
