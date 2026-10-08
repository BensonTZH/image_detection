#!/usr/bin/env python3
"""Build a three-class YOLO pose dataset from reviewed angle folders."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_FOLDERS = {
    "acceptable": (0, "centered"),
    "left_angle": (1, "camera_left"),
    "right_angle": (2, "camera_right"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify-root", type=Path, required=True)
    parser.add_argument("--pose-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def rewrite_class(label: Path, class_id: int) -> str:
    rows = [row for row in label.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"{label}: expected one annotation row, found {len(rows)}")
    fields = rows[0].split()
    if len(fields) != 17:
        raise ValueError(f"{label}: expected 17 pose values, found {len(fields)}")
    if fields[0] != "0":
        raise ValueError(f"{label}: expected original slot class 0, found {fields[0]}")
    fields[0] = str(class_id)
    return " ".join(fields) + "\n"


def main() -> int:
    args = parse_args()
    classify_root = args.classify_root.resolve()
    pose_dataset = args.pose_dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    label_index: dict[str, tuple[Path, str]] = {}
    for split in ("train", "val"):
        label_dir = pose_dataset / "labels" / split
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing pose labels: {label_dir}")
        for label in label_dir.glob("*.txt"):
            if label.stem in label_index:
                raise ValueError(f"Duplicate pose label for {label.stem}")
            label_index[label.stem] = (label, split)

    records: list[tuple[Path, Path, str, int, str]] = []
    seen: set[str] = set()
    for folder, (class_id, class_name) in CLASS_FOLDERS.items():
        source = classify_root / folder
        if not source.is_dir():
            raise FileNotFoundError(f"Missing classification folder: {source}")
        for image in sorted(source.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image.name in seen:
                raise ValueError(f"Duplicate classified image: {image.name}")
            seen.add(image.name)
            match = label_index.get(image.stem)
            if match is None:
                raise FileNotFoundError(f"No pose label found for {image.name}")
            label, split = match
            records.append((image, label, split, class_id, class_name))

    if not records:
        raise ValueError("No classified images were found")
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (output / kind / split).mkdir(parents=True, exist_ok=False)

    counts: Counter[tuple[str, str]] = Counter()
    for image, label, split, class_id, class_name in records:
        image_target = output / "images" / split / image.name
        if args.copy_images:
            image_target.write_bytes(image.read_bytes())
        else:
            os.link(image, image_target)
        (output / "labels" / split / f"{image.stem}.txt").write_text(
            rewrite_class(label, class_id), encoding="utf-8"
        )
        counts[(split, class_name)] += 1

    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {json.dumps(str(output))}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [4, 3]",
                "flip_idx: [1, 0, 3, 2]",
                "names:",
                "  0: centered",
                "  1: camera_left",
                "  2: camera_right",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "source_classify_root": str(classify_root),
        "source_pose_dataset": str(pose_dataset),
        "excluded_folders": ["uncertain", "unclassified"],
        "class_to_guidance": {
            "centered": "continue_alignment",
            "camera_left": "step_right",
            "camera_right": "step_left",
        },
        "total_images": len(records),
        "splits": {
            split: {
                class_name: counts[(split, class_name)]
                for class_name in ("centered", "camera_left", "camera_right")
            }
            for split in ("train", "val")
        },
    }
    (output / "dataset-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
