#!/usr/bin/env python3
"""Copy a three-class pose dataset and add one labelled angle export."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_IDS = {"centered": 0, "camera_left": 1, "camera_right": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new-labels", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--class-name", choices=CLASS_IDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def find_image(source: Path, stem: str) -> Path:
    matches = [p for p in source.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {stem}, found {len(matches)}")
    return matches[0]


def rewrite_class(label: Path, class_id: int) -> str:
    rows = [row for row in label.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"{label}: expected one annotation row, found {len(rows)}")
    fields = rows[0].split()
    if len(fields) != 17:
        raise ValueError(f"{label}: expected 17 pose values, found {len(fields)}")
    if fields[0] != "0":
        raise ValueError(f"{label}: expected exported class 0, found {fields[0]}")
    fields[0] = str(class_id)
    return " ".join(fields) + "\n"


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    labels = args.new_labels.resolve()
    images = args.source_images.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    records = [(label, find_image(images, label.stem)) for label in sorted(labels.glob("*.txt"))]
    if not records:
        raise ValueError(f"No labels found under {labels}")
    labelled_stems = {label.stem for label, _ in records}
    image_stems = {p.stem for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    if image_stems != labelled_stems:
        raise ValueError(
            f"Image/label mismatch: unlabelled={sorted(image_stems-labelled_stems)}, "
            f"missing_images={sorted(labelled_stems-image_stems)}"
        )

    base_names = {
        p.name for split in ("train", "val") for p in (base / "images" / split).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    }
    overlap = base_names & {image.name for _, image in records}
    if overlap:
        raise ValueError(f"New filenames overlap base dataset: {sorted(overlap)}")

    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (output / kind / split).mkdir(parents=True, exist_ok=False)
    shuffled = records[:]
    random.Random(args.seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * args.val_ratio))
    val_stems = {label.stem for label, _ in shuffled[:val_count]}
    class_id = CLASS_IDS[args.class_name]
    additions = Counter()
    for label, image in records:
        split = "val" if label.stem in val_stems else "train"
        shutil.copy2(image, output / "images" / split / image.name)
        (output / "labels" / split / f"{label.stem}.txt").write_text(
            rewrite_class(label, class_id), encoding="utf-8"
        )
        additions[split] += 1

    summary = {
        "base_dataset": str(base),
        "new_labels": str(labels),
        "source_images": str(images),
        "class_name": args.class_name,
        "class_id": class_id,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "additions": dict(additions),
        "total_added": len(records),
    }
    (output / "extension-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "data.yaml").write_text(
        "\n".join(
            [
                f'path: "{output}"',
                "train:",
                f'  - "{base / "images" / "train"}"',
                "  - images/train",
                "val:",
                f'  - "{base / "images" / "val"}"',
                "  - images/val",
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
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
