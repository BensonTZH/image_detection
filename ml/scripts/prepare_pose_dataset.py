#!/usr/bin/env python3
"""Build a validated YOLO pose dataset from one or more CVAT exports."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
SPLIT_MODES = {
    "70-15-15": {"train": 0.70, "val": 0.15, "test": 0.15},
    "80-20": {"train": 0.80, "val": 0.20},
}
UUID_PREFIX = re.compile(
    r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    source_id: str
    label_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument(
        "--label-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing YOLO pose .txt files; repeat for each export",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_MODES),
        default="70-15-15",
        help="Dataset split mode (default: 70-15-15)",
    )
    parser.add_argument("--boundary-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--hardlink-images",
        action="store_true",
        help="Hard-link source images into the dataset instead of copying them",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symbolically link source images into the dataset instead of copying them",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate annotations and report the planned split without copying images",
    )
    return parser.parse_args()


def normalized_label(path: Path, tolerance: float) -> str:
    rows = [row for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one annotation row, found {len(rows)}")

    fields = rows[0].split()
    if len(fields) != 17:
        raise ValueError(f"{path}: expected 17 values, found {len(fields)}")
    values = [float(field) for field in fields]
    if values[0] != 0:
        raise ValueError(f"{path}: expected class 0, found {values[0]}")

    for index in (7, 10, 13, 16):
        if values[index] not in (0, 1, 2):
            raise ValueError(f"{path}: invalid keypoint visibility at field {index}")

    coordinate_indices = (1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15)
    for index in coordinate_indices:
        value = values[index]
        if value < -tolerance or value > 1 + tolerance:
            raise ValueError(f"{path}: coordinate {index}={value} is outside [0, 1]")
        values[index] = min(1.0, max(0.0, value))
    if values[3] <= 0 or values[4] <= 0:
        raise ValueError(f"{path}: bounding-box width and height must be positive")

    points = [values[5 + offset * 3 : 7 + offset * 3] for offset in range(4)]
    tl, tr, br, bl = points
    if not (
        tl[0] < tr[0]
        and bl[0] < br[0]
        and max(tl[1], tr[1]) < min(bl[1], br[1])
    ):
        raise ValueError(f"{path}: keypoints must be ordered TL, TR, BR, BL")

    return (
        " ".join(
            str(int(value)) if index in (0, 7, 10, 13, 16) else f"{value:.6f}"
            for index, value in enumerate(values)
        )
        + "\n"
    )


def find_image(source_images: Path, stem: str) -> Path:
    matches = [
        source_images / f"{stem}{suffix}"
        for suffix in IMAGE_SUFFIXES
        if (source_images / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one source image for {stem}, found {len(matches)}")
    return matches[0]


def source_id(stem: str) -> str:
    match = UUID_PREFIX.match(stem)
    return match.group(1).lower() if match else stem


def load_records(args: argparse.Namespace) -> list[Record]:
    labels_by_stem: dict[str, Path] = {}
    for label_dir in args.label_dir:
        for label in sorted(label_dir.glob("*.txt")):
            previous = labels_by_stem.get(label.stem)
            if previous is not None:
                raise ValueError(f"Duplicate annotation for {label.stem}: {previous} and {label}")
            labels_by_stem[label.stem] = label
    if not labels_by_stem:
        raise ValueError("No label files were found")

    return [
        Record(
            image=find_image(args.source_images, stem),
            label=label,
            source_id=source_id(stem),
            label_text=normalized_label(label, args.boundary_tolerance),
        )
        for stem, label in sorted(labels_by_stem.items())
    ]


def grouped_split(
    records: list[Record], seed: int, split_ratios: dict[str, float]
) -> dict[str, str]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.source_id].append(record)

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    group_ids.sort(key=lambda group_id: len(groups[group_id]), reverse=True)

    targets = {split: len(records) * ratio for split, ratio in split_ratios.items()}
    counts = Counter()
    assignment: dict[str, str] = {}
    for group_id in group_ids:
        size = len(groups[group_id])
        split = min(
            split_ratios,
            key=lambda candidate: sum(
                (
                    counts[name]
                    + (size if name == candidate else 0)
                    - targets[name]
                )
                ** 2
                / max(targets[name], 1)
                for name in split_ratios
            ),
        )
        assignment[group_id] = split
        counts[split] += size

    if any(counts[split] == 0 for split in split_ratios):
        raise ValueError(f"Unable to create non-empty splits: {dict(counts)}")
    return assignment


def prepare_output(output: Path, split_ratios: dict[str, float]) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    for kind in ("images", "labels"):
        for split in split_ratios:
            (output / kind / split).mkdir(parents=True, exist_ok=False)


def main() -> int:
    args = parse_args()
    args.source_images = args.source_images.resolve()
    args.label_dir = [path.resolve() for path in args.label_dir]
    args.output = args.output.resolve()

    split_ratios = SPLIT_MODES[args.split]
    records = load_records(args)
    assignment = grouped_split(records, args.seed, split_ratios)
    planned_counts = Counter(assignment[record.source_id] for record in records)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "images": len(records),
                    "sources": len({record.source_id for record in records}),
                    "planned_splits": dict(planned_counts),
                },
                indent=2,
            )
        )
        return 0
    prepare_output(args.output, split_ratios)

    manifest_rows: list[dict[str, str]] = []
    for record in records:
        split = assignment[record.source_id]
        image_target = args.output / "images" / split / record.image.name
        label_target = args.output / "labels" / split / f"{record.image.stem}.txt"
        if args.hardlink_images and args.symlink_images:
            raise ValueError("Choose only one of --hardlink-images or --symlink-images")
        if args.symlink_images:
            os.symlink(record.image, image_target)
        elif args.hardlink_images:
            os.link(record.image, image_target)
        else:
            shutil.copy2(record.image, image_target)
        label_target.write_text(record.label_text, encoding="utf-8")
        manifest_rows.append(
            {
                "filename": record.image.name,
                "source_id": record.source_id,
                "split": split,
                "image": str(image_target.relative_to(args.output)),
                "label": str(label_target.relative_to(args.output)),
                "source_image": str(record.image),
                "source_label": str(record.label),
            }
        )

    manifest_rows.sort(key=lambda row: row["filename"])
    split_counts = Counter(row["split"] for row in manifest_rows)
    source_counts = {
        split: len({row["source_id"] for row in manifest_rows if row["split"] == split})
        for split in split_ratios
    }
    manifest = {
        "source_images": str(args.source_images),
        "label_dirs": [str(path) for path in args.label_dir],
        "seed": args.seed,
        "ratios": split_ratios,
        "split_summary": {
            split: {"images": split_counts[split], "sources": source_counts[split]}
            for split in split_ratios
        },
        "images": manifest_rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for split in split_ratios:
        rows = [row for row in manifest_rows if row["split"] == split]
        (args.output / f"{split}.txt").write_text(
            "".join(f"images/{split}/{row['filename']}\n" for row in rows),
            encoding="utf-8",
        )

    yaml_rows = [
        f"path: {args.output}",
        "train: train.txt",
        "val: val.txt",
    ]
    if "test" in split_ratios:
        yaml_rows.append("test: test.txt")
    yaml_rows.extend(
        (
            "kpt_shape: [4, 3]",
            "flip_idx: [1, 0, 3, 2]",
            "names:",
            "  0: slot",
            "",
        )
    )
    (args.output / "data.yaml").write_text(
        "\n".join(yaml_rows), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "images": len(records),
                "sources": len({record.source_id for record in records}),
                "splits": dict(split_counts),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
