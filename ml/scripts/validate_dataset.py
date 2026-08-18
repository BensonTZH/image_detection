#!/usr/bin/env python3
"""Read-only validation for the cup-return YOLO pose dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_VALUES = 17  # class + bbox(4) + 4 keypoints(x, y, visibility)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    return orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0


def validate_label(path: Path) -> tuple[list[str], list[str], int]:
    errors: list[str] = []
    warnings: list[str] = []
    object_count = 0

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        object_count += 1
        parts = raw_line.split()
        prefix = f"{path}:{line_number}"
        if len(parts) != EXPECTED_VALUES:
            errors.append(f"{prefix}: expected {EXPECTED_VALUES} values, found {len(parts)}")
            continue
        try:
            values = [float(value) for value in parts]
        except ValueError:
            errors.append(f"{prefix}: contains a non-numeric value")
            continue

        if values[0] != 0 or not values[0].is_integer():
            errors.append(f"{prefix}: class must be 0")

        cx, cy, width, height = values[1:5]
        if not all(0 <= value <= 1 for value in (cx, cy, width, height)):
            errors.append(f"{prefix}: bounding-box values must be within [0, 1]")
        if width <= 0 or height <= 0:
            errors.append(f"{prefix}: bounding-box width and height must be positive")

        keypoints: list[tuple[float, float]] = []
        all_visible = True
        for index in range(4):
            x, y, visibility = values[5 + index * 3 : 8 + index * 3]
            keypoints.append((x, y))
            if visibility not in (0, 1, 2):
                errors.append(f"{prefix}: keypoint {index} visibility must be 0, 1, or 2")
            if visibility <= 0:
                all_visible = False
            elif not (0 <= x <= 1 and 0 <= y <= 1):
                errors.append(f"{prefix}: visible keypoint {index} is outside [0, 1]")

        if all_visible:
            tl, tr, br, bl = keypoints
            if segments_cross(tl, tr, br, bl) or segments_cross(tr, br, bl, tl):
                errors.append(f"{prefix}: keypoint polygon self-intersects")
            signed_double_area = sum(
                keypoints[index][0] * keypoints[(index + 1) % 4][1]
                - keypoints[(index + 1) % 4][0] * keypoints[index][1]
                for index in range(4)
            )
            if signed_double_area <= 0:
                warnings.append(f"{prefix}: unexpected TL, TR, BR, BL winding")
            if (tl[1] + tr[1]) / 2 > (bl[1] + br[1]) / 2:
                warnings.append(f"{prefix}: top corners average below bottom corners")

    if object_count == 0:
        warnings.append(f"{path}: empty label file")
    return errors, warnings, object_count


def validate(dataset: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset": str(dataset.resolve()),
        "splits": {},
        "errors": [],
        "warnings": [],
        "duplicate_groups": [],
    }
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    stems: dict[str, set[str]] = {}

    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            report["errors"].append(f"{split}: expected {image_dir} and {label_dir}")
            report["splits"][split] = {"images": 0, "labels": 0, "background_negatives": 0, "objects": 0}
            stems[split] = set()
            continue

        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        images_by_stem = {path.stem: path for path in images}
        labels_by_stem = {path.stem: path for path in labels}
        stems[split] = set(images_by_stem)
        dimensions: Counter[str] = Counter()
        split_report = {
            "images": len(images),
            "labels": len(labels),
            "background_negatives": len(set(images_by_stem) - set(labels_by_stem)),
            "objects": 0,
            "image_dimensions": {},
        }

        for stem in sorted(set(labels_by_stem) - set(images_by_stem)):
            report["errors"].append(f"{split}: label has no matching image: {labels_by_stem[stem]}")

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
            except Exception as exc:
                report["errors"].append(f"{image_path}: cannot be decoded ({exc})")
            hashes[file_hash(image_path)].append((split, str(image_path)))

        for label_path in labels:
            errors, warnings, object_count = validate_label(label_path)
            report["errors"].extend(errors)
            report["warnings"].extend(warnings)
            split_report["objects"] += object_count

        split_report["image_dimensions"] = dict(dimensions.most_common())
        report["splits"][split] = split_report

    for stem in sorted(stems["train"] & stems["val"]):
        report["errors"].append(f"train/val filename overlap: {stem}")

    for matches in hashes.values():
        if len(matches) < 2:
            continue
        report["duplicate_groups"].append([{"split": split, "path": path} for split, path in matches])
        paths = ", ".join(path for _, path in matches)
        if len({split for split, _ in matches}) > 1:
            report["errors"].append(f"cross-split duplicate image: {paths}")
        else:
            report["warnings"].append(f"duplicate image in one split: {paths}")

    report["summary"] = {
        "status": "PASS" if not report["errors"] else "FAIL",
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"]),
        "duplicate_group_count": len(report["duplicate_groups"]),
    }
    return report


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Phase 1 Dataset Validation Report",
        "",
        f"**Status:** {summary['status']}",
        "",
        "| Split | Images | Labels | Background negatives | Objects |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, values in report["splits"].items():
        lines.append(
            f"| {split} | {values['images']} | {values['labels']} | "
            f"{values['background_negatives']} | {values['objects']} |"
        )
    lines.extend([
        "",
        f"- Errors: {summary['error_count']}",
        f"- Warnings: {summary['warning_count']}",
        f"- Duplicate groups: {summary['duplicate_group_count']}",
        "",
        "## Errors",
        "",
    ])
    lines.extend(f"- {item}" for item in report["errors"][:100])
    if not report["errors"]:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report["warnings"][:100])
    if not report["warnings"]:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--json", type=Path, default=Path("reports/phase-1/dataset-validation.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/phase-1/dataset-validation.md"))
    args = parser.parse_args()
    report = validate(args.dataset)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown.write_text(to_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
