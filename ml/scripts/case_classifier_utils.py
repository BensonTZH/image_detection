"""Utilities for the success/error cup-readability classifier dataset."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")
CLASSES = ("success", "error")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
SOURCE_FOLDERS = {"success": "success2", "error": "error2"}
UUID_PREFIX = re.compile(
    r"^(?P<source>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?:-|$)"
)


@dataclass(frozen=True)
class CaseImage:
    path: Path
    label: str
    source_id: str


def source_id_from_filename(path: Path | str) -> str:
    """Return the capture/source UUID at the start of a case filename."""
    name = Path(path).name
    match = UUID_PREFIX.match(name)
    if not match:
        raise ValueError(f"Filename does not start with a source UUID: {name}")
    return match.group("source").lower()


def scan_case_dataset(root: Path) -> list[CaseImage]:
    """Read the image-level success/error folders without modifying them."""
    root = root.resolve()
    records: list[CaseImage] = []
    missing: list[str] = []
    for label, folder in SOURCE_FOLDERS.items():
        class_dir = root / folder
        if not class_dir.is_dir():
            missing.append(str(class_dir))
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(CaseImage(path.resolve(), label, source_id_from_filename(path)))
    if missing:
        raise FileNotFoundError("Missing case directories: " + ", ".join(missing))
    if not records:
        raise ValueError(f"No supported images found under {root}")
    return records


def validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("Exactly three split ratios are required: train val test")
    parsed = tuple(float(value) for value in ratios)
    if any(value <= 0 for value in parsed):
        raise ValueError("Split ratios must all be greater than zero")
    if abs(sum(parsed) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must total 1.0, got {sum(parsed):.6f}")
    return parsed  # type: ignore[return-value]


def grouped_split(
    records: Sequence[CaseImage],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 17,
) -> dict[str, str]:
    """Assign every source UUID to one split while balancing both classes.

    Related frames often share the same UUID. Grouping prevents those frames
    from appearing in both training and evaluation, which would inflate scores.
    """
    ratios = validate_ratios(ratios)
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record.label not in CLASSES:
            raise ValueError(f"Unsupported label: {record.label}")
        grouped[record.source_id][record.label] += 1

    totals = Counter(record.label for record in records)
    total_images = len(records)
    targets = {
        split: {
            "total": total_images * ratio,
            **{label: totals[label] * ratio for label in CLASSES},
        }
        for split, ratio in zip(SPLITS, ratios)
    }
    counts = {
        split: {"total": 0, **{label: 0 for label in CLASSES}}
        for split in SPLITS
    }

    rng = random.Random(seed)
    groups = list(grouped.items())
    rng.shuffle(groups)
    # Largest and mixed-label groups are hardest to place, so place them first.
    groups.sort(
        key=lambda item: (sum(item[1].values()), min(item[1].values() or [0])),
        reverse=True,
    )

    assignment: dict[str, str] = {}
    for source_id, label_counts in groups:
        candidate_splits = list(SPLITS)
        rng.shuffle(candidate_splits)
        best_split = min(
            candidate_splits,
            key=lambda split: _placement_score(
                counts,
                targets,
                split,
                label_counts,
            ),
        )
        assignment[source_id] = best_split
        group_total = sum(label_counts.values())
        counts[best_split]["total"] += group_total
        for label in CLASSES:
            counts[best_split][label] += label_counts[label]

    return assignment


def _placement_score(
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, float]],
    candidate: str,
    addition: Counter[str],
) -> float:
    score = 0.0
    addition_total = sum(addition.values())
    for split in SPLITS:
        for key in ("total", *CLASSES):
            value = counts[split][key]
            if split == candidate:
                value += addition_total if key == "total" else addition[key]
            target = targets[split][key]
            score += ((value - target) / max(target, 1.0)) ** 2
            if value > target:
                # A small extra cost avoids filling a split far beyond target.
                score += 0.25 * ((value - target) / max(target, 1.0)) ** 2
    return score


def split_summary(
    records: Sequence[CaseImage],
    assignment: dict[str, str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    total = len(records)
    for split in SPLITS:
        selected = [record for record in records if assignment[record.source_id] == split]
        label_counts = Counter(record.label for record in selected)
        summary[split] = {
            "images": len(selected),
            "percentage": round(100 * len(selected) / max(total, 1), 3),
            "sources": len({record.source_id for record in selected}),
            "classes": {label: label_counts[label] for label in CLASSES},
        }
    return summary


def expanded_crop_box(
    xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """Expand a detector box and clamp it to the source image."""
    if len(xyxy) != 4:
        raise ValueError("xyxy must contain four values")
    if margin < 0:
        raise ValueError("margin must not be negative")
    left, top, right, bottom = (float(value) for value in xyxy)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box: {tuple(xyxy)}")
    pad_x = (right - left) * margin
    pad_y = (bottom - top) * margin
    return (
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(image_width, int(right + pad_x + 0.9999)),
        min(image_height, int(bottom + pad_y + 0.9999)),
    )


def letterbox_square(
    image: Image.Image,
    size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    """Fit a complete image inside a square without cropping or stretching it."""
    if size <= 0:
        raise ValueError("Letterbox size must be greater than zero")
    source = image.convert("RGB")
    contained = ImageOps.contain(
        source,
        (size, size),
        method=Image.Resampling.BILINEAR,
    )
    output = Image.new("RGB", (size, size), fill)
    offset = ((size - contained.width) // 2, (size - contained.height) // 2)
    output.paste(contained, offset)
    return output


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.width, image.height


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepared_records(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in manifest.get("images", []):
        if item.get("status") == "prepared":
            yield item
