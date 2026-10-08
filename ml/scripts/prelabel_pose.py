#!/usr/bin/env python3
"""Create reviewable YOLO-pose prelabels from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
POINT_NAMES = ("TL", "TR", "BR", "BL")
POINT_COLORS = ("#22d3ee", "#4ade80", "#fbbf24", "#fb7185")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Manifest whose filenames and source UUIDs must be excluded; may be repeated",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--corner-mode",
        choices=("raw", "box"),
        default="raw",
        help="Use model keypoints or initialize the four points from the detected box",
    )
    return parser.parse_args()


def source_id(path: Path) -> str:
    """Return the leading UUID used to group captures from the same source."""
    return path.name[:36]


def load_exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    filenames: set[str] = set()
    source_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("images", [])
        filenames.update(row["filename"] for row in rows)
        source_ids.update(row["source_id"] for row in rows)
    return filenames, source_ids


def select_images(args: argparse.Namespace) -> list[Path]:
    used_names, used_sources = load_exclusions(args.exclude_manifest)
    candidates = sorted(
        path
        for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    eligible = [
        path
        for path in candidates
        if path.name not in used_names and source_id(path) not in used_sources
    ]
    if len(eligible) < args.limit:
        raise RuntimeError(f"Only {len(eligible)} eligible images remain; requested {args.limit}")
    return eligible[: args.limit]


def write_label(path: Path, box: list[float], points: list[list[float]]) -> None:
    values: list[float | int] = [0, *box]
    for x, y in points:
        values.extend((x, y, 2))
    path.write_text(" ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in values) + "\n", encoding="utf-8")


def draw_overlay(result, output: Path, box: list[float], points: list[list[float]], confidence: float) -> None:
    image = Image.fromarray(result.orig_img[..., ::-1]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    cx, cy, bw, bh = box
    xyxy = (
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    )
    draw.rectangle(xyxy, outline="white", width=max(2, width // 350))
    pixel_points = [(x * width, y * height) for x, y in points]
    draw.line(pixel_points + [pixel_points[0]], fill="#a3e635", width=max(3, width // 250))
    radius = max(5, width // 180)
    for index, (x, y) in enumerate(pixel_points):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=POINT_COLORS[index])
        draw.text((x + radius + 2, y - radius), POINT_NAMES[index], fill=POINT_COLORS[index], font=font)
    draw.rectangle((8, 8, 205, 36), fill="#111827")
    draw.text((14, 15), f"slot {confidence:.1%}", fill="white", font=font)
    image.save(output, quality=92)


def no_detection_overlay(result, output: Path) -> None:
    image = Image.fromarray(result.orig_img[..., ::-1]).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 180, 40), fill="#991b1b")
    draw.text((15, 17), "NO DETECTION", fill="white", font=ImageFont.load_default())
    image.save(output, quality=92)


def make_montage(overlays: list[Path], output: Path) -> None:
    columns, tile_width, tile_height = 4, 320, 270
    rows = (len(overlays) + columns - 1) // columns
    montage = Image.new("RGB", (columns * tile_width, rows * tile_height), "#030712")
    for index, path in enumerate(overlays):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width, tile_height - 24), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), "#111827")
        tile.paste(image, ((tile_width - image.width) // 2, (tile_height - 24 - image.height) // 2))
        draw = ImageDraw.Draw(tile)
        name = path.stem if len(path.stem) <= 41 else path.stem[:38] + "..."
        draw.text((7, tile_height - 18), name, fill="white", font=ImageFont.load_default())
        montage.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    montage.save(output, quality=92)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    selected = select_images(args)

    from ultralytics import YOLO

    image_dir = args.output / "images" / "train"
    label_dir = args.output / "labels" / "train"
    overlay_dir = args.output / "overlays"
    for directory in (image_dir, label_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(path) for path in selected],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.5,
        max_det=1,
        device=args.device,
        verbose=False,
    )

    manifest = {
        "weights": str(args.weights.resolve()),
        "source": str(args.source.resolve()),
        "excluded_manifests": [str(path.resolve()) for path in args.exclude_manifest],
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "corner_mode": args.corner_mode,
        "images": [],
    }
    overlays: list[Path] = []
    for source, result in zip(selected, results, strict=True):
        copied_image = image_dir / source.name
        overlay = overlay_dir / source.name
        shutil.copy2(source, copied_image)
        overlays.append(overlay)
        row = {"filename": source.name, "source_id": source_id(source), "detected": False}
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            no_detection_overlay(result, overlay)
        else:
            confidence = float(result.boxes.conf[0].item())
            box = [float(value) for value in result.boxes.xywhn[0].tolist()]
            raw_points = [[float(value) for value in pair] for pair in result.keypoints.xyn[0].tolist()]
            if args.corner_mode == "box":
                cx, cy, width, height = box
                left, right = cx - width / 2, cx + width / 2
                top, bottom = cy - height / 2, cy + height / 2
                points = [[left, top], [right, top], [right, bottom], [left, bottom]]
            else:
                points = raw_points
            point_confidences = (
                [float(value) for value in result.keypoints.conf[0].tolist()]
                if result.keypoints.conf is not None
                else [None] * 4
            )
            label = label_dir / f"{source.stem}.txt"
            write_label(label, box, points)
            draw_overlay(result, overlay, box, points, confidence)
            row.update(
                detected=True,
                confidence=confidence,
                box=box,
                raw_keypoints=raw_points,
                keypoints=points,
                keypoint_confidences=point_confidences,
                label=str(label.relative_to(args.output)),
            )
        manifest["images"].append(row)

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "train.txt").write_text(
        "".join(f"data/images/train/{path.name}\n" for path in selected),
        encoding="utf-8",
    )
    (args.output / "data.yaml").write_text(
        "\n".join(
            (
                "path: .",
                "train: train.txt",
                "kpt_shape: [4, 3]",
                "flip_idx: [1, 0, 3, 2]",
                "names:",
                "  0: slot",
                "",
            )
        ),
        encoding="utf-8",
    )
    make_montage(overlays, args.output / "prelabel-montage.jpg")
    with zipfile.ZipFile(args.output / "cvat-annotations.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.output / "data.yaml", "data.yaml")
        archive.write(args.output / "train.txt", "train.txt")
        for label in sorted(label_dir.glob("*.txt")):
            archive.write(label, f"labels/train/{label.name}")
    detections = sum(bool(row["detected"]) for row in manifest["images"])
    print(f"Selected: {len(selected)}")
    print(f"Detected: {detections}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
