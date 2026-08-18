#!/usr/bin/env python3
"""Render deterministic label montages for visual dataset QA."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COLORS = ("#36c5f0", "#2eb67d", "#ecb22e", "#e01e5a")
POINT_NAMES = ("TL", "TR", "BR", "BL")


def label_path(dataset: Path, split: str, image_path: Path) -> Path:
    return dataset / "labels" / split / f"{image_path.stem}.txt"


def draw_tile(image_path: Path, labels: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((size[0], size[1] - 30), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, "#111827")
    offset_x = (size[0] - image.width) // 2
    offset_y = (size[1] - 30 - image.height) // 2
    tile.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()

    if not labels.exists():
        draw.rectangle((4, 4, 120, 22), fill="#991b1b")
        draw.text((8, 8), "BACKGROUND NEGATIVE", fill="white", font=font)
    else:
        for row in labels.read_text(encoding="utf-8").splitlines():
            values = [float(value) for value in row.split()]
            if len(values) != 17:
                continue
            cx, cy, width, height = values[1:5]
            draw.rectangle(
                (
                    offset_x + (cx - width / 2) * image.width,
                    offset_y + (cy - height / 2) * image.height,
                    offset_x + (cx + width / 2) * image.width,
                    offset_y + (cy + height / 2) * image.height,
                ),
                outline="white",
                width=2,
            )
            points = []
            for index in range(4):
                x, y, visibility = values[5 + index * 3 : 8 + index * 3]
                point = (offset_x + x * image.width, offset_y + y * image.height)
                points.append(point)
                if visibility > 0:
                    radius = 4
                    draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=COLORS[index])
                    draw.text((point[0] + 5, point[1] - 6), POINT_NAMES[index], fill=COLORS[index], font=font)
            draw.line(points + [points[0]], fill="white", width=2)

    name = image_path.name if len(image_path.name) <= 42 else image_path.name[:39] + "..."
    draw.text((8, size[1] - 21), name, fill="white", font=font)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase-1"))
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    columns, tile_size = 4, (360, 270)

    for split in ("train", "val"):
        image_dir = args.dataset / "images" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        backgrounds = [path for path in images if not label_path(args.dataset, split, path).exists()]
        labeled = [path for path in images if label_path(args.dataset, split, path).exists()]
        selected = rng.sample(backgrounds, min(4, len(backgrounds), args.samples))
        selected += rng.sample(labeled, min(args.samples - len(selected), len(labeled)))
        rng.shuffle(selected)
        rows = (len(selected) + columns - 1) // columns
        montage = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "#030712")
        for index, image_path in enumerate(selected):
            montage.paste(
                draw_tile(image_path, label_path(args.dataset, split, image_path), tile_size),
                ((index % columns) * tile_size[0], (index // columns) * tile_size[1]),
            )
        output = args.output / f"{split}-label-montage.jpg"
        montage.save(output, quality=90)
        print(output)


if __name__ == "__main__":
    main()
