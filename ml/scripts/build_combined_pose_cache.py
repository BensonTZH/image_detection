#!/usr/bin/env python3
"""Extend validated Ultralytics pose caches without rescanning cloud-backed labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics.data.utils import get_hash, img2label_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    return parser.parse_args()


def new_label_record(image: Path, label: Path) -> dict[str, object]:
    fields = [float(value) for value in label.read_text(encoding="utf-8").split()]
    if len(fields) != 17:
        raise ValueError(f"{label}: expected 17 values, found {len(fields)}")
    with Image.open(image) as opened:
        opened.verify()
        width, height = opened.size
    return {
        "im_file": str(image.resolve()),
        "shape": (height, width),
        "cls": np.asarray([[fields[0]]], dtype=np.float32),
        "bboxes": np.asarray([fields[1:5]], dtype=np.float32),
        "segments": [],
        "keypoints": np.asarray(fields[5:], dtype=np.float32).reshape(1, 4, 3),
        "normalized": True,
        "bbox_format": "xywh",
    }


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    extension = args.extension.resolve()
    for split in ("train", "val"):
        base_cache_path = base / "labels" / f"{split}.cache"
        base_cache = np.load(base_cache_path, allow_pickle=True).item()
        additions = []
        for label in sorted((extension / "labels" / split).glob("*.txt")):
            matches = list((extension / "images" / split).glob(f"{label.stem}.*"))
            if len(matches) != 1:
                raise ValueError(f"Expected one image for {label.stem}, found {len(matches)}")
            additions.append(new_label_record(matches[0], label))

        labels = sorted([*base_cache["labels"], *additions], key=lambda item: item["im_file"])
        im_files = [str(item["im_file"]) for item in labels]
        label_files = img2label_paths(im_files)
        total = len(labels)
        cache = {
            "labels": labels,
            "hash": get_hash(label_files + im_files),
            "results": (total, 0, 0, 0, total),
            "msgs": list(base_cache.get("msgs", [])),
            "version": base_cache["version"],
        }
        cache_path = extension / "labels" / f"{split}.cache"
        np.save(str(cache_path), cache)
        cache_path.with_suffix(cache_path.suffix + ".npy").replace(cache_path)
        print(f"{split}: {len(base_cache['labels'])} base + {len(additions)} new = {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
