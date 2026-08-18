"""Shared ONNX export and model-comparison utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Length mismatch: {len(left)} != {len(right)}")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def shape_for_json(shape: Sequence[object]) -> list[object]:
    return [value if isinstance(value, (str, int)) else str(value) for value in shape]
