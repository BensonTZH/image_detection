"""Shared dataset, training, and serialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_dataset_layout(dataset: Path) -> Path:
    dataset = dataset.resolve()
    required = [dataset / kind / split for kind in ("images", "labels") for split in ("train", "val")]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))
    return dataset


def write_resolved_data_yaml(dataset: Path, output: Path) -> Path:
    dataset = validate_dataset_layout(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join([
            f"path: {json.dumps(str(dataset))}",
            "train: images/train",
            "val: images/val",
            "kpt_shape: [4, 3]",
            "flip_idx: [1, 0, 3, 2]",
            "names:",
            "  0: slot",
            "",
        ]),
        encoding="utf-8",
    )
    return output.resolve()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
