#!/usr/bin/env python3
"""Train an isolated three-class slot-angle YOLO pose experiment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--name", default="baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python3 -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    data_yaml = (args.dataset / "data.yaml").resolve()
    model_path = args.model.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset config: {data_yaml}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing starting checkpoint: {model_path}")

    project = Path("runs/angle-pose").resolve()
    output = project / args.name
    device = choose_device(args.device)
    write_json(
        output / "run-config.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "dataset": str(args.dataset.resolve()),
            "data_yaml": str(data_yaml),
            "starting_model": str(model_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": device,
            "workers": args.workers,
            "patience": args.patience,
            "seed": args.seed,
            "horizontal_flip_disabled": True,
        },
    )
    print(f"Training on device: {device}")
    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_yaml),
        project=str(project),
        name=args.name,
        exist_ok=True,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        cache=False,
        plots=True,
        val=True,
        fliplr=0.0,
        flipud=0.0,
        degrees=2.0,
        translate=0.02,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
    )
    write_json(output / "result-summary.json", getattr(results, "results_dict", {}))
    print(f"Run saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
