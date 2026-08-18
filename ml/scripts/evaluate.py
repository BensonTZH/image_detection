#!/usr/bin/env python3
"""Evaluate a Phase 2 pose checkpoint and render validation samples."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from training_utils import choose_device, write_json, write_resolved_data_yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--weights", type=Path, default=Path("runs/phase-2/baseline/weights/best.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--name", default="baseline-evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python3 -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"Checkpoint not found: {args.weights}", file=sys.stderr)
        return 2

    project = Path("runs/phase-2").resolve()
    resolved_yaml = write_resolved_data_yaml(args.dataset, project / "configs" / "resolved-data.yaml")
    device = choose_device(args.device)
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(resolved_yaml), imgsz=args.imgsz, batch=args.batch, device=device,
        project=str(project), name=args.name, exist_ok=True, plots=True,
    )
    images = sorted(path for path in (args.dataset / "images" / "val").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    selected = random.Random(args.seed).sample(images, min(args.samples, len(images)))
    prediction_dir = project / args.name / "predictions"
    model.predict(
        source=[str(path) for path in selected], imgsz=args.imgsz, device=device,
        conf=0.25, max_det=1, save=True, project=str(project / args.name),
        name="predictions", exist_ok=True, verbose=False,
    )
    write_json(project / args.name / "metrics.json", {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "ultralytics": ultralytics.__version__,
        "weights": str(args.weights.resolve()),
        "device": device,
        "imgsz": args.imgsz,
        "sample_count": len(selected),
        "prediction_directory": str(prediction_dir),
        "metrics": getattr(metrics, "results_dict", {}),
    })
    print(f"Metrics: {project / args.name / 'metrics.json'}")
    print(f"Predictions: {prediction_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
