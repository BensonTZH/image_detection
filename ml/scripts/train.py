#!/usr/bin/env python3
"""Train the Phase 2 YOLO11n-pose baseline."""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from training_utils import choose_device, write_json, write_resolved_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--smoke", action="store_true", help="Run one short setup-validation epoch")
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

    project = Path("runs/phase-2").resolve()
    name = "smoke" if args.smoke else args.name
    resolved_yaml = write_resolved_data_yaml(args.dataset, project / "configs" / "resolved-data.yaml")
    device = choose_device(args.device)
    epochs = 1 if args.smoke else args.epochs
    imgsz = min(args.imgsz, 320) if args.smoke else args.imgsz
    batch = min(args.batch, 2) if args.smoke else args.batch
    write_json(project / name / "run-config.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "dataset": str(args.dataset.resolve()),
        "resolved_data_yaml": str(resolved_yaml),
        "model": args.model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "smoke": args.smoke,
    })
    print(f"Training on device: {device}")
    print(f"Resolved dataset config: {resolved_yaml}")
    model = YOLO(args.model)
    results = model.train(
        data=str(resolved_yaml), project=str(project), name=name, exist_ok=True,
        epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=args.workers,
        patience=args.patience, seed=args.seed, deterministic=True, pretrained=True,
        optimizer="auto", cache=False, plots=True, val=not args.smoke,
        fraction=0.05 if args.smoke else 1.0,
        fliplr=0.0, flipud=0.0, degrees=5.0, translate=0.05, scale=0.15,
        perspective=0.0005,
    )
    write_json(project / name / "result-summary.json", getattr(results, "results_dict", {}))
    print(f"Run saved to: {project / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
