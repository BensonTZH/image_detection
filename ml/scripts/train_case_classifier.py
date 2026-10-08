#!/usr/bin/env python3
"""Train a binary classifier for readable success versus unreadable error crops."""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .training_utils import choose_device, write_json
    from .validate_case_classifier import validate
except ImportError:  # Direct execution: python ml/scripts/train_case_classifier.py
    from training_utils import choose_device, write_json
    from validate_case_classifier import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/case-classifier"))
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--smoke", action="store_true", help="Run one short setup-validation epoch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(args.dataset)
    if report["summary"]["status"] != "PASS":
        print("Prepared dataset validation failed:", file=sys.stderr)
        for error in report["errors"][:20]:
            print(f"- {error}", file=sys.stderr)
        return 2
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    project = Path("runs/case-classifier").resolve()
    name = "smoke" if args.smoke else args.name
    device = choose_device(args.device)
    epochs = 1 if args.smoke else args.epochs
    batch = min(args.batch, 8) if args.smoke else args.batch
    fraction = 0.10 if args.smoke else 1.0
    run_dir = project / name
    write_json(run_dir / "run-config.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "device": device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "fraction": fraction,
        "smoke": args.smoke,
        "augmentation": {
            "auto_augment": None,
            "erasing": 0.0,
            "degrees": 2.0,
            "translate": 0.02,
            "scale": 0.05,
            "hsv_h": 0.01,
            "hsv_s": 0.15,
            "hsv_v": 0.15,
            "fliplr": 0.0,
            "flipud": 0.0,
        },
    })
    print(f"Training quality classifier on {device}")
    print(f"Dataset: {args.dataset.resolve()}")
    model = YOLO(args.model)
    results = model.train(
        data=str(args.dataset.resolve()),
        project=str(project),
        name=name,
        exist_ok=True,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        cache=False,
        plots=True,
        val=not args.smoke,
        fraction=fraction,
        auto_augment=None,
        erasing=0.0,
        degrees=2.0,
        translate=0.02,
        scale=0.05,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.15,
        fliplr=0.0,
        flipud=0.0,
    )
    write_json(run_dir / "result-summary.json", getattr(results, "results_dict", {}))
    print(f"Run saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
