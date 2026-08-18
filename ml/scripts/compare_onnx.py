#!/usr/bin/env python3
"""Compare PyTorch and ONNX post-processed predictions on fixed validation images."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from training_utils import write_json
from onnx_utils import max_abs_difference

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("runs/phase-2/baseline/weights/best.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("runs/phase-3/model/slot-pose.onnx"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--imgsz", type=int, default=640)
    # These are measured after predictions are scaled back to the original
    # phone-image resolution. ONNX and PyTorch can differ slightly around weak
    # or out-of-frame keypoints even when their 640px model outputs agree.
    parser.add_argument("--box-tolerance", type=float, default=3.0)
    parser.add_argument("--keypoint-tolerance", type=float, default=32.0)
    parser.add_argument("--confidence-tolerance", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("runs/phase-3/parity/parity-report.json"))
    return parser.parse_args()


def flatten(tensor) -> list[float]:
    return tensor.detach().cpu().reshape(-1).tolist()


def detection(result) -> dict[str, object] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    keypoints = [] if result.keypoints is None else flatten(result.keypoints.xy[0])
    return {
        "box": flatten(result.boxes.xyxy[0]),
        "confidence": float(result.boxes.conf[0].detach().cpu()),
        "class": int(result.boxes.cls[0].detach().cpu()),
        "keypoints": keypoints,
    }


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python3 -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2
    for path in (args.weights, args.onnx):
        if not path.is_file():
            print(f"Model not found: {path}", file=sys.stderr)
            return 2

    candidates = sorted(path for path in (args.dataset / "images" / "val").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    selected = random.Random(args.seed).sample(candidates, min(args.samples, len(candidates)))
    # A Python list of images is collated into one tensor by Ultralytics even
    # when batch=1 is requested. Invoke each backend per image so the parity
    # test exactly matches the fixed [1, 3, 640, 640] browser contract.
    prediction_args = dict(imgsz=args.imgsz, conf=0.25, iou=0.7, max_det=1, verbose=False)
    pytorch_model = YOLO(str(args.weights))
    onnx_model = YOLO(str(args.onnx), task="pose")
    pytorch_results = []
    onnx_results = []
    for image in selected:
        pytorch_results.append(pytorch_model.predict(source=str(image), device="cpu", **prediction_args)[0])
        onnx_results.append(onnx_model.predict(source=str(image), device="cpu", **prediction_args)[0])

    cases = []
    failures = []
    maxima = {"box_pixels": 0.0, "keypoint_pixels": 0.0, "confidence": 0.0}
    for image, pytorch_result, onnx_result in zip(selected, pytorch_results, onnx_results):
        pt = detection(pytorch_result)
        ox = detection(onnx_result)
        case = {"image": str(image), "pytorch": pt, "onnx": ox, "status": "PASS"}
        if (pt is None) != (ox is None):
            case["status"] = "FAIL"
            case["reason"] = "Detection presence differs"
        elif pt is not None and ox is not None:
            differences = {
                "box_pixels": max_abs_difference(pt["box"], ox["box"]),
                "keypoint_pixels": max_abs_difference(pt["keypoints"], ox["keypoints"]),
                "confidence": abs(float(pt["confidence"]) - float(ox["confidence"])),
                "class_matches": pt["class"] == ox["class"],
            }
            case["differences"] = differences
            for key in maxima:
                maxima[key] = max(maxima[key], float(differences[key]))
            if (
                differences["box_pixels"] > args.box_tolerance
                or differences["keypoint_pixels"] > args.keypoint_tolerance
                or differences["confidence"] > args.confidence_tolerance
                or not differences["class_matches"]
            ):
                case["status"] = "FAIL"
                case["reason"] = "Difference exceeds tolerance"
        if case["status"] == "FAIL":
            failures.append({"image": str(image), "reason": case.get("reason")})
        cases.append(case)

    report = {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "pytorch_model": str(args.weights.resolve()),
        "onnx_model": str(args.onnx.resolve()),
        "sample_count": len(cases),
        "seed": args.seed,
        "tolerances": {
            "box_pixels": args.box_tolerance,
            "keypoint_pixels": args.keypoint_tolerance,
            "confidence": args.confidence_tolerance,
        },
        "maximum_differences": maxima,
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures),
        "failures": failures,
        "cases": cases,
    }
    write_json(args.output, report)
    print(f"Parity: {report['status']}")
    print(f"Maximum differences: {maxima}")
    print(f"Report: {args.output.resolve()}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
