#!/usr/bin/env python3
"""Export and inspect the accepted Phase 2 checkpoint as browser-ready ONNX."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from training_utils import write_json
from onnx_utils import sha256, shape_for_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("runs/phase-2/baseline/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase-3/model/slot-pose.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--no-simplify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import onnx
        import onnxruntime as ort
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing dependency. Run: python3 -m pip install -r ml/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"Checkpoint not found: {args.weights}", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=False,
        nms=False,
        batch=1,
        device="cpu",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, args.output)

    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    inputs = [{"name": item.name, "shape": shape_for_json(item.shape), "type": item.type} for item in session.get_inputs()]
    outputs = [{"name": item.name, "shape": shape_for_json(item.shape), "type": item.type} for item in session.get_outputs()]
    expected_input = [1, 3, args.imgsz, args.imgsz]
    expected_output = [1, 17, 8400]
    contract_errors: list[str] = []
    if len(inputs) != 1 or inputs[0]["shape"] != expected_input:
        contract_errors.append(f"Expected one input shaped {expected_input}, found {inputs}")
    if len(outputs) != 1 or outputs[0]["shape"] != expected_output:
        contract_errors.append(f"Expected one output shaped {expected_output}, found {outputs}")
    opsets = {item.domain or "ai.onnx": item.version for item in onnx_model.opset_import}
    if opsets.get("ai.onnx") != args.opset:
        contract_errors.append(f"Expected ai.onnx opset {args.opset}, found {opsets}")

    metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "ultralytics": ultralytics.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "source_weights": str(args.weights.resolve()),
        "model": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "size_bytes": args.output.stat().st_size,
        "opsets": opsets,
        "simplified": not args.no_simplify,
        "nms_embedded": False,
        "inputs": inputs,
        "outputs": outputs,
        "contract_status": "PASS" if not contract_errors else "FAIL",
        "contract_errors": contract_errors,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    write_json(metadata_path, metadata)
    print(f"ONNX model: {args.output.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    print(f"Contract: {metadata['contract_status']}")
    return 0 if not contract_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
