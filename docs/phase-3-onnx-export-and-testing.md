# Phase 3 - ONNX Export and Parity Testing

## Purpose

Phase 3 converts the accepted PyTorch checkpoint into the ONNX model that will later run inside the phone browser. It verifies the graph, fixed input/output contract, opset, and prediction parity before any web-camera integration begins.

The export intentionally does not embed non-maximum suppression. Browser code in a later phase will decode candidates and apply NMS.

## Expected model contract

| Item | Expected value |
| --- | --- |
| Input name | `images` |
| Input shape | `[1, 3, 640, 640]` |
| Input type | FP32 |
| Output name | `output0` |
| Output shape | `[1, 17, 8400]` |
| Output type | FP32 |
| ONNX opset | 12 |
| Dynamic dimensions | No |
| Embedded NMS | No |
| Simplified graph | Yes |

The 17 output channels represent four box values, one class score, and four keypoints with x, y, and visibility/confidence values.

## Setup

```bash
cd /Users/bensontan/Desktop/image_detection
source .venv/bin/activate
python3 -m pip install -r ml/requirements.txt
```

## Test 1 - Automated helper tests

```bash
python3 -m unittest discover -s ml/tests -v
```

## Test 2 - Export and inspect ONNX

```bash
python3 ml/scripts/export_onnx.py \
  --weights runs/phase-2/baseline/weights/best.pt \
  --output runs/phase-3/model/slot-pose.onnx \
  --imgsz 640 \
  --opset 12
```

This performs Ultralytics export, graph simplification, ONNX checker validation, ONNX Runtime loading, and strict shape/opset checks.

Pass criteria:

- Command prints `Contract: PASS`.
- `runs/phase-3/model/slot-pose.onnx` exists.
- `slot-pose.metadata.json` reports the expected contract.
- ONNX Runtime loads the graph without warnings or errors.

## Test 3 - PyTorch/ONNX parity

```bash
python3 ml/scripts/compare_onnx.py \
  --weights runs/phase-2/baseline/weights/best.pt \
  --onnx runs/phase-3/model/slot-pose.onnx \
  --samples 20
```

The comparison uses the same preprocessing, confidence threshold, NMS settings, and 20 deterministic validation images for both backends.

Default tolerances:

- Bounding-box coordinates: at most 3 pixels in the original image.
- Keypoint coordinates: at most 32 pixels in the original image.
- Confidence: at most 0.01.
- Selected class and detection presence must match.

The coordinate limits apply after Ultralytics scales the 640 x 640 model output back to each original phone image. Keypoints have a wider limit because weak or partly out-of-frame corners are more numerically sensitive than the detected box. The report retains exact per-image differences so this tolerance does not hide drift.

Pass criteria: command prints `Parity: PASS` with zero failed cases.

## Generated files

```text
runs/phase-3/
├── model/
│   ├── slot-pose.onnx
│   └── slot-pose.metadata.json
└── parity/
    └── parity-report.json
```

These generated files are ignored by Git. Keep the ONNX artifact in controlled model storage when it is ready for deployment; do not add it to normal Git history.

## Limitations

- This parity test uses Ultralytics preprocessing and post-processing around both models. Browser-specific tensor preparation and JavaScript decoding are tested later against the same ONNX outputs.
- Parity proves conversion fidelity, not real-world model accuracy.
- The images come from the existing validation collection rather than the OPPO Reno Z office-camera test.

## Exit gate

- [x] All automated tests pass (9 tests).
- [x] ONNX export completes (11,004,971 bytes).
- [x] ONNX checker accepts the graph.
- [x] Input/output shape and opset contract pass.
- [x] PyTorch/ONNX parity passes on all 20 fixed samples.
- [x] Model metadata and checksum are recorded (`e11fcb12bae050ae316a51f97bdc2737feff79511473c4590a5dd3b400a35d49`).

Observed maximum PyTorch/ONNX differences were 2.482 pixels for boxes, 27.581 pixels for keypoints, and 0.003052 confidence. An unsimplified control export produced the same maxima, confirming that ONNX graph simplification was not the source of the difference.

After this gate, Phase 4 builds the mobile camera interface with mocked detections. Static browser inference follows before live ML camera detection.
