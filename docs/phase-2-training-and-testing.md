# Phase 2 - Baseline Pose Training and Testing

## Purpose

Phase 2 fine-tunes a pretrained YOLO11n-pose model to detect the cup-return slot and predict four corners: TL, TR, BR, and BL.

Training runs on the development computer, not the OPPO Reno Z. The project Mac has an Apple M1, so `--device auto` selects MPS when available and otherwise falls back to CPU.

The scripts generate `runs/phase-2/configs/resolved-data.yaml` with the current dataset path instead of relying on the obsolete absolute path in the supplied YAML.

## Generated outputs

Generated model files and reports are written under ignored `runs/phase-2/`:

```text
runs/phase-2/
├── configs/resolved-data.yaml
├── smoke/
├── baseline/weights/best.pt
└── baseline-evaluation/
    ├── metrics.json
    └── predictions/
```

## Setup

```bash
cd /Users/bensontan/Desktop/image_detection
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ml/requirements.txt
```

Confirm MPS availability:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

Expected on the project Mac: `True`.

## Test 1 - Automated tests

```bash
python3 -m unittest discover -s ml/tests -v
```

This runs Phase 1 validator tests and Phase 2 configuration tests.

## Test 2 - Training smoke test

```bash
python3 ml/scripts/train.py --smoke
```

This downloads the pretrained checkpoint when necessary, trains one epoch at 320 pixels using 5% of the training set, and skips full validation. It proves that dependencies, dataset loading, pose shape, device execution, and output writing work together.

Pass criteria:

- Process exits successfully.
- Selected device is printed.
- `runs/phase-2/smoke/weights/last.pt` exists.
- No dataset or keypoint-shape exception occurs.

## Test 3 - Full baseline training

Recommended Apple M1 command:

```bash
python3 ml/scripts/train.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 4 \
  --device mps \
  --name baseline
```

If MPS is unstable, use CPU with a smaller batch:

```bash
python3 ml/scripts/train.py --epochs 100 --imgsz 640 --batch 2 --device cpu --name baseline-cpu
```

Keep the Mac connected to power and prevent sleep. Full training passes when `runs/phase-2/baseline/weights/best.pt` exists.

## Test 4 - Evaluate the checkpoint

```bash
python3 ml/scripts/evaluate.py \
  --weights runs/phase-2/baseline/weights/best.pt \
  --device mps
```

Review:

- `runs/phase-2/baseline-evaluation/metrics.json`
- Plots under `runs/phase-2/baseline-evaluation/`
- Images under `runs/phase-2/baseline-evaluation/predictions/`

Confirm that predictions target the slot and place TL, TR, BR, and BL on the intended corners. Pay special attention to distant, tilted, dark, reflective, and cropped slots.

Record pose mAP50, pose mAP50-95, box mAP50, precision, recall, background false positives, and visually usable corner predictions. Metrics alone do not complete Phase 2; the corners must support alignment geometry.

## Exit gate

- [x] Automated tests pass.
- [x] Smoke training completes.
- [x] POC baseline training completes.
- [x] `best.pt` is created.
- [x] Evaluation completes and metrics are recorded.
- [x] Prediction samples are visually inspected.
- [x] Failures and limitations are documented.

Phase 3 begins by exporting the accepted `best.pt` model to ONNX and checking prediction parity.

## Completed POC baseline - 17 August 2026

The first accepted baseline used:

| Setting | Value |
| --- | --- |
| Model | `yolo11n-pose.pt` |
| Epochs | 10 |
| Image size | 640 |
| Batch size | 8 |
| Device | Apple M1 `mps` |
| Seed | 17 |
| Training duration | Approximately 9 minutes |
| Best checkpoint size | Approximately 5.3 MB |

Independent evaluation of `runs/phase-2/baseline/weights/best.pt` produced:

| Metric | Result |
| --- | ---: |
| Box precision | 0.9989 |
| Box recall | 1.0000 |
| Box mAP50 | 0.9950 |
| Box mAP50-95 | 0.6696 |
| Pose precision | 0.9989 |
| Pose recall | 1.0000 |
| Pose mAP50 | 0.9950 |
| Pose mAP50-95 | 0.9859 |
| M1 model inference | 9.4 ms/image |

Six prediction samples were visually inspected across distant, close, bright, dark, and partially framed cases. The model selected the intended slot and placed the four ordered keypoints at the expected top and bottom corners. Confidence in the reviewed samples ranged roughly from 0.70 to 0.85.

### Limitations

- Validation images come from the same prepared dataset collection as training, so these metrics are optimistic and are not proof of office-phone performance.
- The validation split has no background-negative images, so its precision metric does not fully measure false positives in unrelated scenes.
- MPS emitted warnings that two operations do not have deterministic implementations; the seed is recorded, but exact bit-for-bit training reproduction is not guaranteed on MPS.
- The 10-epoch checkpoint is accepted as the POC baseline because its held-out metrics and visual samples are strong. A longer 50-100 epoch run is optional and should be justified by new, independently collected test images rather than the current validation score alone.
- Real-device testing must occur after ONNX export and browser integration.
