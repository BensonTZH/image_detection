# Success/error quality classifier

The files in `dataset/case/success2` and `dataset/case/error2` are image-level
quality labels:

- `success2`: the returned cups are positioned clearly enough for the next
  system to read them.
- `error2`: the cups are positioned badly or are otherwise unsuitable to read.

These images do not contain a slot box or four corner labels, so they must not
be added directly to the pose-detector training folders. The implementation
uses two stages:

1. The existing pose model detects the return slot and supplies its crop.
2. A binary classifier evaluates that crop as `success` or `error`.

The classifier is a capture gate. It can reject an unreadable view, but these
two labels alone cannot say whether the user should move left, right, up, or
down. Directional guidance still comes from the pose model and app rules.

## Why the split is grouped

The source UUID at the beginning of every filename identifies related frames.
All frames with the same UUID are kept in exactly one split, even when that UUID
appears in both class folders. This prevents very similar frames from leaking
from training into validation or testing.

The deterministic split uses seed `17`:

| Split | Original images | Success | Error | Purpose |
| --- | ---: | ---: | ---: | --- |
| Train | 1,400 (70%) | 700 | 700 | Learn model weights |
| Validation | 300 (15%) | 150 | 150 | Select the best checkpoint and stop training |
| Test | 300 (15%) | 150 | 150 | One final unbiased evaluation |

The crop preparation run produces 1,975 usable crops. The pose detector does
not find a slot in 25 source images (24 error and 1 success), so the usable
classifier inputs are 1,383 train, 296 validation, and 296 test. These misses
remain listed in `dataset/case-classifier/manifest.json`; they are not silently
converted to full-frame classifier inputs.

Every detected slot crop is proportionally resized and padded into a 320×320
square. This keeps the complete top and bottom of the tube visible. It avoids
the default classifier behaviour that would center-crop a tall slot image and
discard most of its vertical content.

## Prepare and validate

Run from the repository root with the Python environment activated:

```bash
python ml/scripts/prepare_case_classifier.py \
  --source dataset/case \
  --output dataset/case-classifier \
  --weights runs/phase-2/baseline/weights/best.pt \
  --ratios 0.70 0.15 0.15 \
  --classifier-size 320 \
  --seed 17

python ml/scripts/validate_case_classifier.py \
  --dataset dataset/case-classifier
```

Preparation never changes the source images. It refuses to replace an existing
prepared directory unless `--overwrite` is explicitly supplied. The generated
dataset is ignored by Git.

## Train

First run the short setup check:

```bash
python ml/scripts/train_case_classifier.py \
  --dataset dataset/case-classifier \
  --smoke
```

The smoke run uses only 10% of the training crops for one epoch. Its accuracy
is not a model-quality result.

Run the real training job on the MacBook Air with:

```bash
python ml/scripts/train_case_classifier.py \
  --dataset dataset/case-classifier \
  --epochs 30 \
  --imgsz 320 \
  --batch 8 \
  --device mps \
  --name letterbox-320
```

Training is substantially slower than browser inference. Keep the Mac awake
and connected to power. The best checkpoint is written to:

```text
runs/case-classifier/letterbox-320/weights/best.pt
```

If MPS is unavailable, replace `--device mps` with `--device cpu`. The default
`--device auto` selects CUDA, then MPS, then CPU.

## Evaluate the untouched test split

After full training completes:

```bash
python ml/scripts/evaluate_case_classifier.py \
  --dataset dataset/case-classifier \
  --weights runs/case-classifier/letterbox-320/weights/best.pt \
  --imgsz 320 \
  --device mps \
  --success-threshold 0.80 \
  --name letterbox-320-evaluation
```

Review these outputs in `runs/case-classifier/letterbox-320-evaluation`:

- `test-metrics.json`: acceptance/rejection rates and confusion counts.
- `test-predictions.csv`: probability for every test crop.
- `review-sheet.jpg`: difficult and incorrect examples for visual inspection.

The most important initial metric is `false_accept_rate`: the percentage of
known error images that would still be allowed through. Do not deploy based on
top-1 accuracy alone. The `0.80` threshold is a conservative starting point;
choose the final threshold using validation results and confirm it once on the
test set.

## Current integration status

CupDetector loads `slot-acceptability.onnx` as a separate capture gate. Pose
detection and directional guidance still come from `slot-pose.onnx`. The
acceptability model runs only after the pose, corner, perspective, and angle
checks pass, and automatic capture requires a success probability of at least
0.80 continuously through the existing capture hold.

Rollback is one setting in `web/lib/acceptability.ts`: set
`ENABLE_ACCEPTABILITY_CLASSIFIER` to `false` and rebuild. That bypasses the
classifier and restores pose-only capture without changing the pose model or
its alignment rules.
