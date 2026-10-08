# Phase 1 - Dataset Validation and Testing

## Purpose

Phase 1 proves that the supplied YOLO pose dataset is structurally safe to use for baseline training. It does not train a model and does not change dataset images or labels.

The selected input is `dataset/`, containing an 80/20 train/validation split. Dataset folders remain excluded from Git by `.gitignore`.

## What is validated

- Every image can be decoded.
- Every label has a matching image.
- Missing labels are counted as intentional background negatives.
- Each labeled object contains 17 YOLO pose values.
- The class ID is `0` (`slot`).
- Bounding boxes and visible keypoints use normalized coordinates.
- Each annotation contains four keypoints ordered TL, TR, BR, BL.
- Keypoint polygons do not self-intersect.
- Training and validation filenames do not overlap.
- Byte-identical images do not leak between training and validation.
- Duplicate images within a split are reported.

## One-time setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r ml/requirements.txt
```

## Test 1 - Validator behavior tests

```bash
python3 -m unittest discover -s ml/tests -v
```

Expected result:

```text
Ran 4 tests
OK
```

These tests prove that valid data passes, invalid keypoints fail, cross-split duplicates fail, and unlabeled images are treated as background negatives.

## Test 2 - Validate the real dataset

```bash
python3 ml/scripts/validate_dataset.py --dataset dataset
```

Expected result for the current dataset:

```json
{
  "status": "PASS",
  "error_count": 0,
  "warning_count": 0,
  "duplicate_group_count": 0
}
```

Generated reports:

- `reports/phase-1/dataset-validation.md`
- `reports/phase-1/dataset-validation.json`

The `reports/` directory is ignored by Git because these files can be regenerated.

If the command returns a non-zero exit code, review the report before training. Correct only a working copy of the affected label; retain the original export as a reference.

## Test 3 - Visual label inspection

```bash
python3 ml/scripts/render_label_montage.py --dataset dataset
```

Generated files:

- `artifacts/phase-1/train-label-montage.jpg`
- `artifacts/phase-1/val-label-montage.jpg`

Open both montages and confirm:

- Cyan `TL` is the top-left corner.
- Green `TR` is the top-right corner.
- Yellow `BR` is the bottom-right corner.
- Pink `BL` is the bottom-left corner.
- The white polygon follows the slot opening.
- Background-negative samples are intentionally unlabeled.
- Boxes do not target unrelated objects.

The renderer uses a fixed random seed, so rerunning it selects the same samples unless the dataset changes. Use `--seed 42` to inspect a different deterministic sample.

## Test 4 - Confirm Git excludes dataset and generated output

```bash
git status --short
git check-ignore -v dataset/images/train/* | head
git check-ignore -v reports/phase-1/dataset-validation.json
git check-ignore -v artifacts/phase-1/train-label-montage.jpg
```

The dataset images, reports, and montages should be ignored. The scripts, tests, requirements file, and this document should appear as files that can be committed.

## Current validation result

| Split | Images | Labels | Background negatives | Labeled objects |
| --- | ---: | ---: | ---: | ---: |
| Train | 480 | 467 | 13 | 467 |
| Validation | 116 | 116 | 0 | 116 |

Phase 1 passes when automated tests pass, the real dataset reports zero hard errors, and both montages pass visual inspection.

## Phase 1 exit checklist

- [x] Dataset format documented.
- [x] Automated validator implemented.
- [x] Validator behavior tests implemented.
- [x] Dataset contains no detected hard errors.
- [x] No detected train/validation duplication.
- [x] Background negatives accounted for.
- [x] Training and validation montages inspected.
- [x] Testing procedure documented.

The dataset is ready for Phase 2 baseline training.
