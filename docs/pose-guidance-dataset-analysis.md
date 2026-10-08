# Analyze the success/error photographs for pose guidance

This analysis uses the existing slot pose detector on every original image in
`dataset/case/success2` and `dataset/case/error2`. It does not crop the images,
train a success/error classifier, change CupDetector, or modify the source
photographs.

The purpose is to answer three separate questions:

1. Does the current model find the slot?
2. Are all four predicted corners reliable enough for angle guidance?
3. Which measured pose values differ between readable (`success`) and badly
   positioned (`error`) photographs?

## Run the analysis

From the repository root, activate the Python environment and run:

```bash
source .venv/bin/activate
python ml/scripts/analyze_case_pose_guidance.py
```

The defaults deliberately reuse the same UUID-grouped 70/15/15 split and seed
as the earlier case experiment. Related frames with the same UUID stay in one
split, which prevents near-duplicate photographs from leaking into validation
or test.

The existing pose checkpoint is expected at:

```text
runs/phase-2/baseline/weights/best.pt
```

You can explicitly select a device if automatic selection is unsuitable:

```bash
python ml/scripts/analyze_case_pose_guidance.py --device mps
python ml/scripts/analyze_case_pose_guidance.py --device cpu
```

## Review the outputs

Generated files are written to `reports/pose-guidance/` and remain ignored by
Git:

- `case-pose-metrics.csv`: one row per original image, including the split,
  label, detector confidence, slot box, four corner points and confidences,
  centre X/Y, frame area, rotation, side perspective, vertical perspective,
  and corner reliability.
- `case-pose-summary.json`: complete machine-readable statistics.
- `case-pose-summary.md`: shorter tables intended for manual review.

Open the Markdown summary on macOS with:

```bash
open reports/pose-guidance/case-pose-summary.md
```

The report derives reference ranges only from reliable `train/success`
detections. It compares those ranges with validation images. It does not use
the test split to choose rules; keep test untouched until the final guidance
rules have been selected.

## Interpret the result

- Many `no_detection` rows, especially for successful photographs, mean the
  pose detector needs additional manually verified box-and-corner labels from
  this office setup.
- A detected box with unreliable corners means left/right perspective, vertical
  tilt, and rotation guidance should not be trusted for that frame.
- If success and error photographs strongly overlap on a pose value, changing
  that guidance threshold cannot reliably separate them by itself.
- If a metric retains most validation success images while excluding many
  validation error images, it is a useful candidate for a site-tuned guidance
  rule. Confirm the chosen rule once on the untouched test split before putting
  it into CupDetector.
