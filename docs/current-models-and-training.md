# Models, datasets and training

All browser model inputs are float32 `[1,3,320,320]`, RGB CHW scaled by 1/255. Each `.onnx` has adjacent metadata with its SHA-256. The metadata records the model version, preprocessing and original training run.

| Function | Browser model | Training run | Prepared dataset |
| --- | --- | --- | --- |
| Slot box, four corners, side class | `slot-pose.onnx` | `runs/angle-pose/baseline-v3-left-angle` | `slot-angle-pose-combined-v3-left-angle-80-20` plus `slot-angle-pose-v2-right-angle-80-20` |
| High-angle rejection | `slot-high-angle.onnx` | `runs/high-angle-classifier/v7-six` | `high-angle-classifier-v7-six` |
| Overall viewing acceptability | `slot-acceptability.onnx` | `runs/case-classifier/reviewed-acceptability-v3-addon` | `reviewed-acceptability-v3-addon` |

The high-angle model is **v7**, not the later v8 jitter experiment. Acceptability metadata contains an old “pending deployment” status string; the snapshot's runtime imports use the v3 Addon model. Treat file hashes and runtime URLs as authoritative for this release.

## Input and label contracts

- Pose uses full-frame letterboxing with RGB padding 114. Output `[1,19,2100]`: four box channels, three class scores, then four `(x,y,confidence)` corner triplets. Class order: centered, camera_left, camera_right. Corner order: top-left, top-right, bottom-right, bottom-left. Browser code applies decoding and NMS.
- Pose text labels use normalized YOLO pose format: class, center-x, center-y, width, height, then four x/y/visibility triplets. Preserve corner order. Do not flip side-class labels without correcting their meaning.
- High angle uses the **detected slot ROI**, not the whole frame: unrectified 9:16 crop centered on the box; crop height is `max(boxHeight*1.25, boxWidth*3*16/9)`. It occupies the central 180-pixel strip of a 320-square canvas, with gray padding. See `web/lib/high-angle-input.ts` for exact browser sampling. Output order: acceptable, high_angle. Warning threshold: 0.40.
- Acceptability uses full-frame RGB letterboxing to 320 with padding 114. Output order: error, success. Success threshold: 0.75.
- Both classifier training sets contain already-prepared PNG tensors. Avoid default classification center crops or geometric augmentation that changes the intended perspective.

## Included data

The training archive includes:

- `dataset/Categorize/slot-angle-pose-v3-full-984`: canonical categorized original images/labels, including Addon and new photos retained there.
- Exact prepared datasets named in the mapping above, preserving split assignments and repeated training samples.
- `dataset/slot-angle-pose-v4-reviewed-232-80-20`: additional reviewed work retained for future work; **not claimed to be used by the delivered pose model**.
- Current run weights, results, args and summaries; initialization checkpoints from pose v2, high-angle v6, and acceptability v2.
- Small historical manifests/metrics/configs for provenance, and available video-review stills. Older experiment image trees are not all included.

The high-angle manifest records 247 acceptable and 34 high-angle unique sources. Six September 23 additions (five high-angle and `IMG20260923145827.jpg` acceptable) were all used in training. Their 18/18 framing result is a training-fit check, not independent test accuracy. Repeated images and crop variants are correlated; preserve their source grouping when building future splits.

## Reproduce the pose recipe

After extracting the training archive into the repository root, preview and update its historical paths:

```sh
python ml/scripts/relocate_training_paths.py
python ml/scripts/relocate_training_paths.py --apply
```

This updates JSON/YAML files under dataset, runs and artifacts, keeping `.original-paths` backups. It does not change images or weights. Then, from the repository root:

```sh
python ml/scripts/train_angle_pose.py \
  --dataset dataset/slot-angle-pose-combined-v3-left-angle-80-20 \
  --model runs/angle-pose/baseline-v2-right-angle/weights/best.pt \
  --epochs 15 --imgsz 320 --batch 16 --device cpu \
  --workers 2 --patience 5 --seed 17 --name local-pose-candidate
```

Use a new run name every time: the historical script permits existing output directories. The original device was MPS; use `--device mps` on a suitable Mac or the supported CUDA setting on your machine. Numerical results may differ across platforms and library versions.

Export a candidate, keeping production untouched:

```sh
python ml/scripts/export_onnx.py \
  --weights runs/angle-pose/local-pose-candidate/weights/best.pt \
  --output runs/local-exports/slot-pose.onnx --imgsz 320 --opset 12
```

## Train classifiers

The prepared-data helper preserves each delivered experiment's preprocessing and core training parameters while requiring a new output name:

```sh
python ml/scripts/train_prepared_classifier.py high-angle --name height-candidate --dry-run
python ml/scripts/train_prepared_classifier.py high-angle --name height-candidate --device cpu
python ml/scripts/train_prepared_classifier.py acceptability --name acceptance-candidate --device cpu
```

Default initialization matches the historical runs (v6 ROI and acceptability v2). Add `--from-current` to fine-tune the delivered v7/v3 weights instead. It trains for 25 epochs by default and exports a separate opset-17 ONNX candidate. No production model is replaced. The helper supports a dry run without ML dependencies. A full retraining run is not part of local web setup.

Original scripts are retained under `ml/scripts/`: `high_angle_six_addon.py`, `high_angle_roi_trial.py`, `acceptability_addon.py`, and their preparation/audit helpers. They explain how the prepared images were made, but some expect old audit directories and cannot be rerun safely over the existing outputs without adaptation.

## Before promoting any candidate

Evaluate acceptable and rejected examples separately, including moderately elevated views, side angles, close views and live-video crops. Keep source sessions together when splitting; do not tune a threshold on the final test set. Compare new and old models at fixed thresholds and inspect failures visually. Check PyTorch/ONNX probability parity and browser preprocessing parity. Then test the entire app's guidance and capture gates on-device. Preserve old weights, metadata, hashes and a copy of the previous working application before replacing files and cache-version URLs.

## Guidance rules and code locations

These settings are separate from model weights. Check source before changing a
value; some guidance and capture checks have different tolerances or hysteresis.

| Check | Current value / location |
| --- | --- |
| Detection and corner confidence | 35% each, `web/lib/pose.ts` |
| Positive vertical perspective limit | +16%, `DEFAULT_ALIGNMENT_REFERENCE` in `pose.ts` |
| Negative vertical perspective limit | −28%, same reference |
| Horizontal centre band | 40–60% of image width, same reference |
| Slot frame area | 3–20%, same reference |
| Roll | ±7°, same reference |
| Capture side perspective | ±12%, `pose.ts` |
| High-angle classifier | 40% threshold, `web/lib/high-angle.ts` |
| Acceptability | 75% success, `web/lib/acceptability.ts` |
| Capture hold | 800 ms of qualifying evidence, plus reading/freshness checks |

Perspective percentages are geometric ratios, **not phone angles in degrees**.
Lowering a threshold does not reliably fix wrong corner measurements or a
classifier assigning the wrong class.

- `web/app/page.tsx` joins camera frames, inference, guidance and capture.
- `web/lib/measurement-stability.ts` filters measurements.
- `web/lib/action-guidance.ts` retains supported corrective instructions.
- `web/lib/live-guidance.ts` handles instruction and capture timing.
- The `web/tests/` suite checks these behaviors; run `npm test` from `web`.

## Known limitations and validation

Moderately elevated views have sometimes been accepted with a very low
high-angle score. Side views and noisy corners can also produce misleading
height guidance. The v7 classifier was selected for live testing despite known
boundary failures; its six new training photos do not establish independent
accuracy. Acceptability is a separate score, not a guarantee that all geometry
checks are correct.

For any candidate, test known acceptable views and known rejected high/side
views from independent capture sessions. Include near/far framing, slow motion,
sudden motion and brief detection loss. Inspect instruction persistence and
capture timing on the actual phone, not only static image scores. Keep the
existing dataset splits intact until you deliberately design a new evaluation.

To replace a model, evaluate and export to a candidate run first, check the
class order/preprocessing against browser code, then update the `.onnx`, its
metadata and the corresponding cache-version URL together. Save the old working
commit and rebuild/restart for a local rollback.
