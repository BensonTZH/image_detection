# High-angle v3 Addon candidate — 2026-09-17

Imported 18 user-reviewed acceptable photos and their matching pose labels from
`dataset/Categorize/slot-angle-pose-v3-full-984/Addon`. There were 20 photos and
18 annotations. Removed the two photos without annotations, IMG20260917115626.jpg
and IMG20260917115629.jpg, from Addon; recovery copies and hashes are in
`artifacts/high-angle-v3-addon/removed-unmatched` and `import.json`.

The 18 matched pairs remain in Addon and have canonical copies under
`images/centered/acceptable` and `labels/centered/acceptable`. Canonical acceptable
image count is now 246. All 18 pose annotations passed shape, finite-value,
coordinate, and visibility checks. Folder membership supplies the binary
acceptable label; pose annotations do not retrain the pose detector.

## Training and split

Prepared `dataset/high-angle-classifier-v3-addon`, preserving every original v2
split assignment. New photos use chronological blocks: 12 training, 3 validation,
3 testing. Unique totals: training 172 acceptable + 19 high-angle; validation and
test each 37 acceptable + 5 high-angle. Training high-angle photos were replicated
153 times to balance classes. Hash checks prevent identical images crossing splits.

Trained from generic YOLO11n classification weights with the v2 full-frame 320px
RGB letterbox preprocessing, color jitter only, seed 17, batch16, and Apple MPS.
Training stopped early after 19 of a maximum30 epochs. Exported best checkpoint to
`runs/high-angle-classifier/v3-addon/weights/best.onnx`. CPU ONNX/PyTorch maximum
probability difference across eight parity samples was 4.43e-9.

## Evaluation at the unchanged 75% threshold

| Group | Previous | Candidate |
| --- | --- | --- |
| Addon training acceptable | 0/12 pass | 12/12 pass |
| Addon validation acceptable | 0/3 pass | 3/3 pass |
| Addon test acceptable | 0/3 pass | 3/3 pass |
| Original acceptable, all splits | 228/228 pass | 228/228 pass |
| High-angle training | 19/19 detected | 19/19 detected |
| High-angle validation | 5/5 detected | 4/5 detected |
| High-angle test | 5/5 detected | 5/5 detected |

Full predictions are in `runs/high-angle-classifier/v3-addon/comparison.json`.
The photo gallery is `artifacts/high-angle-v3-addon/comparison.html`.

The six held-out Addon photos are from the same short session as its training
photos. High-angle validation/test blocks also share one capture session. These
are correlated development checks, not independent live-camera accuracy estimates.
The one missed high-angle validation image is a regression to inspect before
deployment. The 75% threshold was fixed before evaluation, not tuned on test data.

## Status and reproduction

Candidate only: no deployment and no changes to the production ONNX files,
runtime thresholds, pose detector, or 300ms alignment period. Production file
hashes were verified unchanged after evaluation.

Use `/private/tmp/cup-high-angle-env/bin/python ml/scripts/high_angle_addon.py`
with `prepare`, `train --epochs 30`, or `evaluate`. Preparation and training refuse
to overwrite the existing import or model; use new versioned paths for another run.

## User-authorized deployment

Deployed successfully on 2026-09-17 as CupDetector version97, source commit
309b78f9178e24173ba82c5cde727a29cb904646. Threshold remains75%, hold300ms.
Previous version96 / commit fe789cbe40945930ee7e20bcf06800c2e24980a9 is the
rollback target. Model, metadata, and loader backups are in
artifacts/high-angle-v3-addon/deployment-rollback. All45 behavior tests and
2 rendering checks passed before publication.
