# Acceptability Addon update — 2026-09-17

The high-angle update learned the 18 Addon acceptable photos, but the separate deployed acceptability model still rejected 14 of them at its 75% threshold. A repeated side class could also veto capture despite frontal corner geometry.

## Candidate

- Fine-tune reviewed acceptability v2 using all previous training examples and 12 Addon photos. Keep all 752 rejected examples in their original train/validation/test splits.
- Preserve the old 228 acceptable-image splits. Addon uses the same chronological 12/3/3 assignment as the high-angle model.
- Keep full-frame RGB letterboxing at 320 pixels, fill114, divide255. Apply color jitter only, without changing perspective.
- Train at a small learning rate, select the best validation checkpoint, and evaluate the fixed75% operating threshold against the previous ONNX model.
- Addon holdouts come from the same session; they are development checks, not independent evidence of unseen-machine or live-camera accuracy.

## Guidance changes

- A repeated side label needs more than8% supporting side perspective to veto capture. The absolute12% geometry limit remains.
- Side-vote correction uses measured perspective for direction, avoiding a contradictory class label.
- Weak corners no longer trigger uncorroborated side instructions. Box centering/distance guidance remains available and capture still waits for reliable corners.
- Acceptability side corrections use the same8% corroboration threshold instead of4%.
- Keep300ms continuous alignment, fresh acceptability checks, and the current high-angle classifier unchanged.

## Evidence and rollback

Training and evaluation: `runs/case-classifier/reviewed-acceptability-v3-addon`.

Combined gate replay: `artifacts/acceptability-v3-addon/pipeline-results.json`. This repeats static images through the actual application functions; it does not reproduce camera motion, browser interpolation, or mobile timing.

Rollback: redeploy saved Site version97. Its exact version ID and commit, the previous acceptability model, metadata, and all model hashes are preserved in `artifacts/acceptability-v3-addon/rollback`.

## Results

The exported acceptability model passes all18 Addon images, including all6 holdouts. Original test acceptable passes improve30/34 to31/34, with1/113 rejected images passing for both models (different images).

Using the same corrected guidance functions, combined static capture improves2/18 to13/18 Addon images. Remaining failures:2 missing pose detections and3 vertical perspective vetoes. Original combined test captures30/34 acceptable and1/113 rejected (previous model under corrected rules:29/34 and0/113). High-angle combined checks block28/29, including5/5 test images; one known validation high-angle image still passes. These limitations are recorded rather than tuning on the test set.
