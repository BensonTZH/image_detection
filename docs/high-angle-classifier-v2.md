# High-angle classifier v2 pilot — 2026-09-16

User-authorized training and deployment from the new high_angle folder.
Sources were read without moving/relabeling originals: 29 high_angle/High JPEGs,
29 matching CVAT YOLO pose annotations, and 228 centered/acceptable JPEGs.
The folder supplies the binary image label. All 29 pose files were validated
and their hashes recorded; corner annotations do not retrain the existing pose model.

Training: generic YOLO11n classification weights, full-frame RGB proportional
320px letterbox (fill114), lossless PNG preparation, /255, color jitter only.
160 acceptable +19 high unique training photos, high oversampled to160.
Validation34 acceptable+5 high; test34 acceptable+5 high. Acceptable UUIDs
stay in one split. High photos are contiguous blocks from the SAME session.
This does NOT measure unseen-machine or unseen-session generalization.
A classifier could still learn machine/background differences; collect paired
acceptable-height photos of this machine and separate sessions for further validation.

Early stopped after11 epochs, best epoch1. Validation-selected threshold0.50.
Validation:5/5 high caught,34/34 acceptable passed. Test:5/5 high caught,
34/34 acceptable passed. ONNX/PyTorch maximum probability difference1.2e-7.
These small correlated development results must not be advertised as general100% accuracy.

Files: dataset/high-angle-classifier-v2/{manifest,annotation-audit}.json;
runs/high-angle-classifier/v2/{evaluation,model.metadata}.json and weights/best.*.
Reproduce using /private/tmp/cup-high-angle-env/bin/python
ml/scripts/high_angle_classifier.py {prepare,train,evaluate}; existing datasets
and checkpoints are protected against overwrite. Temp environment may need recreation.

Integration: separate third model, classes[acceptable,high_angle]. Runs on same
320px input as pose. One high score immediately blocks capture and resets its
hold; two consecutive scores >=.50 establish Lower your phone guidance.
Two scores <.40 clear an established warning. Existing guidance smoothing remains.
The hold remains300ms and starts only once all gates pass and movement guidance
has cleared. Height warning can override sideways guidance without requiring
geometric corroboration. Missing detection resets height history. Static image
testing reports the height warning immediately. Model load errors stop detection.

Verification:42 behavior tests +2 production render/asset tests passed.
Production browser upload: IMG20260916141218.jpg -> Lower your phone;
0515932f-09c0-4c33-a0e3-30f170600c1c-2026-08-02T05-16-03.479Z.jpg -> Hold still.
Development Vite preview had a WASM-loading failure; production preview
loads packaged WASM and all three models successfully. Phone latency remains
unmeasured; desktop browser total inference observed281–513ms.

Rollback: set ENABLE_HIGH_ANGLE_CLASSIFIER=false in web/lib/high-angle.ts,
rebuild and redeploy. This bypasses model loading, height guidance and capture
blocking. Full baseline web commit0cc41970eea6767e7ad417a6176cd1bcf5e8471b
(live version93 before this change). Baseline copies/hashes are in
artifacts/high-angle-v2-baseline-20260916. Old pose and acceptability model
hashes were verified unchanged. No alignment or geometry thresholds changed.
