#!/usr/bin/env python3
"""Prepare, train and test the experimental high-viewpoint classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from case_classifier_utils import CaseImage, grouped_split, letterbox_square, source_id_from_filename, write_json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataset/Categorize/slot-angle-pose-v3-full-984/images"
DATA = ROOT / "dataset/high-angle-classifier-v2"
RUN = ROOT / "runs/high-angle-classifier/v2"
LABELS = ("acceptable", "high_angle")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def prepare():
    """Relocate the approved shortlist, with a rollback manifest, then split it."""
    if DATA.exists():
        raise FileExistsError(DATA)
    accepted = SOURCE / "centered/acceptable"
    high = SOURCE / "high_angle/High"
    annotation_root = SOURCE / "high_angle/task_2596633_annotations_2026_09_16_09_25_58_ultralytics yolo pose 1.0/labels/train"
    selected = images(high)
    if len(selected) != 29:
        raise ValueError("Expected the 29 newly labelled high-angle photos")
    annotations = {}
    for image in selected:
        label = annotation_root / (image.stem + ".txt")
        values = [float(v) for v in label.read_text().split()]
        if len(values) != 17 or values[0] != 0 or not all(0 <= v <= 1 for v in values[1:5]):
            raise ValueError(f"Invalid pose label: {label}")
        if values[3] <= 0 or values[4] <= 0:
            raise ValueError(f"Empty bounding box: {label}")
        for i in range(5,17,3):
            if not (0 <= values[i] <= 1 and 0 <= values[i+1] <= 1 and values[i+2] in (0,1,2)):
                raise ValueError(f"Invalid corner: {label}")
        annotations[str(image)] = {"path": str(label), "sha256": digest(label), "values": values}
    records = [CaseImage(p, "success", source_id_from_filename(p)) for p in images(accepted)]
    assignment = grouped_split(records, seed=17)
    # Contiguous blocks prevent random interleaving; all positives remain one
    # capture session, so these are development checks, not independent tests.
    for i,p in enumerate(selected):
        group = "high-session-train" if i < 19 else "high-session-val" if i < 24 else "high-session-test"
        records.append(CaseImage(p, "error", group))
        assignment[group] = group.rsplit("-",1)[1]
    label_map = {"success": "acceptable", "error": "high_angle"}
    write_json(DATA / "annotation-audit.json", annotations)
    items = []
    hash_splits = {}
    for record in records:
        split = assignment[record.source_id]
        label = label_map[record.label]
        sha = digest(record.path)
        if sha in hash_splits and hash_splits[sha] != split:
            raise ValueError("Identical image crosses splits")
        hash_splits[sha] = split
        dest = DATA / split / label / f"{record.path.stem}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record.path) as im:
            letterbox_square(ImageOps.exif_transpose(im), 320).save(dest)
        items.append({"source": str(record.path), "source_id": record.source_id,
                      "sha256": sha, "split": split, "label": label,
                      "prepared": str(dest.relative_to(DATA)), "replica": False})
    # Oversampling only affects training; validation and test remain untouched.
    train = {label: [x for x in items if x["split"] == "train" and x["label"] == label] for label in LABELS}
    target = max(map(len, train.values()))
    replicas = []
    for label, originals in train.items():
        for i in range(target - len(originals)):
            item = originals[i % len(originals)]
            src = DATA / item["prepared"]
            dst = src.with_name(f"{src.stem}__replica_{i}.png")
            shutil.copy2(src, dst)
            replicas.append({**item, "prepared": str(dst.relative_to(DATA)), "replica": True})
    for split in ("train", "val", "test"):
        for label in LABELS:
            if not any(x["split"] == split and x["label"] == label for x in items):
                raise ValueError(f"Empty split/class: {split}/{label}")
    summary = {split: dict(Counter(x["label"] for x in items if x["split"] == split)) for split in ("train", "val", "test")}
    write_json(DATA / "manifest.json", {"classes": LABELS, "input_size": 320,
        "preprocessing": "full-frame RGB letterbox, fill 114, divide by 255",
        "split_seed": 17, "split_group": "acceptable UUID; high-angle contiguous session blocks (not independent sessions)", "original_counts": summary,
        "training_replicas": len(replicas), "images": items + replicas})
    print(json.dumps(summary, indent=2), flush=True)


def train(epochs):
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from torchvision import transforms as T
    from training_utils import choose_device

    class ViewpointTrainer(ClassificationTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            # Perspective is the target. Never crop, stretch, flip or rotate it.
            ops = [T.ColorJitter(brightness=.15, contrast=.15, saturation=.15, hue=.01)] if mode == "train" else []
            dataset.torch_transforms = T.Compose(ops + [T.ToTensor()])
            return dataset

    if (RUN / "weights/best.pt").exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    device = choose_device("auto")
    write_json(RUN / "experiment.json", {"dataset": str(DATA), "model": "yolo11n-cls.pt",
        "epochs": epochs, "device": device, "classes": LABELS, "seed": 17,
        "augmentation": "color jitter only; full square preserved", "production_models_modified": False})
    YOLO(str(ROOT / "yolo11n-cls.pt")).train(trainer=ViewpointTrainer,
        data=str(DATA), project=str(RUN.parent), name=RUN.name, exist_ok=True,
        epochs=epochs, imgsz=320, batch=16, device=device, workers=0, patience=10,
        seed=17, deterministic=True, pretrained=True, cache=False, plots=True,
        auto_augment=None, erasing=0, fliplr=0, flipud=0, scale=0)


def metrics(rows, threshold):
    tp = sum(r["label"] == "high_angle" and r["probability"] >= threshold for r in rows)
    fn = sum(r["label"] == "high_angle" and r["probability"] < threshold for r in rows)
    fp = sum(r["label"] == "acceptable" and r["probability"] >= threshold for r in rows)
    tn = len(rows) - tp - fn - fp
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {"threshold": threshold, "high_detected": tp, "high_missed": fn,
        "acceptable_blocked": fp, "acceptable_passed": tn,
        "high_recall": recall, "acceptable_pass_rate": specificity,
        "balanced_accuracy": (recall + specificity) / 2, "accuracy": (tp + tn) / len(rows)}


def evaluate():
    from ultralytics import YOLO
    import numpy as np
    import onnxruntime as ort
    model = YOLO(str(RUN / "weights/best.pt"))
    high_index = next(i for i, name in model.names.items() if name == "high_angle")
    manifest = json.loads((DATA / "manifest.json").read_text())
    predictions = {}
    for split in ("val", "test"):
        rows = []
        items = [x for x in manifest["images"] if x["split"] == split]
        for item in items:
            # Evaluate original uploads through exactly the test-page pipeline.
            with Image.open(item["source"]) as original:
                prepared = letterbox_square(ImageOps.exif_transpose(original), 320)
            result = model.predict(prepared, imgsz=320, device="cpu", verbose=False)[0]
            rows.append({**item, "probability": float(result.probs.data[high_index])})
        predictions[split] = rows
    # Choose operating threshold using validation only, then report test once.
    candidates = [metrics(predictions["val"], t / 100) for t in range(20, 91, 5)]
    eligible = [m for m in candidates if m["acceptable_pass_rate"] >= .90]
    chosen = max(eligible or candidates, key=lambda m: (m["balanced_accuracy"], m["acceptable_pass_rate"], -abs(m["threshold"] - .5)))
    report = {"validation": chosen, "test": metrics(predictions["test"], chosen["threshold"]),
        "validation_thresholds": candidates, "predictions": predictions,
        "limitations": "Only 29 positives from one machine/session. High-angle validation and test blocks are correlated development samples, not unseen-session or unseen-machine estimates. Threshold selected on validation with acceptable pass rate >=90% when achievable."}
    write_json(RUN / "evaluation.json", report)
    exported = Path(model.export(format="onnx", imgsz=320, opset=17, simplify=False, dynamic=False, device="cpu"))
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    errors = []
    for row in predictions["test"]:
        with Image.open(row["source"]) as original:
            prepared = letterbox_square(ImageOps.exif_transpose(original), 320)
        arr = np.asarray(prepared, dtype=np.float32) / 255
        out = session.run(None, {session.get_inputs()[0].name: arr.transpose(2, 0, 1)[None]})[0]
        errors.append(abs(float(out[0, high_index]) - row["probability"]))
    if max(errors) > .001:
        raise ValueError(f"ONNX parity failed: {max(errors)}")
    write_json(RUN / "model.metadata.json", {"model": str(exported), "sha256": digest(exported),
        "classes": model.names, "high_angle_threshold": chosen["threshold"], "input_size": 320,
        "preprocessing": manifest["preprocessing"], "onnx_max_probability_difference": max(errors),
        "test": report["test"], "status": "candidate for user-authorized pilot deployment"})
    print(json.dumps({"validation": chosen, "test": report["test"], "onnx_max_difference": max(errors)}, indent=2), flush=True)


def predict(path):
    import numpy as np
    import onnxruntime as ort
    metadata = json.loads((RUN / "model.metadata.json").read_text())
    session = ort.InferenceSession(metadata["model"], providers=["CPUExecutionProvider"])
    high_index = next(int(i) for i, name in metadata["classes"].items() if name == "high_angle")
    for image_path in images(path) if path.is_dir() else [path]:
        with Image.open(image_path) as im:
            prepared = letterbox_square(ImageOps.exif_transpose(im), 320)
            tensor = np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1)[None] / 255
        probability = float(session.run(None, {session.get_inputs()[0].name: tensor})[0][0, high_index])
        print(json.dumps({"image": str(image_path), "high_angle_probability": probability,
            "guidance": "Lower your phone" if probability >= metadata["high_angle_threshold"] else "No high-angle warning"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "predict"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "train": train(args.epochs)
    elif args.command == "evaluate": evaluate()
    elif args.image: predict(args.image)
    else: parser.error("predict requires --image PATH (image or folder)")
