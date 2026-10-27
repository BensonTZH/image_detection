"""Train a separate candidate from the delivered prepared classifier dataset."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "high-angle": {
        "dataset": "high-angle-classifier-v7-six",
        "initial": "runs/high-angle-classifier/v6-roi/weights/best.pt",
        "current": "runs/high-angle-classifier/v7-six/weights/best.pt",
        "classes": ["acceptable", "high_angle"], "batch": 32, "seed": 23,
        "patience": 7, "freeze": 5, "jitter": [.2, .2, .15, .01],
    },
    "acceptability": {
        "dataset": "reviewed-acceptability-v3-addon",
        "initial": "runs/case-classifier/reviewed-acceptability-v2/weights/best.pt",
        "current": "runs/case-classifier/reviewed-acceptability-v3-addon/weights/best.pt",
        "classes": ["error", "success"], "batch": 16, "seed": 17,
        "patience": 8, "freeze": None, "jitter": [.15, .15, .15, .01],
    },
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=CONFIG)
    p.add_argument("--name", required=True, help="New directory name under runs/classifier-candidates")
    p.add_argument("--device", default="cpu", help="cpu, mps, or CUDA device index")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--from-current", action="store_true", help="Fine-tune delivered weights instead of historical initialization")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in (".", "..") or args.epochs < 1:
        p.error("Use a simple new run name and positive epochs")
    c = CONFIG[args.task]
    data = ROOT / "dataset" / c["dataset"]
    weights = ROOT / c["current" if args.from_current else "initial"]
    run = ROOT / "runs/classifier-candidates" / args.name
    if run.exists():
        p.error(f"Refusing to overwrite {run}")
    plan = {"task": args.task, "dataset": str(data), "weights": str(weights),
            "run": str(run), "epochs": args.epochs, "device": args.device,
            "classes": c["classes"], "historical_settings": c}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if not weights.is_file() or any(not (data / split).is_dir() for split in ("train", "val", "test")):
        p.error("Extract the training archive into the repository root first")
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from torchvision import transforms as T
    class PreparedImageTrainer(ClassificationTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            d = super().build_dataset(img_path, mode, batch)
            ops = [T.ColorJitter(*c["jitter"])] if mode == "train" else []
            d.torch_transforms = T.Compose(ops + [T.ToTensor()])
            return d
    run.mkdir(parents=True)
    (run / "training-plan.json").write_text(json.dumps(plan, indent=2))
    YOLO(str(weights)).train(
        trainer=PreparedImageTrainer, data=str(data), project=str(run.parent), name=run.name,
        exist_ok=True, epochs=args.epochs, imgsz=320, batch=c["batch"], device=args.device,
        workers=0, patience=c["patience"], optimizer="AdamW", lr0=.0003, lrf=.1,
        warmup_epochs=1, seed=c["seed"], deterministic=True, freeze=c["freeze"],
        cache=False, plots=args.task == "acceptability", auto_augment=None, erasing=0,
        fliplr=0, flipud=0, scale=0,
    )
    model = YOLO(str(run / "weights/best.pt"))
    if list(model.names.values()) != c["classes"]:
        raise RuntimeError("Export stopped: unexpected class ordering")
    model.export(format="onnx", imgsz=320, opset=17, simplify=False, dynamic=False, device="cpu")
    print(f"Candidate saved at {run}; production files have not been replaced.")

if __name__ == "__main__":
    main()
