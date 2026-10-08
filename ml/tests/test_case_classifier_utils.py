from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from PIL import Image

from ml.scripts.case_classifier_utils import (
    CaseImage,
    expanded_crop_box,
    grouped_split,
    letterbox_square,
    source_id_from_filename,
    split_summary,
)
from ml.scripts.evaluate_case_classifier import threshold_metrics
from ml.scripts.validate_case_classifier import validate


def fake_uuid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012d}"


class CaseClassifierUtilityTests(unittest.TestCase):
    def test_source_id_is_uuid_prefix(self) -> None:
        source_id = "3c66e475-04e2-4199-88bc-1b5fa080a7fc"
        path = Path(f"{source_id}-2026-08-16T08-07-27.241Z.jpg")
        self.assertEqual(source_id_from_filename(path), source_id)

    def test_grouped_split_is_reproducible_balanced_and_has_no_leakage(self) -> None:
        records = []
        for index in range(200):
            source_id = fake_uuid(index)
            label = "success" if index % 2 == 0 else "error"
            records.extend([
                CaseImage(Path(f"{source_id}-a.jpg"), label, source_id),
                CaseImage(Path(f"{source_id}-b.jpg"), label, source_id),
            ])
        first = grouped_split(records, seed=17)
        second = grouped_split(records, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 200)
        summary = split_summary(records, first)
        self.assertAlmostEqual(summary["train"]["percentage"], 70.0, delta=2.0)
        self.assertAlmostEqual(summary["val"]["percentage"], 15.0, delta=2.0)
        self.assertAlmostEqual(summary["test"]["percentage"], 15.0, delta=2.0)
        self.assertEqual({first[record.source_id] for record in records if record.source_id == fake_uuid(4)}, {first[fake_uuid(4)]})

    def test_crop_box_expands_and_clamps(self) -> None:
        self.assertEqual(expanded_crop_box((10, 20, 50, 80), 100, 100, 0.25), (0, 5, 60, 95))

    def test_letterbox_preserves_complete_tall_crop(self) -> None:
        source = Image.new("RGB", (20, 100), "red")
        prepared = letterbox_square(source, 200)
        self.assertEqual(prepared.size, (200, 200))
        # The complete 1:5 source becomes 40x200 with padding on both sides.
        self.assertEqual(prepared.getpixel((79, 0)), (114, 114, 114))
        self.assertEqual(prepared.getpixel((99, 0)), (255, 0, 0))
        self.assertEqual(prepared.getpixel((99, 199)), (255, 0, 0))

    def test_validator_detects_source_leakage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_id = fake_uuid(1)
            items = []
            for split, label in (("train", "success"), ("val", "error")):
                for class_name in ("success", "error"):
                    (root / split / class_name).mkdir(parents=True, exist_ok=True)
                relative = Path(split) / label / f"{source_id}-{split}.jpg"
                Image.new("RGB", (20, 30), "white").save(root / relative)
                items.append({
                    "source_path": f"/source/{split}.jpg",
                    "filename": relative.name,
                    "source_id": source_id,
                    "label": label,
                    "split": split,
                    "status": "prepared",
                    "prepared_path": relative.as_posix(),
                })
            for split in ("test",):
                for class_name in ("success", "error"):
                    (root / split / class_name).mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(
                json.dumps({"classifier_input_size": 20, "images": items}),
                encoding="utf-8",
            )
            report = validate(root)
            self.assertEqual(report["summary"]["status"], "FAIL")
            self.assertTrue(any("source leakage" in error for error in report["errors"]))

    def test_threshold_metrics_distinguish_false_accepts(self) -> None:
        rows = [
            {"actual": "success", "success_probability": 0.9},
            {"actual": "success", "success_probability": 0.6},
            {"actual": "error", "success_probability": 0.85},
            {"actual": "error", "success_probability": 0.2},
        ]
        metrics = threshold_metrics(rows, 0.8)
        self.assertEqual(metrics["confusion"]["error_false_accepted"], 1)
        self.assertEqual(metrics["false_accept_rate"], 0.5)
        self.assertEqual(metrics["success_acceptance_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
