from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from ml.scripts.validate_dataset import validate


VALID_ROW = "0 0.5 0.5 0.4 0.6 0.3 0.2 2 0.7 0.2 2 0.7 0.8 2 0.3 0.8 2\n"


class DatasetValidatorTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> None:
        for split, color in (("train", "white"), ("val", "black")):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (32, 32), color).save(root / "images" / split / f"{split}.jpg")
            (root / "labels" / split / f"{split}.txt").write_text(VALID_ROW, encoding="utf-8")

    def test_valid_dataset_passes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            self.assertEqual(validate(root)["summary"]["status"], "PASS")

    def test_invalid_keypoint_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            invalid = VALID_ROW.replace("0.3 0.2 2", "-0.1 0.2 2", 1)
            (root / "labels" / "train" / "train.txt").write_text(invalid, encoding="utf-8")
            report = validate(root)
            self.assertEqual(report["summary"]["status"], "FAIL")
            self.assertTrue(any("outside [0, 1]" in error for error in report["errors"]))

    def test_cross_split_duplicate_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            Image.new("RGB", (32, 32), "white").save(root / "images" / "val" / "val.jpg")
            report = validate(root)
            self.assertEqual(report["summary"]["status"], "FAIL")
            self.assertTrue(any("cross-split duplicate" in error for error in report["errors"]))

    def test_missing_label_is_background_negative(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            (root / "labels" / "train" / "train.txt").unlink()
            report = validate(root)
            self.assertEqual(report["summary"]["status"], "PASS")
            self.assertEqual(report["splits"]["train"]["background_negatives"], 1)


if __name__ == "__main__":
    unittest.main()
