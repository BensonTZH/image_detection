from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ml.scripts.training_utils import validate_dataset_layout, write_resolved_data_yaml


class Phase2CommonTests(unittest.TestCase):
    def test_missing_dataset_directories_fail(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                validate_dataset_layout(Path(directory))

    def test_resolved_yaml_contains_absolute_dataset_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for split in ("train", "val"):
                (dataset / "images" / split).mkdir(parents=True)
                (dataset / "labels" / split).mkdir(parents=True)
            output = root / "resolved.yaml"
            write_resolved_data_yaml(dataset, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn(str(dataset.resolve()), text)
            self.assertIn("kpt_shape: [4, 3]", text)
            self.assertIn("0: slot", text)


if __name__ == "__main__":
    unittest.main()
