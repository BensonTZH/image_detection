import unittest

from ml.scripts.onnx_utils import max_abs_difference, shape_for_json


class Phase3CommonTests(unittest.TestCase):
    def test_max_abs_difference(self) -> None:
        self.assertAlmostEqual(max_abs_difference([1, 5, -2], [1.5, 4, -2]), 1.0)

    def test_difference_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            max_abs_difference([1], [1, 2])

    def test_shape_for_json_preserves_static_dimensions(self) -> None:
        self.assertEqual(shape_for_json([1, 3, 640, 640]), [1, 3, 640, 640])


if __name__ == "__main__":
    unittest.main()
