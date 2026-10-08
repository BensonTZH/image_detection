import unittest

from ml.scripts.pose_guidance_metrics import (
    alignment_metrics,
    corners_reliable,
    default_guidance,
    numeric_summary,
    range_overlap,
    success_reference,
)


def point(x: float, y: float, confidence: float = 0.9) -> dict[str, float]:
    return {"x": x, "y": y, "confidence": confidence}


class PoseGuidanceMetricsTests(unittest.TestCase):
    def test_rectangle_matches_browser_alignment_measurements(self) -> None:
        box = (60, 80, 140, 320)
        keypoints = (
            point(70, 100),
            point(130, 100),
            point(130, 300),
            point(70, 300),
        )
        metrics = alignment_metrics(box, keypoints, 200, 400, 0.8)
        self.assertAlmostEqual(metrics["center_x"], 0.5)
        self.assertAlmostEqual(metrics["center_y"], 0.5)
        self.assertAlmostEqual(metrics["frame_area"], 0.24)
        self.assertAlmostEqual(metrics["roll_degrees"], 0)
        self.assertAlmostEqual(metrics["side_perspective"], 0)
        self.assertAlmostEqual(metrics["vertical_perspective"], 0)
        self.assertTrue(metrics["corners_reliable"])

    def test_perspective_signs_match_guidance_rules(self) -> None:
        box = (10, 10, 90, 190)
        keypoints = (
            point(20, 20),
            point(80, 20),
            point(75, 160),
            point(25, 180),
        )
        metrics = alignment_metrics(box, keypoints, 100, 200, 0.8)
        self.assertGreater(metrics["vertical_perspective"], 0)
        self.assertGreater(metrics["side_perspective"], 0)
        self.assertTrue(metrics["corners_reliable"])

    def test_low_confidence_corner_is_not_reliable(self) -> None:
        box = (10, 10, 90, 190)
        keypoints = (
            point(20, 20),
            point(80, 20),
            point(80, 180, confidence=0.2),
            point(20, 180),
        )
        self.assertFalse(corners_reliable(box, keypoints, 0.35))
        self.assertFalse(alignment_metrics(box, keypoints, 100, 200, 0.8)["corners_reliable"])

    def test_default_guidance_matches_browser_priority(self) -> None:
        row = {
            "status": "detected",
            "corners_reliable": True,
            "center_x": 0.2,
            "frame_area": 0.01,
            "roll_degrees": 12.0,
            "side_perspective": 0.20,
            "vertical_perspective": 0.30,
        }
        # Side perspective is checked before tilt, centring, distance, and roll.
        self.assertEqual(default_guidance(row), "step_right")
        row["side_perspective"] = 0.0
        self.assertEqual(default_guidance(row), "tilt_down")
        row["vertical_perspective"] = 0.0
        self.assertEqual(default_guidance(row), "aim_left")

    def test_numeric_summary_uses_interpolated_quantiles(self) -> None:
        summary = numeric_summary([0, 10, 20, 30, 40])
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["median"], 20)
        self.assertEqual(summary["p10"], 4)
        self.assertEqual(summary["p90"], 36)

    def test_reference_uses_only_reliable_train_success_rows(self) -> None:
        def row(split: str, label: str, center: float, reliable: bool = True) -> dict[str, object]:
            return {
                "split": split,
                "label": label,
                "status": "detected",
                "corners_reliable": reliable,
                "center_x": center,
                "center_y": 0.5,
                "frame_area": 0.1,
                "roll_degrees": 0.0,
                "side_perspective": 0.0,
                "vertical_perspective": 0.0,
            }

        rows = [
            row("train", "success", 0.4),
            row("train", "success", 0.6),
            row("train", "error", 0.9),
            row("val", "success", 0.5),
            row("train", "success", 0.99, reliable=False),
        ]
        reference = success_reference(rows)
        self.assertEqual(reference["images"], 2)
        self.assertAlmostEqual(reference["features"]["center_x"]["median"], 0.5)
        overlap = range_overlap(rows, reference)
        self.assertEqual(overlap["success"]["center_x"]["images"], 1)
        self.assertEqual(overlap["success"]["center_x"]["rate"], 1)


if __name__ == "__main__":
    unittest.main()
