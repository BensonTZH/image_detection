"""Pure geometry and reporting helpers for CupDetector pose analysis."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

KEYPOINT_NAMES = ("tl", "tr", "br", "bl")
METRIC_NAMES = (
    "detection_confidence",
    "center_x",
    "center_y",
    "frame_area",
    "roll_degrees",
    "side_perspective",
    "vertical_perspective",
    "min_corner_confidence",
)
REFERENCE_METRICS = (
    "center_x",
    "center_y",
    "frame_area",
    "roll_degrees",
    "side_perspective",
    "vertical_perspective",
)

DEFAULT_REFERENCE = {
    "center_x": 0.5,
    "center_tolerance": 0.10,
    "target_area": 0.115,
    "area_tolerance": 0.085,
    "target_roll_degrees": 0.0,
    "target_vertical_perspective": 0.0,
    "target_side_perspective": 0.0,
    "maximum_roll_degrees": 5.0,
    "maximum_side_perspective": 0.18,
    "maximum_vertical_perspective": 0.24,
}


def _distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return math.hypot(second["x"] - first["x"], second["y"] - first["y"])


def _line_orientation(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    degrees = math.degrees(math.atan2(second["y"] - first["y"], second["x"] - first["x"]))
    return (degrees + 180) % 180


def _orientation_difference(first: float, second: float) -> float:
    difference = abs(first - second)
    return min(difference, 180 - difference)


def corners_reliable(
    box: Sequence[float],
    keypoints: Sequence[Mapping[str, float]],
    confidence_threshold: float = 0.35,
) -> bool:
    """Match the corner-reliability checks in ``web/lib/pose.ts``."""
    if len(box) != 4 or len(keypoints) != 4:
        return False
    if any(
        not math.isfinite(float(point.get(key, math.nan)))
        for point in keypoints
        for key in ("x", "y", "confidence")
    ):
        return False
    if any(float(point["confidence"]) < confidence_threshold for point in keypoints):
        return False
    left, top, right, bottom = (float(value) for value in box)
    box_width, box_height = right - left, bottom - top
    if box_width <= 0 or box_height <= 0:
        return False
    tl, tr, br, bl = keypoints
    if (tl["y"] + tr["y"]) / 2 >= (bl["y"] + br["y"]) / 2:
        return False
    if (tl["x"] + bl["x"]) / 2 >= (tr["x"] + br["x"]) / 2:
        return False
    top_width, bottom_width = _distance(tl, tr), _distance(bl, br)
    left_height, right_height = _distance(tl, bl), _distance(tr, br)
    if min(top_width, bottom_width) < box_width * 0.15:
        return False
    if min(left_height, right_height) < box_height * 0.30:
        return False
    if _orientation_difference(_line_orientation(tl, tr), _line_orientation(bl, br)) > 25:
        return False
    if _orientation_difference(_line_orientation(tl, bl), _line_orientation(tr, br)) > 25:
        return False
    return True


def alignment_metrics(
    box: Sequence[float],
    keypoints: Sequence[Mapping[str, float]],
    image_width: int,
    image_height: int,
    detection_confidence: float,
    keypoint_confidence_threshold: float = 0.35,
) -> dict[str, Any]:
    """Calculate the same pose measurements used by CupDetector.

    ``vertical_perspective`` is the web app's width difference and
    ``side_perspective`` is its height difference. Keeping those names explicit
    makes the generated report easier to interpret.
    """
    if len(box) != 4 or len(keypoints) != 4:
        raise ValueError("A box and exactly four TL/TR/BR/BL keypoints are required")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    left, top, right, bottom = (float(value) for value in box)
    tl, tr, br, bl = keypoints
    top_width, bottom_width = _distance(tl, tr), _distance(bl, br)
    left_height, right_height = _distance(tl, bl), _distance(tr, br)
    vertical_perspective = (top_width - bottom_width) / max(top_width, bottom_width, 1)
    side_perspective = (left_height - right_height) / max(left_height, right_height, 1)
    metrics: dict[str, Any] = {
        "detection_confidence": float(detection_confidence),
        "center_x": ((left + right) / 2) / image_width,
        "center_y": ((top + bottom) / 2) / image_height,
        "frame_area": ((right - left) * (bottom - top)) / (image_width * image_height),
        "roll_degrees": math.degrees(math.atan2(tr["y"] - tl["y"], tr["x"] - tl["x"])),
        "top_width": top_width,
        "bottom_width": bottom_width,
        "left_height": left_height,
        "right_height": right_height,
        "vertical_perspective": vertical_perspective,
        "side_perspective": side_perspective,
        "perspective_difference": max(abs(vertical_perspective), abs(side_perspective)),
        "min_corner_confidence": min(float(point["confidence"]) for point in keypoints),
        "corners_reliable": corners_reliable(box, keypoints, keypoint_confidence_threshold),
    }
    return metrics


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated quantile without requiring NumPy."""
    if not values:
        raise ValueError("At least one value is required")
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values: Iterable[float]) -> dict[str, float | int] | None:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return None
    return {
        "count": len(cleaned),
        "minimum": min(cleaned),
        "p05": quantile(cleaned, 0.05),
        "p10": quantile(cleaned, 0.10),
        "p25": quantile(cleaned, 0.25),
        "median": quantile(cleaned, 0.50),
        "mean": fmean(cleaned),
        "p75": quantile(cleaned, 0.75),
        "p90": quantile(cleaned, 0.90),
        "p95": quantile(cleaned, 0.95),
        "maximum": max(cleaned),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    detected = [row for row in rows if row.get("status") == "detected"]
    reliable = [row for row in detected if row.get("corners_reliable") is True]
    return {
        "images": len(rows),
        "detected": len(detected),
        "detection_rate": len(detected) / max(len(rows), 1),
        "reliable_corners": len(reliable),
        "reliable_corner_rate_all_images": len(reliable) / max(len(rows), 1),
        "reliable_corner_rate_detected": len(reliable) / max(len(detected), 1),
        "default_guidance": dict(sorted(_guidance_counts(rows).items())),
        "metrics": {
            name: numeric_summary(
                float(row[name])
                for row in (detected if name in ("detection_confidence", "center_x", "center_y", "frame_area") else reliable)
                if isinstance(row.get(name), (int, float))
            )
            for name in METRIC_NAMES
        },
    }


def default_guidance(row: Mapping[str, Any]) -> str:
    """Apply the browser's default single-frame guidance priority to a row."""
    if row.get("status") != "detected":
        return "searching"
    center_x = float(row["center_x"])
    area = float(row["frame_area"])
    if row.get("corners_reliable") is not True:
        if center_x < DEFAULT_REFERENCE["center_x"] - DEFAULT_REFERENCE["center_tolerance"]:
            return "aim_left"
        if center_x > DEFAULT_REFERENCE["center_x"] + DEFAULT_REFERENCE["center_tolerance"]:
            return "aim_right"
        if area < DEFAULT_REFERENCE["target_area"] - DEFAULT_REFERENCE["area_tolerance"]:
            return "closer"
        if area > DEFAULT_REFERENCE["target_area"] + DEFAULT_REFERENCE["area_tolerance"]:
            return "farther"
        return "steady"

    side_difference = float(row["side_perspective"]) - DEFAULT_REFERENCE["target_side_perspective"]
    vertical_difference = float(row["vertical_perspective"]) - DEFAULT_REFERENCE["target_vertical_perspective"]
    roll_difference = float(row["roll_degrees"]) - DEFAULT_REFERENCE["target_roll_degrees"]
    if abs(side_difference) > DEFAULT_REFERENCE["maximum_side_perspective"]:
        return "step_right" if side_difference > 0 else "step_left"
    if abs(vertical_difference) > DEFAULT_REFERENCE["maximum_vertical_perspective"]:
        return "tilt_down" if vertical_difference > 0 else "tilt_up"
    if center_x < DEFAULT_REFERENCE["center_x"] - DEFAULT_REFERENCE["center_tolerance"]:
        return "aim_left"
    if center_x > DEFAULT_REFERENCE["center_x"] + DEFAULT_REFERENCE["center_tolerance"]:
        return "aim_right"
    if area < DEFAULT_REFERENCE["target_area"] - DEFAULT_REFERENCE["area_tolerance"]:
        return "closer"
    if area > DEFAULT_REFERENCE["target_area"] + DEFAULT_REFERENCE["area_tolerance"]:
        return "farther"
    if abs(roll_difference) > DEFAULT_REFERENCE["maximum_roll_degrees"]:
        return "rotate_right" if roll_difference > 0 else "rotate_left"
    return "aligned"


def _guidance_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        guidance = str(row.get("default_guidance") or default_guidance(row))
        counts[guidance] = counts.get(guidance, 0) + 1
    return counts


def success_reference(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe train-success ranges; do not mutate the app's thresholds."""
    selected = [
        row for row in rows
        if row.get("split") == "train"
        and row.get("label") == "success"
        and row.get("status") == "detected"
        and row.get("corners_reliable") is True
    ]
    return {
        "source": "train/success reliable detections only",
        "images": len(selected),
        "features": {
            name: numeric_summary(float(row[name]) for row in selected)
            for name in REFERENCE_METRICS
        },
    }


def range_overlap(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure validation rows inside each train-success p05-p95 range."""
    output: dict[str, Any] = {}
    for label in ("success", "error"):
        selected = [
            row for row in rows
            if row.get("split") == "val"
            and row.get("label") == label
            and row.get("status") == "detected"
            and row.get("corners_reliable") is True
        ]
        feature_results: dict[str, Any] = {}
        for name in REFERENCE_METRICS:
            bounds = reference.get("features", {}).get(name)
            if not bounds:
                feature_results[name] = None
                continue
            inside = sum(bounds["p05"] <= float(row[name]) <= bounds["p95"] for row in selected)
            feature_results[name] = {
                "inside": inside,
                "images": len(selected),
                "rate": inside / max(len(selected), 1),
                "range": [bounds["p05"], bounds["p95"]],
            }
        output[label] = feature_results
    return output
