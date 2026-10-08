#!/usr/bin/env python3
"""Validate the prepared success/error classifier dataset and grouped split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .case_classifier_utils import CLASSES, SPLITS, verify_image, write_json
except ImportError:  # Direct execution: python ml/scripts/validate_case_classifier.py
    from case_classifier_utils import CLASSES, SPLITS, verify_image, write_json


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    report: dict[str, Any] = {
        "dataset": str(root),
        "errors": [],
        "warnings": [],
        "splits": {},
    }
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        report["errors"].append(f"Missing manifest: {manifest_path}")
        return _finish(report)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["errors"].append(f"Cannot read manifest: {exc}")
        return _finish(report)

    source_splits: dict[str, set[str]] = defaultdict(set)
    seen_sources: set[str] = set()
    disk_counts: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[str] = Counter()
    expected_size = manifest.get("classifier_input_size")
    if not isinstance(expected_size, int) or expected_size <= 0:
        report["errors"].append("manifest is missing a valid classifier_input_size")

    for index, item in enumerate(manifest.get("images", [])):
        prefix = f"manifest image {index}"
        split = item.get("split")
        label = item.get("label")
        source_id = item.get("source_id")
        source_path = item.get("source_path")
        status = item.get("status")
        if split not in SPLITS:
            report["errors"].append(f"{prefix}: invalid split {split!r}")
            continue
        if label not in CLASSES:
            report["errors"].append(f"{prefix}: invalid label {label!r}")
            continue
        if not isinstance(source_id, str) or not source_id:
            report["errors"].append(f"{prefix}: missing source_id")
            continue
        if not isinstance(source_path, str) or source_path in seen_sources:
            report["errors"].append(f"{prefix}: missing or duplicate source_path {source_path!r}")
        else:
            seen_sources.add(source_path)
        source_splits[source_id].add(split)
        status_counts[str(status)] += 1
        if status == "no_detection":
            continue
        if status != "prepared":
            report["errors"].append(f"{prefix}: unsupported status {status!r}")
            continue

        relative = item.get("prepared_path")
        if not isinstance(relative, str):
            report["errors"].append(f"{prefix}: missing prepared_path")
            continue
        path = root / relative
        expected_parent = root / split / label
        if path.parent != expected_parent:
            report["errors"].append(f"{prefix}: prepared_path is not under {expected_parent}")
        if not path.is_file():
            report["errors"].append(f"{prefix}: missing prepared crop {path}")
            continue
        try:
            width, height = verify_image(path)
            if isinstance(expected_size, int) and (width, height) != (expected_size, expected_size):
                report["errors"].append(
                    f"{path}: expected {expected_size}x{expected_size}, found {width}x{height}"
                )
        except Exception as exc:
            report["errors"].append(f"{path}: cannot be decoded ({exc})")
        disk_counts[(split, label)] += 1

    for source_id, splits in sorted(source_splits.items()):
        if len(splits) > 1:
            report["errors"].append(
                f"source leakage: {source_id} appears in {', '.join(sorted(splits))}"
            )

    for split in SPLITS:
        report["splits"][split] = {
            label: disk_counts[(split, label)] for label in CLASSES
        }
        for label in CLASSES:
            folder = root / split / label
            if not folder.is_dir():
                report["errors"].append(f"Missing class directory: {folder}")
                continue
            actual = sum(path.is_file() for path in folder.iterdir())
            expected = disk_counts[(split, label)]
            if actual != expected:
                report["errors"].append(
                    f"{folder}: found {actual} files but manifest contains {expected} prepared crops"
                )

    missed = status_counts["no_detection"]
    if missed:
        report["warnings"].append(
            f"{missed} source images had no pose detection and were intentionally excluded from classifier input"
        )
    report["status_counts"] = dict(status_counts)
    return _finish(report)


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    report["summary"] = {
        "status": "PASS" if not report["errors"] else "FAIL",
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"]),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/case-classifier"))
    parser.add_argument("--report", type=Path, default=Path("reports/case-classifier/dataset-validation.json"))
    args = parser.parse_args()
    report = validate(args.dataset)
    write_json(args.report, report)
    print(json.dumps(report["summary"], indent=2))
    for split, counts in report.get("splits", {}).items():
        print(f"{split}: {counts}")
    return 0 if report["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
