#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return math.nan


def fail_if_bad_number(errors: list[str], row: dict[str, str], key: str):
    value = parse_float(row, key, math.nan)
    if not math.isfinite(value):
        errors.append(f"{key} is not finite: {row.get(key)!r}")


def check_run(run_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    diagnostics_path = run_dir / "cclkd_yolov5_diagnostics.csv"
    errors: list[str] = []
    warnings: list[str] = []
    if not diagnostics_path.exists():
        return [f"missing diagnostics file: {diagnostics_path}"], warnings, {}

    with diagnostics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [f"diagnostics file has no rows: {diagnostics_path}"], warnings, {}

    row = rows[-1]
    mode = row.get("mode", "")
    nan_or_inf = parse_float(row, "nan_or_inf_detected", math.nan)
    if not math.isfinite(nan_or_inf) or nan_or_inf != 0.0:
        errors.append(f"nan_or_inf_detected is nonzero or invalid: {row.get('nan_or_inf_detected')!r}")

    fail_if_bad_number(errors, row, "weighted_kd_to_student_det_ratio")
    ratio = parse_float(row, "weighted_kd_to_student_det_ratio", math.nan)
    if math.isfinite(ratio) and ratio > 10:
        warnings.append(f"weighted_kd_to_student_det_ratio is high: {ratio:.6g}")

    if mode.startswith("paper_"):
        feature_capture_ok = parse_float(row, "feature_capture_ok", 0.0)
        student_levels = parse_float(row, "student_feature_levels", 0.0)
        teacher_levels = parse_float(row, "teacher_feature_levels", 0.0)
        valid = parse_float(row, "cop_valid_candidates", 0.0)
        positives = parse_float(row, "cop_positive_candidates", 0.0)
        positive_ratio = parse_float(row, "cop_positive_ratio", math.nan)
        if feature_capture_ok < 1:
            errors.append(f"feature_capture_ok < 1: {feature_capture_ok}")
        if student_levels < 1:
            errors.append(f"student_feature_levels < 1: {student_levels}")
        if teacher_levels < 1:
            errors.append(f"teacher_feature_levels < 1: {teacher_levels}")
        if valid <= 0:
            errors.append(f"cop_valid_candidates <= 0: {valid}")
        if positives <= 0:
            errors.append(f"cop_positive_candidates <= 0: {positives}")
        if not math.isfinite(positive_ratio) or not (0.0 < positive_ratio <= 1.0):
            errors.append(f"cop_positive_ratio outside (0, 1]: {row.get('cop_positive_ratio')!r}")

    return errors, warnings, row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    errors, warnings, row = check_run(args.run_dir)
    print("# YOLOv5 CCLKD Smoke Check")
    print()
    print(f"- run_dir: `{args.run_dir}`")
    if row:
        print(f"- mode: `{row.get('mode', '')}`")
        print(f"- epoch: `{row.get('epoch', '')}`")
        print(f"- cop_positive_ratio: `{row.get('cop_positive_ratio', '')}`")
        print(f"- weighted_kd_to_student_det_ratio: `{row.get('weighted_kd_to_student_det_ratio', '')}`")
        print(f"- feature_capture_ok: `{row.get('feature_capture_ok', '')}`")
    if warnings:
        print()
        print("## Warnings")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print()
        print("## Errors")
        for error in errors:
            print(f"- {error}")
        return 1
    print()
    print("PASS: smoke diagnostics look usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
