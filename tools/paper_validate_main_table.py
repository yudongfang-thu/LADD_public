#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ALLOWED_METHODS = {
    "sar_baseline",
    "rgb_teacher",
    "vanilla_feature_kd",
    "fgd",
    "ld",
    "cmdistill",
    "hallucidet_yolo",
    "cclkd_online",
    "ladd_probea",
    "ladd_static_ablation",
    "ladd_dynamic_ablation",
}


def is_yes(value: str) -> bool:
    return value.strip().lower() == "yes"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paper main-table candidate CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    errors: list[str] = []
    usable_rows = [row for row in rows if is_yes(row.get("usable_for_main_table", ""))]
    seen: dict[tuple[str, str, str, str], int] = {}
    baselines: set[tuple[str, str, str]] = set()

    for i, row in enumerate(rows, start=2):
        method = row.get("method", "")
        if method and method not in ALLOWED_METHODS:
            errors.append(f"line {i}: method not whitelisted: {method}")
        if method in {"sar_baseline", "rgb_teacher"} and row.get("protocol") == "mosaic100":
            baselines.add((method, row.get("model_size", ""), row.get("seed", "")))

    for i, row in enumerate(usable_rows, start=1):
        line = rows.index(row) + 2
        key = (row.get("dataset", ""), row.get("method", ""), row.get("model_size", ""), row.get("seed", ""))
        if key in seen:
            errors.append(f"line {line}: duplicate verified row for {key}; first seen at usable row {seen[key]}")
        seen[key] = i
        checks = {
            "protocol": "mosaic100",
            "imgsz": "256",
            "epochs": "800",
            "mosaic": "1.0",
            "close_mosaic": "700",
            "status": "verified",
        }
        for field, expected in checks.items():
            value = row.get(field, "")
            if field == "mosaic":
                ok = value in {"1", "1.0"}
            else:
                ok = value == expected
            if not ok:
                errors.append(f"line {line}: {field}={value!r}, expected {expected!r}")
        for field in ("results_csv", "args_yaml", "manifest", "code_commit"):
            if not row.get(field):
                errors.append(f"line {line}: missing {field}")
        method = row.get("method", "")
        if method == "ladd_probea":
            if row.get("phase_chain") != "A1->B":
                errors.append(f"line {line}: LADD must use phase_chain=A1->B")
            if "a2" in row.get("run_tag", "").lower():
                errors.append(f"line {line}: LADD run_tag appears to contain A2")
        if method == "cmdistill" and "affine" not in (row.get("notes", "") + row.get("manifest", "") + row.get("run_tag", "")).lower():
            # The collector already checks meta when available; this warning catches hand-authored rows.
            errors.append(f"line {line}: CMDistill usable row must document KD_CALIBRATION_MODE=affine")
        if method in {"ladd_probea", "fgd", "ld", "cmdistill"}:
            size_seed = (row.get("model_size", ""), row.get("seed", ""))
            if ("sar_baseline", *size_seed) not in baselines:
                errors.append(f"line {line}: missing same-size/seed SAR baseline row for {size_seed}")
            if ("rgb_teacher", *size_seed) not in baselines:
                errors.append(f"line {line}: missing same-size/seed RGB teacher row for {size_seed}")

    invalid_usable = [
        row.get("paper_result_id", f"line {idx + 2}")
        for idx, row in enumerate(rows)
        if row.get("status") in {"archive", "diagnostic", "smoke", "partial", "invalid"}
        and is_yes(row.get("usable_for_main_table", ""))
    ]
    if invalid_usable:
        errors.append("invalid/diagnostic rows marked usable: " + ", ".join(invalid_usable))

    if errors:
        print("ERROR: paper main-table validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"OK: {args.csv_path} passed paper main-table validation ({len(usable_rows)} usable rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
