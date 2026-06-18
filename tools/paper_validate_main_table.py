#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PAPER_PROTOCOL_ID = "ogsod_hbb_mosaic100_clean_a1b_probea_20260618"
ALLOWED_SEEDS = {"0", "42", "123"}
FORBIDDEN_TOKENS = ("smoke", "probe_run", "partial", "snapshot", "diagnostic", "archive", "old")
LADD_FORBIDDEN_NOTES = ("a2", "bn-freeze", "bn_freeze", "no-mosaic", "nomosaic", "a1-a2-b")

REQUIRED_COLUMNS = [
    "dataset",
    "task",
    "protocol_id",
    "method",
    "method_display",
    "model_size",
    "seed",
    "init_type",
    "student_modality",
    "teacher_modality",
    "inference_modality",
    "imgsz",
    "epochs",
    "batch",
    "mosaic",
    "close_mosaic",
    "phase_chain",
    "ladd_mode",
    "run_tag",
    "project_dir",
    "results_csv",
    "args_yaml",
    "manifest",
    "git_commit",
    "best_ap50_95",
    "best_ap50",
    "final_ap50_95",
    "final_ap50",
    "best_epoch",
    "status",
    "claim_usable",
    "notes",
]


def norm(value: object) -> str:
    return str(value or "").strip()


def is_float(value: str, expected: float) -> bool:
    try:
        return float(value) == expected
    except ValueError:
        return False


def row_text(row: dict[str, str]) -> str:
    return " ".join(norm(v) for v in row.values()).lower()


def validate_row(row: dict[str, str], line: int) -> list[str]:
    errors: list[str] = []
    text = row_text(row)
    method = norm(row.get("method")).lower()
    run_tag = norm(row.get("run_tag"))
    notes = norm(row.get("notes")).lower()

    checks = {
        "protocol_id": PAPER_PROTOCOL_ID,
        "imgsz": "256",
        "epochs": "800",
    }
    for field, expected in checks.items():
        value = norm(row.get(field))
        if value != expected:
            errors.append(f"line {line}: {field}={value!r}, expected {expected!r}")

    if not is_float(norm(row.get("mosaic")), 1.0):
        errors.append(f"line {line}: mosaic={norm(row.get('mosaic'))!r}, expected 1.0")
    if norm(row.get("close_mosaic")) != "700":
        errors.append(f"line {line}: close_mosaic={norm(row.get('close_mosaic'))!r}, expected '700'")
    if norm(row.get("seed")) not in ALLOWED_SEEDS:
        errors.append(f"line {line}: seed={norm(row.get('seed'))!r}, expected one of 0, 42, 123")
    if not run_tag:
        errors.append(f"line {line}: missing run_tag")
    if not norm(row.get("results_csv")):
        errors.append(f"line {line}: missing results_csv")
    if norm(row.get("claim_usable")).lower() != "yes":
        errors.append(f"line {line}: claim_usable must be yes for a main-table row")
    if norm(row.get("status")).lower() not in {"verified", "main_table"}:
        errors.append(f"line {line}: status={norm(row.get('status'))!r}, expected verified/main_table")

    for token in FORBIDDEN_TOKENS:
        if token in text:
            errors.append(f"line {line}: forbidden main-table token found: {token}")

    if "ladd" in method:
        if norm(row.get("ladd_mode")) != "dynamic_probe":
            errors.append(f"line {line}: LADD row must use ladd_mode=dynamic_probe")
        if "clean_a1b_dynprobe" not in run_tag:
            errors.append(f"line {line}: LADD run_tag must contain clean_a1b_dynprobe")
        if norm(row.get("phase_chain")) != "A1->B":
            errors.append(f"line {line}: LADD row must use phase_chain=A1->B")
        for token in LADD_FORBIDDEN_NOTES:
            if token in notes:
                errors.append(f"line {line}: LADD notes contain forbidden historical token: {token}")

    if "cmdistill" in method and "vedai" in text:
        errors.append(f"line {line}: OGSOD main table must not use CMDistill native VEDAI archive path")
    if "cclkd" in method:
        if "online" not in text:
            errors.append(f"line {line}: CCLKD main-table row must be online")
        if "frozen" in text or "comparison_kd_profile" in text:
            errors.append(f"line {line}: CCLKD main-table row must not be a frozen-teacher profile")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a LADD paper main-table CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [field for field in REQUIRED_COLUMNS if field not in (reader.fieldnames or [])]
        if missing:
            print("FAIL: missing required columns: " + ", ".join(missing))
            return 1
        rows = list(reader)

    errors: list[str] = []
    for line, row in enumerate(rows, start=2):
        errors.extend(validate_row(row, line))

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print(f"PASS: {len(rows)} rows validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
