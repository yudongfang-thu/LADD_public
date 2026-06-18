#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_PROTOCOL_ID = "ogsod_hbb_mosaic100_clean_a1b_probea_20260618"
ALLOWED_SEEDS = {"0", "42", "123"}
EXPECTED_DATASET_NAMES = {"bridge", "harbor", "storage_tank"}
INVALID_GIT_COMMITS = {"unknown", "dirty", "none", "null"}
FORBIDDEN_PATTERNS = (
    ("smoke", re.compile(r"(^|[_\-/\s])smoke($|[_\-/\s])", re.IGNORECASE)),
    ("partial", re.compile(r"(^|[_\-/\s])partial($|[_\-/\s])", re.IGNORECASE)),
    ("snapshot", re.compile(r"(^|[_\-/\s])snapshot($|[_\-/\s])", re.IGNORECASE)),
    ("diagnostic", re.compile(r"(^|[_\-/\s])diagnostic($|[_\-/\s])", re.IGNORECASE)),
    ("archive", re.compile(r"(^|[_\-/\s])archive($|[_\-/\s])", re.IGNORECASE)),
    ("old", re.compile(r"(^|[_\-/\s])old($|[_\-/\s])", re.IGNORECASE)),
    ("legacy", re.compile(r"(^|[_\-/\s])legacy($|[_\-/\s])", re.IGNORECASE)),
    ("bn-freeze", re.compile(r"bn[-_]freeze", re.IGNORECASE)),
    ("a1-a2-b", re.compile(r"a1[-_]a2[-_]b", re.IGNORECASE)),
    ("probe_only", re.compile(r"(^|[_\-/\s])probe[-_]only($|[_\-/\s])", re.IGNORECASE)),
    ("probe_run", re.compile(r"(^|[_\-/\s])probe[-_]run($|[_\-/\s])", re.IGNORECASE)),
    ("diagnostic_probe", re.compile(r"(^|[_\-/\s])diagnostic[-_]probe($|[_\-/\s])", re.IGNORECASE)),
)
LADD_FORBIDDEN_NOTE_PATTERNS = (
    ("a2", re.compile(r"(^|[_\-/\s])a2($|[_\-/\s])", re.IGNORECASE)),
    ("bn-freeze", re.compile(r"bn[-_]freeze", re.IGNORECASE)),
    ("no-mosaic", re.compile(r"no[-_]mosaic|nomosaic", re.IGNORECASE)),
    ("a1-a2-b", re.compile(r"a1[-_]a2[-_]b", re.IGNORECASE)),
)

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
AP_METRIC_FIELDS = ("best_ap50_95", "best_ap50", "final_ap50_95", "final_ap50")
NUMERIC_REQUIRED_FIELDS = (*AP_METRIC_FIELDS, "best_epoch")


def norm(value: object) -> str:
    return str(value or "").strip()


def is_float(value: str, expected: float) -> bool:
    try:
        return float(value) == expected
    except ValueError:
        return False


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def row_text(row: dict[str, str]) -> str:
    return " ".join(norm(v) for v in row.values()).lower()


def forbidden_labels(text: str) -> list[str]:
    return [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(text)]


def is_sha_like(value: str) -> bool:
    lowered = norm(value).lower()
    return lowered not in INVALID_GIT_COMMITS and re.fullmatch(r"[0-9a-f]{7,40}", lowered) is not None


def is_remote_reference(value: str) -> bool:
    text = norm(value)
    return (
        not text
        or text.startswith("<")
        or text.startswith(("remote:", "wandb:", "s3:", "gs:", "ssh:", "scp:"))
        or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text) is not None
    )


def resolve_local_path(value: str, csv_dir: Path) -> Path | None:
    text = norm(value)
    if is_remote_reference(text):
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    csv_candidate = csv_dir / path
    if csv_candidate.exists():
        return csv_candidate
    return repo_candidate


def parse_yaml_file(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        data: dict[str, object] = {}
        names: dict[str, str] = {}
        in_names = False
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("names:"):
                in_names = True
                continue
            if in_names and ":" in stripped and raw[:1].isspace():
                key, value = stripped.split(":", 1)
                names[key.strip()] = value.strip().strip("'\"")
                continue
            in_names = False
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                data[key.strip()] = value.strip().strip("'\"")
        if names:
            data["names"] = names
        return data


def yaml_value(data: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in data:
            return data[key]
    return None


def value_equals(value: object, expected: float) -> bool:
    try:
        return float(str(value)) == expected
    except (TypeError, ValueError):
        return False


def dataset_names(value: object) -> set[str]:
    if isinstance(value, dict):
        return {norm(v) for v in value.values() if norm(v)}
    if isinstance(value, list):
        return {norm(v) for v in value if norm(v)}
    if isinstance(value, str):
        return {part.strip().strip("'\"") for part in re.split(r"[,;\s]+", value) if part.strip()}
    return set()


def check_args_yaml(path: Path, line: int) -> list[str]:
    errors: list[str] = []
    data = parse_yaml_file(path)
    expected = (
        ("imgsz", ("imgsz",), 256.0),
        ("epochs", ("epochs",), 800.0),
        ("mosaic", ("mosaic",), 1.0),
        ("close_mosaic", ("close_mosaic", "close-mosaic"), 700.0),
    )
    for label, keys, target in expected:
        value = yaml_value(data, *keys)
        if value is None or not value_equals(value, target):
            errors.append(f"line {line}: args_yaml {path} has {label}={value!r}, expected {target:g}")
    return errors


def check_dataset_yaml(path: Path, field: str, line: int) -> list[str]:
    errors: list[str] = []
    data = parse_yaml_file(path)
    if not value_equals(data.get("nc"), 3.0):
        errors.append(f"line {line}: {field} {path} has nc={data.get('nc')!r}, expected 3")
    names = dataset_names(data.get("names"))
    if not EXPECTED_DATASET_NAMES.issubset(names):
        errors.append(f"line {line}: {field} {path} names={sorted(names)!r}, expected OGSOD HBB names")
    return errors


def validate_row(row: dict[str, str], line: int, *, csv_dir: Path, check_files: bool, check_yaml: bool) -> list[str]:
    errors: list[str] = []
    text = row_text(row)
    method = norm(row.get("method")).lower()
    run_tag = norm(row.get("run_tag"))
    notes = norm(row.get("notes")).lower()
    git_commit = norm(row.get("git_commit"))

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
    for field in ("results_csv", "args_yaml", "manifest", "git_commit"):
        if not norm(row.get(field)):
            errors.append(f"line {line}: missing source provenance field: {field}")
    if git_commit and not is_sha_like(git_commit):
        errors.append(f"line {line}: git_commit={git_commit!r}, expected sha-like 7-40 hex characters")
    if norm(row.get("claim_usable")).lower() != "yes":
        errors.append(f"line {line}: claim_usable must be yes for a main-table row")
    if norm(row.get("status")).lower() not in {"complete", "verified", "main_table"}:
        errors.append(f"line {line}: status={norm(row.get('status'))!r}, expected complete/verified/main_table")
    for field in NUMERIC_REQUIRED_FIELDS:
        value = norm(row.get(field))
        if not value:
            errors.append(f"line {line}: missing required numeric metric field: {field}")
            continue
        number = parse_number(value)
        if number is None:
            errors.append(f"line {line}: {field}={value!r}, expected numeric value")
            continue
        if field in AP_METRIC_FIELDS and not (0.0 <= number <= 1.0):
            errors.append(f"line {line}: {field}={value!r}, expected AP value in [0, 1]")
        if field == "best_epoch" and number < 0:
            errors.append(f"line {line}: best_epoch={value!r}, expected >= 0")

    for label in forbidden_labels(text):
        errors.append(f"line {line}: forbidden main-table pattern found: {label}")

    if "ladd" in method:
        if norm(row.get("ladd_mode")) != "dynamic_probe":
            errors.append(f"line {line}: LADD row must use ladd_mode=dynamic_probe")
        if "clean_a1b_dynprobe" not in run_tag:
            errors.append(f"line {line}: LADD run_tag must contain clean_a1b_dynprobe")
        if norm(row.get("phase_chain")) != "A1->B":
            errors.append(f"line {line}: LADD row must use phase_chain=A1->B")
        for label, pattern in LADD_FORBIDDEN_NOTE_PATTERNS:
            if pattern.search(notes):
                errors.append(f"line {line}: LADD notes contain forbidden historical pattern: {label}")

    if "cmdistill" in method and "vedai" in text:
        errors.append(f"line {line}: OGSOD main table must not use CMDistill native VEDAI archive path")
    if "cclkd" in method:
        if "online" not in text:
            errors.append(f"line {line}: CCLKD main-table row must be online")
        if "frozen" in text or "comparison_kd_profile" in text:
            errors.append(f"line {line}: CCLKD main-table row must not be a frozen-teacher profile")

    path_fields = ("results_csv", "args_yaml", "manifest", "data_yaml", "teacher_data_yaml", "student_data_yaml")
    local_paths: dict[str, Path] = {}
    for field in path_fields:
        value = norm(row.get(field))
        if not value:
            continue
        local_path = resolve_local_path(value, csv_dir)
        if local_path is None:
            continue
        local_paths[field] = local_path
        if check_files and not local_path.is_file():
            errors.append(f"line {line}: local provenance path does not exist: {field}={value}")

    if check_yaml:
        args_path = local_paths.get("args_yaml")
        if args_path and args_path.is_file():
            errors.extend(check_args_yaml(args_path, line))
        for field in ("data_yaml", "teacher_data_yaml", "student_data_yaml"):
            dataset_path = local_paths.get(field)
            if dataset_path and dataset_path.is_file():
                errors.extend(check_dataset_yaml(dataset_path, field, line))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a LADD paper main-table CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--check-files", action="store_true", help="Require local-looking provenance paths to exist.")
    parser.add_argument("--check-yaml", action="store_true", help="Parse local args/data YAMLs and validate paper protocol fields.")
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
        errors.extend(
            validate_row(
                row,
                line,
                csv_dir=args.csv_path.resolve().parent,
                check_files=args.check_files,
                check_yaml=args.check_yaml,
            )
        )

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print(f"PASS: {len(rows)} rows validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
