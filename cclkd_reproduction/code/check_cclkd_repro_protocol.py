#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ALLOWED_MODELS = {"n", "s"}
EXPECTED_EPOCHS = 400
EXPECTED_BATCH = 32
EXPECTED_OPTIMIZER = "SGD"
EXPECTED_LR0 = 0.01
EXPECTED_MOSAIC = 1.0


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or line.startswith((" ", "\t")) or ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key == "nc":
            data[key] = int(value)
        elif key in {"train", "val", "test", "path"}:
            data[key] = value
        elif key == "names":
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                data[key] = [] if not inner else [x.strip().strip("'\"") for x in inner.split(",")]
            elif value.startswith("{") and value.endswith("}"):
                inner = value[1:-1].strip()
                entries = [] if not inner else inner.split(",")
                data[key] = {idx: item.split(":", 1)[-1].strip().strip("'\"") for idx, item in enumerate(entries)}
            elif value:
                data[key] = value
            else:
                names: list[str] = []
                j = i + 1
                while j < len(lines) and (lines[j].startswith((" ", "\t")) or not lines[j].strip()):
                    child = lines[j].split("#", 1)[0].strip()
                    if child.startswith("-"):
                        names.append(child[1:].strip().strip("'\""))
                    elif ":" in child:
                        names.append(child.split(":", 1)[1].strip().strip("'\""))
                    j += 1
                data[key] = names
                i = j - 1
        i += 1
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping.")
    return data


def _names_len(names: Any) -> int | None:
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, (list, tuple)):
        return len(names)
    return None


def _check_dataset(path: Path, label: str) -> list[str]:
    errors: list[str] = []
    data = _load_yaml(path)
    nc = data.get("nc")
    names = data.get("names")
    if int(nc) != 3:
        errors.append(f"{label}: expected nc=3, got {nc!r} in {path}.")
    names_count = _names_len(names)
    if names_count is not None and names_count != 3:
        errors.append(f"{label}: expected 3 class names, got {names_count} in {path}.")
    for split in ("train", "val"):
        if split not in data:
            errors.append(f"{label}: missing '{split}' entry in {path}.")
    return errors


def _is_close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast checker for CCLKD paper-protocol reproduction runs.")
    parser.add_argument("--model-size", choices=sorted(ALLOWED_MODELS), required=True)
    parser.add_argument("--student-data", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--mosaic", type=float, required=True)
    parser.add_argument("--mixup", type=float, required=True)
    parser.add_argument("--online-trainer", action="store_true")
    parser.add_argument("--student-weights", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--allow-unverified-mixup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    errors += _check_dataset(args.student_data, "student/SAR data")
    errors += _check_dataset(args.teacher_data, "teacher/RGB data")

    if args.epochs != EXPECTED_EPOCHS:
        errors.append(f"expected epochs={EXPECTED_EPOCHS}, got {args.epochs}.")
    if args.batch != EXPECTED_BATCH:
        errors.append(f"expected batch={EXPECTED_BATCH}, got {args.batch}.")
    if args.optimizer != EXPECTED_OPTIMIZER:
        errors.append(f"expected optimizer={EXPECTED_OPTIMIZER}, got {args.optimizer}.")
    if not _is_close(args.lr0, EXPECTED_LR0):
        errors.append(f"expected lr0={EXPECTED_LR0}, got {args.lr0}.")
    if not _is_close(args.mosaic, EXPECTED_MOSAIC):
        errors.append(f"expected mosaic={EXPECTED_MOSAIC}, got {args.mosaic}.")
    if args.mixup <= 0 and not args.allow_unverified_mixup:
        errors.append("CCLKD paper protocol uses MixUp; pass --mixup > 0 or --allow-unverified-mixup with a note.")
    if not args.online_trainer:
        errors.append("online teacher-student trainer flag is required; frozen teacher runs are not paper reproduction.")

    expected_weight_name = f"yolo11{args.model_size}.pt"
    for role, path in (("student", args.student_weights), ("teacher", args.teacher_weights)):
        if path.name != expected_weight_name:
            errors.append(f"{role} weights must be {expected_weight_name}, got {path}.")

    if errors:
        print("CCLKD reproduction protocol check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(2)
    print("CCLKD reproduction protocol check passed.")


if __name__ == "__main__":
    main()
