#!/usr/bin/env python3
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STRICT_PAIRS = [
    (
        ROOT / "ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py",
        ROOT / "ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py",
    ),
    (
        ROOT / "ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py",
        ROOT / "ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/trainer.py",
    ),
    (
        ROOT / "ladd/code/src/teacher_student_decomposition_kd_hbb/model.py",
        ROOT / "ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/model.py",
    ),
    (
        ROOT / "ladd/code/src/teacher_student_decomposition_kd_hbb/base_hbb.py",
        ROOT / "ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/base_hbb.py",
    ),
    (
        ROOT / "ladd/code/src/teacher_student_decomposition_kd_hbb/schedule.py",
        ROOT / "ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/schedule.py",
    ),
]

ENTRY_PAIR = (
    ROOT / "ladd/code/train_ladd_hbb.py",
    ROOT / "ladd/code_versions/current_hbb/tools/train_ladd_hbb.py",
)

PHASE_SCRIPT = ROOT / "ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh"


def read(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Missing paper-critical file: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8").splitlines()


def normalized_entry(lines: list[str]) -> list[str]:
    out: list[str] = []
    skip_current_root = False
    for line in lines:
        if line.startswith("CURRENT_HBB_ROOT = "):
            skip_current_root = True
            continue
        if line.startswith("REPO_ROOT = "):
            out.append("REPO_ROOT = <normalized>")
            continue
        if line.startswith("SRC_ROOT = "):
            out.append("SRC_ROOT = <normalized>")
            continue
        if skip_current_root:
            skip_current_root = False
        out.append(line)
    return out


def diff_text(a_path: Path, b_path: Path, a_lines: list[str], b_lines: list[str]) -> str:
    return "\n".join(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=str(a_path.relative_to(ROOT)),
            tofile=str(b_path.relative_to(ROOT)),
            lineterm="",
        )
    )


def check_identical(a: Path, b: Path) -> list[str]:
    a_lines = read(a)
    b_lines = read(b)
    if a_lines == b_lines:
        return []
    return [diff_text(a, b, a_lines, b_lines)]


def check_entry_pair() -> list[str]:
    a, b = ENTRY_PAIR
    a_lines = normalized_entry(read(a))
    b_lines = normalized_entry(read(b))
    if a_lines == b_lines:
        return []
    return [diff_text(a, b, a_lines, b_lines)]


def check_phase_script() -> list[str]:
    text = "\n".join(read(PHASE_SCRIPT))
    required = [
        "ladd-b-a2-core",
        "ladd-b-frozen-reach-probe",
        "--resume",
        "COMPARISON_KD_PROFILE",
        "PROFILE_KD_REPLACE_BASE",
        "KD_CALIBRATION_MODE",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        return [f"{PHASE_SCRIPT.relative_to(ROOT)} missing required tokens: {', '.join(missing)}"]
    if not re.search(r"CLOSE_MOSAIC.*takes precedence|close_mosaic=.*RESOLVED_CLOSE_MOSAIC", text, re.S):
        return [f"{PHASE_SCRIPT.relative_to(ROOT)} does not clearly pass resolved close_mosaic."]
    return []


def main() -> int:
    failures: list[str] = []
    for a, b in STRICT_PAIRS:
        failures.extend(check_identical(a, b))
    failures.extend(check_entry_pair())
    failures.extend(check_phase_script())
    if failures:
        print("ERROR: current_hbb synchronization check failed.")
        for failure in failures:
            print()
            print(failure)
        return 1
    print("OK: current_hbb is synchronized with ladd/code for paper-critical files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
