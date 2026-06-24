#!/usr/bin/env python3
"""Inspect capR/KD settings and diagnostics for one LADD run directory."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is normally available in the training env
    yaml = None


SUMMARY_KEYS = (
    "rank_d_neg_cap",
    "normalize_reach",
    "capR_effectively_enabled",
    "kd_weight_mode",
    "kd_target_branch",
    "kd_reach_use_capped_gap",
    "kd_reach_tau",
    "alpha_kd",
    "alpha_s_rec",
    "lambda_reach",
    "lambda_match_inner",
    "lambda_rank_inner",
    "reach_input_mode",
    "teacher_feature_mode",
    "student_branch_mode",
    "use_fg_mask_for_reach",
    "shuffle_teacher_pairs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is not None:
        with path.open() as handle:
            data = yaml.safe_load(handle) or {}
        return dict(data)
    data: dict[str, Any] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def load_diag(path: Path) -> tuple[dict[str, str], dict[str, str], int]:
    if not path.exists():
        return {}, {}, 0
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, {}, 0
    return rows[0], rows[-1], len(rows)


def repo_root_from_run(run_dir: Path) -> Path:
    marker = "runs_public"
    parts = run_dir.resolve().parts
    if marker in parts:
        return Path(*parts[: parts.index(marker)])
    return Path.cwd()


def parse_command_text(text: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    normalized = text.replace("\\\n", " ")
    try:
        tokens = shlex.split(normalized, comments=True)
    except ValueError:
        tokens = normalized.split()
    if "train_ladd_hbb.py" not in " ".join(tokens):
        return args
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if "=" in key:
            key, value = key.split("=", 1)
            args[key] = value
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[key] = tokens[i + 1]
            i += 2
        else:
            args[key] = True
            i += 1
    return args


def load_command_args(run_dir: Path) -> tuple[dict[str, Any], str]:
    repo_root = repo_root_from_run(run_dir)
    run_name = run_dir.name
    candidates: list[Path] = []
    for root_name in ("logs", "debug"):
        root = repo_root / root_name
        if root.exists():
            candidates.extend(sorted(root.rglob(f"*{run_name}*.cmd.sh")))
    if not candidates:
        return {}, ""
    path = candidates[0]
    return parse_command_text(path.read_text(errors="ignore")), str(path)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_present(*sources: dict[str, Any], key: str, default: Any = None) -> Any:
    for source in sources:
        if key in source and source[key] not in ("", None):
            return source[key]
    return default


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_args = load_args(run_dir / "args.yaml")
    command_args, command_source = load_command_args(run_dir)
    first_diag, last_diag, diag_rows = load_diag(run_dir / "ladd_diagnostics.csv")
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "args_yaml": str(run_dir / "args.yaml"),
        "command_source": command_source,
        "ladd_diagnostics_csv": str(run_dir / "ladd_diagnostics.csv"),
        "diagnostic_rows": diag_rows,
    }
    for key in SUMMARY_KEYS:
        summary[key] = first_present(last_diag, command_args, run_args, first_diag, key=key)

    rank_cap = as_float(summary.get("rank_d_neg_cap"), 4.0)
    normalize = as_bool(summary.get("normalize_reach", True))
    summary["capR_effectively_enabled_computed"] = bool(normalize and rank_cap < 4.0)
    summary["capR_effectively_disabled_reason"] = (
        ""
        if summary["capR_effectively_enabled_computed"]
        else "normalize_reach is false or rank_d_neg_cap >= 4.0"
    )
    for key in (
        "cap_saturation_ratio",
        "rank_active_ratio",
        "cap_blocked_active_ratio",
        "zero_loss_feasible_ratio",
        "kd_reach_active_ratio",
        "kd_reach_weight_mean",
        "kd_reach_weight_std",
        "gap_capped_mean",
        "reachable_margin_mean",
    ):
        summary[f"last_{key}"] = first_present(last_diag, key=key)

    output = args.output or (run_dir / "run_args_capr_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
