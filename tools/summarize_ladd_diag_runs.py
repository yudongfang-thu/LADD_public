#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - fallback keeps the script usable in minimal envs.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DOC = REPO_ROOT / "docs" / "experiments" / "BASELINE_LADD_STATUS_CN.md"

# Fallback values must be checked against BASELINE_LADD_STATUS_CN.md when that document changes.
FALLBACK_BASELINES = {
    ("n", 0): 0.55654,
    ("n", 42): 0.55794,
    ("n", 123): 0.56128,
    ("s", 0): 0.62897,
    ("m", 0): 0.65580,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LADD H1 diagnostic runs.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Run directories containing results.csv.")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if yaml is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def load_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def parse_baseline_doc(path: Path = BASELINE_DOC) -> dict[tuple[str, int], float]:
    baselines = dict(FALLBACK_BASELINES)
    if not path.is_file():
        return baselines
    row_re = re.compile(r"^\|\s*YOLO11([nslmx])\s*\|\s*(\d+)\s*\|[^|]*\|\s*([0-9.]+)@")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = row_re.match(line)
        if match:
            baselines[(match.group(1), int(match.group(2)))] = float(match.group(3))
    return baselines


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_column(fieldnames: list[str], candidates: list[str], *, exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    lowered = {name: name.lower().replace(" ", "") for name in fieldnames}
    for candidate in candidates:
        needle = candidate.lower().replace(" ", "")
        for name, normalized in lowered.items():
            if needle in normalized and all(term not in normalized for term in exclude):
                return name
    return None


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def infer_model_size(run_dir: Path, args: dict[str, Any]) -> str:
    for key in ("model_size", "size"):
        value = args.get(key)
        if value in {"n", "s", "m", "l", "x"}:
            return str(value)
    text = f"{run_dir.name} {args.get('model', '')} {args.get('name', '')}"
    match = re.search(r"yolo11([nslmx])|ogsod11([nslmx])|_([nslmx])_", text)
    if match:
        return next(group for group in match.groups() if group)
    return ""


def infer_seed(run_dir: Path, args: dict[str, Any]) -> int | None:
    if "seed" in args:
        try:
            return int(args["seed"])
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?:_s|seed)(\d+)(?:_|$)", run_dir.name)
    return int(match.group(1)) if match else None


def find_manifest(run_dir: Path) -> dict[str, str]:
    for candidate in (run_dir / "manifest.txt", run_dir.parent / "manifest.txt"):
        data = load_key_value_file(candidate)
        if data:
            return data
    return {}


def summarize_run(run_dir: Path, baselines: dict[tuple[str, int], float]) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    args_data = load_yaml(run_dir / "args.yaml")
    manifest = find_manifest(run_dir)
    rows = read_results(run_dir / "results.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    map95_col = find_column(fieldnames, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"])
    map50_col = find_column(fieldnames, ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"], exclude=["50-95"])
    epoch_col = find_column(fieldnames, ["epoch"])

    model_size = infer_model_size(run_dir, {**args_data, **manifest})
    seed = infer_seed(run_dir, {**args_data, **manifest})
    baseline = baselines.get((model_size, seed)) if seed is not None else None
    notes: list[str] = []
    if baseline is None:
        notes.append("baseline not found")

    valid_rows = [row for row in rows if map95_col and math.isfinite(as_float(row.get(map95_col)))]
    best_row = max(valid_rows, key=lambda row: as_float(row.get(map95_col)), default={})
    last_row = valid_rows[-1] if valid_rows else {}
    best_map95 = as_float(best_row.get(map95_col)) if best_row else float("nan")
    last_map95 = as_float(last_row.get(map95_col)) if last_row else float("nan")
    best_map50 = as_float(best_row.get(map50_col)) if best_row and map50_col else float("nan")
    last_map50 = as_float(last_row.get(map50_col)) if last_row and map50_col else float("nan")
    best_epoch = best_row.get(epoch_col, "") if epoch_col and best_row else ""
    expected_epochs = int(as_float(args_data.get("epochs"))) if math.isfinite(as_float(args_data.get("epochs"))) else None
    epochs_finished = len(valid_rows)

    best_gain = best_map95 - baseline if baseline is not None and math.isfinite(best_map95) else float("nan")
    last_gain = last_map95 - baseline if baseline is not None and math.isfinite(last_map95) else float("nan")
    has_nonfinite = any(not math.isfinite(as_float(row.get(map95_col))) for row in rows) if map95_col else False
    if not rows or not valid_rows or (expected_epochs is not None and epochs_finished < expected_epochs):
        status = "RUNNING/INCOMPLETE"
    elif has_nonfinite:
        status = "FAIL"
        notes.append("NaN/Inf detected")
    elif baseline is None or best_gain <= 0:
        status = "FAIL"
    elif last_gain < -0.002:
        status = "WEAK"
    else:
        status = "PASS"

    return {
        "run_name": args_data.get("name", run_dir.name),
        "git_commit": args_data.get("git_commit", manifest.get("git_commit", "")),
        "server_tag": args_data.get("server_tag", manifest.get("server_tag", "")),
        "model_size": model_size,
        "seed": "" if seed is None else seed,
        "phase": args_data.get("phase", manifest.get("phase", "")),
        "run_type": manifest.get("chain", ""),
        "epochs_finished": epochs_finished,
        "best_epoch": best_epoch,
        "best_mAP50_95": best_map95,
        "last_mAP50_95": last_map95,
        "best_mAP50": best_map50,
        "last_mAP50": last_map50,
        "baseline_mAP50_95": baseline if baseline is not None else float("nan"),
        "best_gain_vs_baseline": best_gain,
        "last_gain_vs_baseline": last_gain,
        "status": status,
        "notes": "; ".join(notes),
    }


FIELDS = [
    "run_name",
    "git_commit",
    "server_tag",
    "model_size",
    "seed",
    "phase",
    "run_type",
    "epochs_finished",
    "best_epoch",
    "best_mAP50_95",
    "last_mAP50_95",
    "best_mAP50",
    "last_mAP50",
    "baseline_mAP50_95",
    "best_gain_vs_baseline",
    "last_gain_vs_baseline",
    "status",
    "notes",
]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.5f}"
    return str(value)


def write_outputs(rows: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: fmt(row.get(field, "")) for field in FIELDS} for row in rows)

    md_lines = [
        "# LADD H1 主线诊断结果汇总",
        "",
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join(["---"] * len(FIELDS)) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in FIELDS) + " |")
    md_lines.append("")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baselines = parse_baseline_doc()
    summaries = [summarize_run(path, baselines) for path in args.runs]
    write_outputs(summaries, args.out_csv, args.out_md)


if __name__ == "__main__":
    main()
