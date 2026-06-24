#!/usr/bin/env python3
"""Monitor LADD capR/gated-KD runs against a same-pipeline det-only control."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline run dir or results.csv.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Candidate as name=path or path. Path may be run dir or results.csv. Repeatable.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def result_path(path: Path) -> Path:
    return path if path.name == "results.csv" else path / "results.csv"


def run_dir(path: Path) -> Path:
    return path.parent if path.name == "results.csv" else path


def metric(row: dict[str, str], *names: str) -> float:
    stripped = {key.strip(): value for key, value in row.items()}
    for name in names:
        value = stripped.get(name)
        if value not in (None, ""):
            return float(value)
    return float("nan")


def read_results(path: Path) -> list[dict[str, float]]:
    csv_path = result_path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for idx, row in enumerate(rows, 1):
        out.append(
            {
                "epoch": float(metric(row, "epoch") if row.get("epoch") else idx),
                "ap50": metric(row, "metrics/mAP50(B)", "metrics/mAP50"),
                "ap5095": metric(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
            }
        )
    return out


def read_last_diag(path: Path) -> dict[str, str]:
    diag = run_dir(path) / "ladd_diagnostics.csv"
    if not diag.exists():
        return {}
    with diag.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def late(rows: list[dict[str, float]], count: int = 20) -> float:
    return mean([row["ap5095"] for row in rows[-min(count, len(rows)) :]])


def best(rows: list[dict[str, float]]) -> tuple[float, int]:
    if not rows:
        return float("nan"), 0
    idx, row = max(enumerate(rows, 1), key=lambda item: item[1]["ap5095"])
    return row["ap5095"], idx


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.5f}"
    return str(value)


def parse_run_item(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name, Path(path)
    path = Path(text)
    return run_dir(path).name, path


def summarize_candidate(name: str, path: Path, baseline_rows: list[dict[str, float]]) -> dict[str, Any]:
    rows = read_results(path)
    diag = read_last_diag(path)
    matched = min(len(rows), len(baseline_rows))
    latest = rows[-1]["ap5095"] if rows else float("nan")
    latest_base = baseline_rows[matched - 1]["ap5095"] if matched else float("nan")
    best_value, best_epoch = best(rows)
    late20_value = late(rows, 20)
    baseline_late20 = late(baseline_rows[:matched], 20) if matched else float("nan")
    latest_delta = latest - latest_base if matched else float("nan")
    late20_delta = late20_value - baseline_late20 if matched else float("nan")
    status = "missing"
    if rows:
        if matched < 100:
            status = "pre100"
        elif late20_delta >= 0.020 and latest_delta > 0:
            status = "STRONG_EARLY"
        elif late20_delta >= 0.010 and latest_delta > 0:
            status = "PROMISING_EARLY"
        elif matched >= 120 and late20_delta <= 0:
            status = "LOW_PRIORITY"
        else:
            status = "WATCH"
    return {
        "run_name": name,
        "rows": len(rows),
        "matched_rows": matched,
        "latest_map5095": latest,
        "best_map5095": best_value,
        "best_epoch": best_epoch,
        "late20_map5095": late20_value,
        "latest_delta": latest_delta,
        "late20_delta": late20_delta,
        "capR_enabled": diag.get("capR_effectively_enabled", ""),
        "cap_saturation_ratio": diag.get("cap_saturation_ratio", ""),
        "rank_active_ratio": diag.get("rank_active_ratio", ""),
        "kd_active_ratio": diag.get("kd_reach_active_ratio", ""),
        "status": status,
        "notes": diag.get("kd_weight_mode", ""),
        "path": str(run_dir(path)),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no runs)")
        return
    columns = [
        "run_name",
        "rows",
        "latest_map5095",
        "best_map5095",
        "late20_map5095",
        "latest_delta",
        "late20_delta",
        "capR_enabled",
        "cap_saturation_ratio",
        "rank_active_ratio",
        "kd_active_ratio",
        "status",
        "notes",
    ]
    widths = {col: max(len(col), *(len(fmt(row.get(col, ""))) for row in rows)) for col in columns}
    print(" | ".join(col.ljust(widths[col]) for col in columns))
    print("-+-".join("-" * widths[col] for col in columns))
    for row in rows:
        print(" | ".join(fmt(row.get(col, "")).ljust(widths[col]) for col in columns))


def main() -> None:
    args = parse_args()
    baseline_rows = read_results(args.baseline)
    if not baseline_rows:
        raise FileNotFoundError(f"Missing or empty baseline results: {result_path(args.baseline)}")
    rows = [summarize_candidate(name, path, baseline_rows) for name, path in map(parse_run_item, args.run)]
    print_table(rows)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
