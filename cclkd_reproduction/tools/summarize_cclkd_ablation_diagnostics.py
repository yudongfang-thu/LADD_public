#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = (
    REPO_ROOT
    / "cclkd_reproduction"
    / "experiment_versions"
    / "baseline_reference"
    / "sar_yolo11n_400ep_laddproto_results.csv"
)
VARIANTS = ("lld", "lld_fld", "lld_fld_rld", "ccl_only", "atkd", "full")
MILESTONES = (150, 200, 250, 300, 350, 400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CCLKD ablation diagnostics.")
    parser.add_argument("root", type=Path, help="CCLKD version directory or paper_ablation run root.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-csv", type=Path, default=Path("summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("summary.md"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_map(row: dict[str, Any]) -> dict[str, str]:
    return {str(k).strip(): str(k) for k in row.keys()}


def get(row: dict[str, Any], *keys: str) -> str:
    mapping = key_map(row)
    for key in keys:
        actual = mapping.get(key)
        if actual is not None and row.get(actual, "") != "":
            return str(row[actual])
    return ""


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def epoch(row: dict[str, Any]) -> int | None:
    value = to_float(get(row, "epoch"))
    return int(value) if math.isfinite(value) else None


def ap(row: dict[str, Any]) -> float:
    return to_float(get(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"))


def ap50(row: dict[str, Any]) -> float:
    return to_float(get(row, "metrics/mAP50(B)", "metrics/mAP50"))


def exact_epoch(rows: list[dict[str, str]], target: int) -> dict[str, str] | None:
    for row in rows:
        if epoch(row) == target:
            return row
    return None


def latest(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[-1] if rows else {}


def best_by_ap(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: -1.0 if math.isnan(ap(row)) else ap(row), default={})


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def infer_variant(path: Path) -> str:
    text = path.parent.name if path.name == "results.csv" else path.name
    for variant in sorted(VARIANTS, key=len, reverse=True):
        if re.search(rf"(^|_)({re.escape(variant)})(_|$)", text):
            return variant
    for variant in VARIANTS:
        if variant in text:
            return variant
    return path.parent.name


def find_result_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    files = list(root.rglob("results.csv"))
    files.extend(root.rglob("*_results.csv"))
    unique: dict[Path, Path] = {}
    for path in files:
        if "baseline_reference" in path.parts:
            continue
        unique[path.resolve()] = path
    return sorted(unique.values())


def diagnostics_path_for(results_path: Path) -> Path:
    if results_path.name == "results.csv":
        return results_path.parent / "cclkd_diagnostics.csv"
    candidate = results_path.parents[1] / "diagnostics" / results_path.name.replace("_results.csv", "_diagnostics.csv")
    if candidate.is_file():
        return candidate
    return results_path.with_name(results_path.name.replace("_results.csv", "_diagnostics.csv"))


def metric_at(rows: list[dict[str, str]], target: int, metric: str) -> float:
    row = exact_epoch(rows, target)
    if row is None:
        return math.nan
    return ap50(row) if metric == "ap50" else ap(row)


def summarize_diagnostics(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "mean_kd_to_det_ratio": mean([to_float(get(row, "kd_to_student_det_ratio")) for row in rows]),
        "mean_cop_positive_ratio": mean([to_float(get(row, "cop_positive_ratio")) for row in rows]),
        "mean_ccl_loss": mean([to_float(get(row, "cclkd_ccl_loss", "ccl_raw_or_weighted")) for row in rows]),
        "mean_rld_loss": mean([to_float(get(row, "cclkd_rld_loss", "rld_raw_or_weighted")) for row in rows]),
        "mean_temperature": mean([to_float(get(row, "temperature_mean")) for row in rows]),
    }


def fallback_component_means(results_rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "mean_ccl_loss": mean([to_float(get(row, "train/cclkd_ccl_loss")) for row in results_rows]),
        "mean_rld_loss": mean([to_float(get(row, "train/cclkd_rld_loss")) for row in results_rows]),
    }


def summarize_one(results_path: Path, baseline_rows: list[dict[str, str]]) -> dict[str, Any]:
    rows = read_csv(results_path)
    diag_path = diagnostics_path_for(results_path)
    diag_rows = read_csv(diag_path)
    variant = infer_variant(results_path)
    last = latest(rows)
    best = best_by_ap(rows)
    current_epoch = epoch(last) or 0
    baseline_same = exact_epoch(baseline_rows, current_epoch) if current_epoch else None
    diag = summarize_diagnostics(diag_rows)
    if not diag_rows:
        fallback = fallback_component_means(rows)
        diag["mean_ccl_loss"] = fallback["mean_ccl_loss"]
        diag["mean_rld_loss"] = fallback["mean_rld_loss"]
    out: dict[str, Any] = {
        "variant": variant,
        "epoch": current_epoch,
        "best_epoch": epoch(best) or "",
        "best_ap50": ap50(best),
        "best_ap": ap(best),
        "last_ap50": ap50(last),
        "last_ap": ap(last),
        "baseline_ap_same_epoch": ap(baseline_same) if baseline_same else math.nan,
        "delta_ap": ap(last) - ap(baseline_same) if baseline_same else math.nan,
        "diagnostics_path": str(diag_path) if diag_path.is_file() else "",
        "results_path": str(results_path),
        **diag,
    }
    for milestone in MILESTONES:
        out[f"ap_at_{milestone}"] = metric_at(rows, milestone, "ap")
        baseline_at = metric_at(baseline_rows, milestone, "ap")
        out[f"delta_at_{milestone}"] = out[f"ap_at_{milestone}"] - baseline_at if math.isfinite(out[f"ap_at_{milestone}"]) and math.isfinite(baseline_at) else math.nan
    return out


FIELDS = [
    "variant",
    "epoch",
    "best_epoch",
    "best_ap50",
    "best_ap",
    "last_ap50",
    "last_ap",
    "baseline_ap_same_epoch",
    "delta_ap",
    "ap_at_150",
    "ap_at_200",
    "ap_at_250",
    "ap_at_300",
    "ap_at_350",
    "ap_at_400",
    "delta_at_150",
    "delta_at_200",
    "delta_at_250",
    "delta_at_300",
    "delta_at_350",
    "delta_at_400",
    "mean_kd_to_det_ratio",
    "mean_cop_positive_ratio",
    "mean_ccl_loss",
    "mean_rld_loss",
    "mean_temperature",
    "diagnostics_path",
    "results_path",
]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def md_table(rows: list[dict[str, Any]], cols: list[str], limit: int = 50) -> str:
    if not rows:
        return "_None._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def ranking_at(rows: list[dict[str, Any]], milestone: int) -> str:
    key = f"ap_at_{milestone}"
    available = [row for row in rows if math.isfinite(to_float(row.get(key)))]
    if not available:
        return f"### Epoch {milestone}\n\n_None._\n"
    ordered = sorted(available, key=lambda row: to_float(row.get(key)), reverse=True)
    return f"### Epoch {milestone}\n\n" + md_table(ordered, ["variant", key, f"delta_at_{milestone}"])


def warning_text(rows: list[dict[str, Any]]) -> str:
    by_variant = {str(row.get("variant")): row for row in rows}
    warnings: list[str] = []
    lfr = by_variant.get("lld_fld_rld")
    lf = by_variant.get("lld_fld")
    if lfr and lf:
        matched = min(int(lfr.get("epoch") or 0), int(lf.get("epoch") or 0))
        key = next((f"ap_at_{m}" for m in reversed(MILESTONES) if m <= matched and math.isfinite(to_float(lfr.get(f"ap_at_{m}"))) and math.isfinite(to_float(lf.get(f"ap_at_{m}")))), "")
        if key and to_float(lfr.get(key)) <= to_float(lf.get(key)):
            warnings.append(f"- RLD warning: `lld_fld_rld <= lld_fld` at matched {key.replace('ap_at_', 'epoch ')}; RLD is not yet positive.")
    full = by_variant.get("full")
    atkd = by_variant.get("atkd")
    if full and atkd:
        matched = min(int(full.get("epoch") or 0), int(atkd.get("epoch") or 0))
        key = next((f"ap_at_{m}" for m in reversed(MILESTONES) if m <= matched and math.isfinite(to_float(full.get(f"ap_at_{m}"))) and math.isfinite(to_float(atkd.get(f"ap_at_{m}")))), "")
        if key and to_float(full.get(key)) - to_float(atkd.get(key)) < 0.003:
            warnings.append(f"- CCL warning: `full - atkd < 0.003 AP` at matched {key.replace('ap_at_', 'epoch ')}; CCL additive gain is weak.")
    return "\n".join(warnings) if warnings else "_No automatic warnings triggered._"


def write_outputs(rows: list[dict[str, Any]], output_csv: Path, output_md: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    latest = sorted(rows, key=lambda row: to_float(row.get("last_ap")), reverse=True)
    text = [
        "# CCLKD Ablation Diagnostics Summary",
        "",
        f"Runs summarized: {len(rows)}",
        "",
        "## Latest Ranking",
        md_table(latest, ["variant", "epoch", "last_ap", "last_ap50", "delta_ap", "best_ap", "best_epoch"]),
        "## Fixed-Epoch Ranking",
        *(ranking_at(rows, milestone) for milestone in MILESTONES),
        "## Component Loss Scale",
        md_table(sorted(rows, key=lambda row: str(row.get("variant"))), ["variant", "mean_kd_to_det_ratio", "mean_ccl_loss", "mean_rld_loss"]),
        "## COP Density",
        md_table(sorted(rows, key=lambda row: str(row.get("variant"))), ["variant", "mean_cop_positive_ratio", "mean_temperature", "diagnostics_path"]),
        "## Warnings",
        warning_text(rows),
        "",
    ]
    output_md.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_rows = read_csv(args.baseline)
    results = find_result_files(args.root)
    rows = [summarize_one(path, baseline_rows) for path in results]
    # Prefer one row per variant: if duplicates exist, keep the latest epoch.
    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = str(row.get("variant", ""))
        if variant not in by_variant or int(row.get("epoch") or 0) > int(by_variant[variant].get("epoch") or 0):
            by_variant[variant] = row
    ordered = [by_variant[v] for v in VARIANTS if v in by_variant]
    ordered.extend(row for variant, row in sorted(by_variant.items()) if variant not in VARIANTS)
    write_outputs(ordered, args.output_csv, args.output_md)
    print(f"Wrote {args.output_csv} and {args.output_md} for {len(ordered)} variants.")


if __name__ == "__main__":
    main()
