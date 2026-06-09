#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LADD LR/BN diagnostic runs.")
    parser.add_argument("roots", nargs="+", type=Path, help="Run root directories to scan recursively.")
    parser.add_argument("--output-csv", type=Path, default=Path("summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("summary.md"))
    return parser.parse_args()


def clean_key_map(row: dict[str, Any]) -> dict[str, str]:
    return {k.strip(): k for k in row.keys()}


def get_value(row: dict[str, Any], key: str, default: str = "") -> str:
    keys = clean_key_map(row)
    return str(row.get(keys.get(key, key), default))


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_args_yaml(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("'\"")
    return parsed


def infer_model_size(run_dir: Path, args: dict[str, str]) -> str:
    for text in (run_dir.name, str(run_dir)):
        match = re.search(r"yolo11([nslmx])", text)
        if match:
            return match.group(1)
    model = args.get("model", "")
    match = re.search(r"yolo11([nslmx])", model)
    return match.group(1) if match else ""


def infer_seed(run_dir: Path, args: dict[str, str]) -> str:
    if args.get("seed") not in {None, ""}:
        return str(args.get("seed"))
    match = re.search(r"_s(\d+)(?:_|$)", run_dir.name)
    return match.group(1) if match else ""


def infer_variant(run_dir: Path) -> str:
    parts = set(run_dir.parts)
    if "cap2" in parts or "_cap2_" in run_dir.name:
        return "cap2"
    if "original" in parts or "_original_" in run_dir.name:
        return "original"
    return ""


def boolish(value: str) -> str:
    if value == "":
        return ""
    if value.lower() in {"true", "1", "yes"}:
        return "1"
    if value.lower() in {"false", "0", "no"}:
        return "0"
    return value


def value_is_nonfinite(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return True
    number = to_float(value)
    return not math.isnan(number) and not math.isfinite(number)


def find_results(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    results: list[Path] = []
    for root in roots:
        for path in root.expanduser().resolve().rglob("results.csv"):
            if path not in seen:
                seen.add(path)
                results.append(path)
    return sorted(results)


def summarize_one(results_path: Path) -> dict[str, Any]:
    run_dir = results_path.parent
    args_path = run_dir / "args.yaml"
    diagnostics_path = run_dir / "ladd_diagnostics.csv"
    rows = read_csv(results_path)
    args = read_args_yaml(args_path)
    diag_rows = read_csv(diagnostics_path)

    ap_key = "metrics/mAP50-95(B)"
    ap50_key = "metrics/mAP50(B)"
    if rows and math.isnan(to_float(get_value(rows[-1], ap_key))):
        ap_key = "metrics/mAP50-95"
    if rows and math.isnan(to_float(get_value(rows[-1], ap50_key))):
        ap50_key = "metrics/mAP50"

    best = max(rows, key=lambda r: to_float(get_value(r, ap_key)), default={})
    last = rows[-1] if rows else {}
    best_ap = to_float(get_value(best, ap_key))
    last_ap = to_float(get_value(last, ap_key))
    best_ap50 = to_float(get_value(best, ap50_key))
    last_ap50 = to_float(get_value(last, ap50_key))
    best_epoch = to_int(get_value(best, "epoch"))
    last_epoch = to_int(get_value(last, "epoch"))

    bn_var_max = math.nan
    bn_p95_max = math.nan
    nan_any = False
    bn_mode = ""
    for row in diag_rows:
        bn_var_max = max(bn_var_max, to_float(get_value(row, "bn_running_var_max"))) if not math.isnan(bn_var_max) else to_float(get_value(row, "bn_running_var_max"))
        bn_p95_max = max(bn_p95_max, to_float(get_value(row, "bn_running_var_p95"))) if not math.isnan(bn_p95_max) else to_float(get_value(row, "bn_running_var_p95"))
        nan_any = nan_any or bool(to_int(get_value(row, "nan_or_inf_detected")) or False)
        if get_value(row, "bn_stats_mode"):
            bn_mode = get_value(row, "bn_stats_mode")

    for row in rows:
        for value in row.values():
            if value_is_nonfinite(value):
                nan_any = True

    freeze_bn_after = args.get("freeze_bn_after_epoch", args.get("b_freeze_bn_after_epoch", ""))
    freeze_bn_stats = boolish(args.get("freeze_bn_stats", ""))

    return {
        "run_dir": str(run_dir),
        "model_size": infer_model_size(run_dir, args),
        "seed": infer_seed(run_dir, args),
        "variant": infer_variant(run_dir),
        "run_tag": run_dir.name,
        "epochs_recorded": len(rows),
        "best_epoch_by_ap": best_epoch if best_epoch is not None else "",
        "best_ap50": best_ap50,
        "best_ap": best_ap,
        "last_epoch": last_epoch if last_epoch is not None else "",
        "last_ap50": last_ap50,
        "last_ap": last_ap,
        "drop_best_to_last": best_ap - last_ap if math.isfinite(best_ap) and math.isfinite(last_ap) else math.nan,
        "lr0": args.get("lr0", ""),
        "lrf": args.get("lrf", ""),
        "cos_lr": boolish(args.get("cos_lr", "")),
        "freeze_bn_stats": freeze_bn_stats,
        "freeze_bn_after_epoch": freeze_bn_after,
        "bn_stats_mode": bn_mode,
        "bn_running_var_max_global": bn_var_max,
        "bn_running_var_p95_max": bn_p95_max,
        "nan_or_inf_any": int(nan_any),
        "args_path": str(args_path) if args_path.is_file() else "",
        "results_path": str(results_path),
        "diagnostics_path": str(diagnostics_path) if diagnostics_path.is_file() else "",
    }


FIELDS = [
    "run_dir",
    "model_size",
    "seed",
    "variant",
    "run_tag",
    "epochs_recorded",
    "best_epoch_by_ap",
    "best_ap50",
    "best_ap",
    "last_epoch",
    "last_ap50",
    "last_ap",
    "drop_best_to_last",
    "lr0",
    "lrf",
    "cos_lr",
    "freeze_bn_stats",
    "freeze_bn_after_epoch",
    "bn_stats_mode",
    "bn_running_var_max_global",
    "bn_running_var_p95_max",
    "nan_or_inf_any",
    "args_path",
    "results_path",
    "diagnostics_path",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "_None._\n"
    rows = rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group_rows = []
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model_size", "")),
            str(row.get("lr0", "")),
            str(row.get("lrf", "")),
            str(row.get("cos_lr", "")),
            str(row.get("freeze_bn_stats", "")),
            str(row.get("freeze_bn_after_epoch", "")),
        )
        groups[key].append(row)
    for key, items in groups.items():
        group_rows.append(
            {
                "model_size": key[0],
                "lr0": key[1],
                "lrf": key[2],
                "cos_lr": key[3],
                "freeze_bn_stats": key[4],
                "freeze_bn_after_epoch": key[5],
                "n": len(items),
                "mean_best_ap": mean([to_float(r.get("best_ap")) for r in items]),
                "mean_last_ap": mean([to_float(r.get("last_ap")) for r in items]),
                "mean_drop": mean([to_float(r.get("drop_best_to_last")) for r in items]),
            }
        )

    cols = ["model_size", "seed", "run_tag", "best_epoch_by_ap", "best_ap", "last_ap", "drop_best_to_last", "bn_running_var_max_global"]
    text = [
        "# LADD LR / BN Diagnostic Summary",
        "",
        f"Runs summarized: {len(rows)}",
        "",
        "## Best AP Top 20",
        md_table(sorted(rows, key=lambda r: to_float(r.get("best_ap")), reverse=True), cols),
        "## Largest Best-to-Last Drop Top 20",
        md_table(sorted(rows, key=lambda r: to_float(r.get("drop_best_to_last")), reverse=True), cols),
        "## Largest BN Running Var Top 20",
        md_table(sorted(rows, key=lambda r: to_float(r.get("bn_running_var_max_global")), reverse=True), cols),
        "## NaN / Inf Runs",
        md_table([r for r in rows if str(r.get("nan_or_inf_any")) == "1"], cols),
        "## Configuration Group Means",
        md_table(
            sorted(group_rows, key=lambda r: to_float(r.get("mean_best_ap")), reverse=True),
            ["model_size", "lr0", "lrf", "cos_lr", "freeze_bn_stats", "freeze_bn_after_epoch", "n", "mean_best_ap", "mean_last_ap", "mean_drop"],
            limit=200,
        ),
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [summarize_one(path) for path in find_results(args.roots)]
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {args.output_csv} and {args.output_md} for {len(rows)} runs.")


if __name__ == "__main__":
    main()
