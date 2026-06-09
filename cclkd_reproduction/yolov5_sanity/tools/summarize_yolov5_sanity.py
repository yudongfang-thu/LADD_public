#!/usr/bin/env python3
"""Summarize YOLOv5 sanity baseline runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

TARGET_AP50 = 80.9
TARGET_AP = 46.3
PASS_AP50 = 78.0
PASS_AP = 44.0
FIELDS = [
    "run_dir",
    "tag",
    "modality",
    "model",
    "init",
    "batch",
    "seed",
    "yolov5_ref",
    "params_m",
    "epochs_recorded",
    "best_epoch",
    "best_ap50",
    "best_ap",
    "last_epoch",
    "last_ap50",
    "last_ap",
    "drop_best_to_last",
    "oil_tank_ap50",
    "bridge_ap50",
    "harbor_ap50",
    "class_name_order",
    "target_ap50",
    "target_ap",
    "delta_to_target_ap50",
    "delta_to_target_ap",
    "pass_baseline_threshold",
    "results_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs_dir",
        nargs="?",
        default="cclkd_reproduction/yolov5_sanity/results/runs",
        help="YOLOv5 project runs directory",
    )
    parser.add_argument(
        "--output-dir",
        default="cclkd_reproduction/yolov5_sanity/results",
        help="Directory for summary.csv and summary.md",
    )
    return parser.parse_args()


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def parse_run_name(name: str) -> dict[str, str]:
    pattern = re.compile(
        r"^yolov5_(?P<modality>sar|rgb)_(?P<model>x6|x)_(?P<init>pretrained|scratch)_b(?P<batch>\d+)_s(?P<seed>\d+)_(?P<tag>.+)$"
    )
    match = pattern.match(name)
    return match.groupdict() if match else {}


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {k.strip(): v.strip() for k, v in row.items() if k is not None}


def find_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    simplified = {name.replace(" ", "").lower(): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.replace(" ", "").lower()
        if key in simplified:
            return simplified[key]
    for name in fieldnames:
        low = name.replace(" ", "").lower()
        if any(candidate.replace(" ", "").lower() in low for candidate in candidates):
            return name
    return None


def to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def metric_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if 0.0 <= value <= 1.5 else value


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4g}"


def summarize_run(run_dir: Path) -> dict[str, str]:
    meta = read_key_values(run_dir / "run_meta.txt")
    parsed = parse_run_name(run_dir.name)
    opt = load_yaml(run_dir / "opt.yaml")
    hyp = load_yaml(run_dir / "hyp.yaml")
    results_path = run_dir / "results.csv"

    row: dict[str, str] = {field: "" for field in FIELDS}
    row["run_dir"] = str(run_dir)
    for key in ("tag", "modality", "model", "init", "batch", "seed"):
        row[key] = meta.get(key) or parsed.get(key) or str(opt.get(key, ""))
    row["yolov5_ref"] = meta.get("yolov5_ref", "")
    row["params_m"] = meta.get("params_m", "")
    row["class_name_order"] = meta.get("class_name_order", "")
    row["target_ap50"] = str(TARGET_AP50)
    row["target_ap"] = str(TARGET_AP)
    row["results_path"] = str(results_path) if results_path.exists() else ""
    for cls_key in ("oil_tank_ap50", "bridge_ap50", "harbor_ap50"):
        row[cls_key] = meta.get(cls_key, "")

    if not results_path.exists():
        row["pass_baseline_threshold"] = "missing_results"
        return row

    with results_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = [normalize_row(r) for r in reader]
        fieldnames = [x.strip() for x in (reader.fieldnames or [])]

    if not rows:
        row["pass_baseline_threshold"] = "empty_results"
        return row

    epoch_col = find_col(fieldnames, ["epoch"])
    ap50_col = find_col(fieldnames, ["metrics/mAP_0.5", "map50", "mAP@0.5"])
    ap_col = find_col(fieldnames, ["metrics/mAP_0.5:0.95", "map50-95", "mAP@0.5:0.95"])

    scored: list[tuple[float, int, dict[str, str]]] = []
    for idx, item in enumerate(rows):
        ap = to_float(item.get(ap_col or ""))
        ap50 = to_float(item.get(ap50_col or ""))
        score = ap if ap is not None else ap50 if ap50 is not None else float("-inf")
        scored.append((score, idx, item))

    best = max(scored, key=lambda x: x[0])[2]
    last = rows[-1]
    best_ap50 = metric_percent(to_float(best.get(ap50_col or "")))
    best_ap = metric_percent(to_float(best.get(ap_col or "")))
    last_ap50 = metric_percent(to_float(last.get(ap50_col or "")))
    last_ap = metric_percent(to_float(last.get(ap_col or "")))
    best_epoch = best.get(epoch_col or "", "")
    last_epoch = last.get(epoch_col or "", "")
    drop = best_ap - last_ap if best_ap is not None and last_ap is not None else None

    row["epochs_recorded"] = str(len(rows))
    row["best_epoch"] = best_epoch
    row["best_ap50"] = fmt(best_ap50)
    row["best_ap"] = fmt(best_ap)
    row["last_epoch"] = last_epoch
    row["last_ap50"] = fmt(last_ap50)
    row["last_ap"] = fmt(last_ap)
    row["drop_best_to_last"] = fmt(drop)
    row["delta_to_target_ap50"] = fmt(best_ap50 - TARGET_AP50 if best_ap50 is not None else None)
    row["delta_to_target_ap"] = fmt(best_ap - TARGET_AP if best_ap is not None else None)
    row["pass_baseline_threshold"] = str(
        best_ap50 is not None and best_ap is not None and best_ap50 >= PASS_AP50 and best_ap >= PASS_AP
    )
    if hyp and not row["tag"]:
        row["tag"] = str(hyp.get("name", ""))
    return row


def markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# YOLOv5 Sanity Summary",
        "",
        "## Baseline Target Reminder",
        "",
        "- YOLOv5 CSPDarkNet-X / YOLOv5x target: 86.23M params, AP50 80.9, AP 46.3.",
        "- Loose pass threshold: AP50 >= 78 and AP >= 44.",
        "",
        "## Best AP Ranking",
        "",
        "| rank | run | best AP50 | best AP | pass |",
        "|---:|---|---:|---:|---|",
    ]

    def rank_key(item: dict[str, str]) -> float:
        return to_float(item.get("best_ap")) or -1.0

    for rank, row in enumerate(sorted(rows, key=rank_key, reverse=True), start=1):
        lines.append(
            f"| {rank} | `{Path(row['run_dir']).name}` | {row['best_ap50']} | {row['best_ap']} | {row['pass_baseline_threshold']} |"
        )

    lines.extend([
        "",
        "## Comparison Against Target",
        "",
        "| run | delta AP50 | delta AP | params M | class order |",
        "|---|---:|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            f"| `{Path(row['run_dir']).name}` | {row['delta_to_target_ap50']} | {row['delta_to_target_ap']} | {row['params_m']} | `{row['class_name_order']}` |"
        )

    lines.extend([
        "",
        "## Per-Class AP50",
        "",
        "| run | Oil Tank | Bridge | Harbor |",
        "|---|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| `{Path(row['run_dir']).name}` | {row['oil_tank_ap50']} | {row['bridge_ap50']} | {row['harbor_ap50']} |"
        )

    lines.extend(["", "## Failure Diagnosis Hints", ""])
    any_low = any(row.get("pass_baseline_threshold") == "False" for row in rows)
    if any_low:
        lines.append(
            "- Do not proceed to CCLKD/CMDistill reproduction. Check data split, class mapping, anchors, augmentation, YOLOv5 version, and evaluation protocol first."
        )
    pretrained = [r for r in rows if r.get("init") == "pretrained" and to_float(r.get("best_ap")) is not None]
    scratch = [r for r in rows if r.get("init") == "scratch" and to_float(r.get("best_ap")) is not None]
    if pretrained and scratch and max(to_float(r["best_ap"]) or -1 for r in pretrained) > max(to_float(r["best_ap"]) or -1 for r in scratch) + 2:
        lines.append("- Pretrained is much stronger than scratch; the paper likely used or benefited from pretrained initialization.")
    sar = [r for r in rows if r.get("modality") == "sar" and to_float(r.get("best_ap")) is not None]
    rgb = [r for r in rows if r.get("modality") == "rgb" and to_float(r.get("best_ap")) is not None]
    if sar and rgb and max(to_float(r["best_ap"]) or -1 for r in rgb) <= max(to_float(r["best_ap"]) or -1 for r in sar) + 2:
        lines.append("- RGB teacher is only slightly above SAR; teacher-student gap may be too small for large KD gains.")
    lines.append("- If class-specific AP is mismatched, especially Oil Tank, check class mapping and small-object distribution.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(p for p in runs_dir.glob("*") if p.is_dir()) if runs_dir.exists() else []
    rows = [summarize_run(run_dir) for run_dir in run_dirs]

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    md_path.write_text(markdown(rows), encoding="utf-8")
    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")
    print(f"runs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
