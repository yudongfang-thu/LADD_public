#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


BASELINES = {
    ("n", 0): 0.55654,
    ("n", 42): 0.55794,
    ("n", 123): 0.56128,
    ("s", 0): 0.62897,
    ("s", 42): 0.62879,
    ("s", 123): 0.62357,
    ("m", 0): 0.65580,
    ("l", 0): 0.65427,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LADD capacity-aware KD diagnostics.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories containing results.csv.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/experiments/ladd_capacity_kd_diag_20260610_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("docs/experiments/LADD_CAPACITY_KD_DIAG_RESULTS_20260610_CN.md"),
    )
    return parser.parse_args()


def as_float(value, default=float("nan")) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_model_seed_phase(run_dir: Path, rows: list[dict[str, str]]) -> tuple[str, int, str]:
    text_parts = [str(run_dir)]
    for meta_name in ("args.yaml", "manifest.txt"):
        meta_path = run_dir / meta_name
        if meta_path.is_file():
            text_parts.append(meta_path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(text_parts)
    model_match = re.search(r"yolo11([nslmx])|ogsod11([nslmx])", text)
    seed_match = re.search(r"(?:^|[_/])s(\d+)(?:[_/]|$)", text) or re.search(r"^seed:\s*(\d+)\s*$", text, re.MULTILINE)
    phase_match = re.search(r"_(a1|a2|b|c|b1|b2)_e\d+", run_dir.name)
    model_size = (model_match.group(1) or model_match.group(2)) if model_match else ""
    seed = int(seed_match.group(1)) if seed_match else -1
    phase = phase_match.group(1) if phase_match else ""
    diag = run_dir / "ladd_diagnostics.csv"
    if diag.is_file():
        diag_rows = load_csv(diag)
        if diag_rows and diag_rows[-1].get("stage"):
            phase = diag_rows[-1]["stage"]
    return model_size, seed, phase


def mean_window(rows: list[dict[str, str]], key: str, start: int, end: int) -> float:
    vals = [
        as_float(row.get(key))
        for row in rows
        if start <= int(as_float(row.get("epoch"), 0)) <= end
    ]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def pct_delta(late: float, base: float) -> float:
    if not math.isfinite(late) or not math.isfinite(base) or base == 0:
        return float("nan")
    return (late / base - 1.0) * 100.0


def fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.5f}"


def summarize_run(run_dir: Path) -> dict[str, str]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results.csv: {run_dir}")
    rows = load_csv(results_path)
    if not rows:
        raise ValueError(f"Empty results.csv: {results_path}")

    model_size, seed, phase = infer_model_seed_phase(run_dir, rows)
    ap_points = [
        (int(as_float(row.get("epoch"), idx + 1)), as_float(row.get("metrics/mAP50-95(B)")))
        for idx, row in enumerate(rows)
    ]
    finite_ap = [(epoch, ap) for epoch, ap in ap_points if math.isfinite(ap)]
    best_epoch, best_ap = max(finite_ap, key=lambda item: item[1])
    last_epoch, last_ap = finite_ap[-1]
    baseline = BASELINES.get((model_size, seed), float("nan"))
    best_gain = best_ap - baseline if math.isfinite(baseline) else float("nan")
    last_gain = last_ap - baseline if math.isfinite(baseline) else float("nan")
    late_start = max(1, last_epoch - 99)
    best_start = max(1, best_epoch - 20)
    best_end = min(last_epoch, best_epoch + 20)

    deltas = {}
    for metric, out_key in (
        ("train/box_loss", "best_to_late_window_train_box_delta_pct"),
        ("train/cls_loss", "best_to_late_window_train_cls_delta_pct"),
        ("train/kd_loss", "best_to_late_window_train_kd_delta_pct"),
        ("val/box_loss", "best_to_late_window_val_box_delta_pct"),
        ("val/cls_loss", "best_to_late_window_val_cls_delta_pct"),
    ):
        best_mean = mean_window(rows, metric, best_start, best_end)
        late_mean = mean_window(rows, metric, late_start, last_epoch)
        deltas[out_key] = pct_delta(late_mean, best_mean)

    diag_path = run_dir / "ladd_diagnostics.csv"
    latest_diag = {}
    if diag_path.is_file():
        diag_rows = load_csv(diag_path)
        latest_diag = diag_rows[-1] if diag_rows else {}

    nonfinite = any(
        not math.isfinite(as_float(value))
        for row in rows
        for value in row.values()
        if value not in {"", None}
    )
    if latest_diag.get("nan_or_inf_detected") not in {"", None, "0", "0.0"}:
        nonfinite = True
    if "detonly" in run_dir.name.lower() or phase in {"a1", "a2"} or last_epoch <= 1:
        status = "DIAG"
    elif nonfinite or (math.isfinite(best_gain) and best_gain <= 0):
        status = "FAIL"
    elif math.isfinite(last_gain) and last_gain >= -0.002:
        status = "PASS"
    else:
        status = "WEAK"

    kd_decay_policy = latest_diag.get("ladd_kd_decay_mode", "")
    if kd_decay_policy:
        kd_decay_policy = (
            f"{kd_decay_policy}:start={latest_diag.get('ladd_kd_decay_start_epoch', '')},"
            f"end={latest_diag.get('ladd_kd_decay_end_epoch', '')},"
            f"final={latest_diag.get('ladd_kd_final_mult', '')},"
            f"stop={latest_diag.get('ladd_kd_stop_after_epoch', '')}"
        )

    return {
        "run_name": run_dir.name,
        "model_size": model_size,
        "seed": str(seed),
        "phase": phase,
        "epochs_finished": str(len(rows)),
        "best_epoch": str(best_epoch),
        "best_mAP50_95": fmt(best_ap),
        "last_mAP50_95": fmt(last_ap),
        "baseline_mAP50_95": fmt(baseline),
        "best_gain_vs_baseline": fmt(best_gain),
        "last_gain_vs_baseline": fmt(last_gain),
        "last_minus_best": fmt(last_ap - best_ap),
        **{key: fmt(value) for key, value in deltas.items()},
        "effective_alpha_kd": latest_diag.get("effective_alpha_kd", ""),
        "kd_decay_policy": kd_decay_policy,
        "ladd_b_det_only": latest_diag.get("ladd_b_det_only", "0"),
        "ladd_a2_det_only": latest_diag.get("ladd_a2_det_only", "0"),
        "status": status,
        "notes": "",
    }


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LADD Capacity-aware KD 诊断结果",
        "",
        "日期：2026-06-10",
        "",
        "本文件由 `tools/summarize_ladd_capacity_diag.py` 生成。Baseline 字典需与 `docs/experiments/BASELINE_LADD_STATUS_CN.md` 核对。",
        "",
        "| run | model | seed | phase | best | last | baseline | best gain | last gain | status |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_name']}` | {row['model_size']} | {row['seed']} | {row['phase']} | "
            f"{row['best_mAP50_95']}@{row['best_epoch']} | {row['last_mAP50_95']} | "
            f"{row['baseline_mAP50_95']} | {row['best_gain_vs_baseline']} | "
            f"{row['last_gain_vs_baseline']} | {row['status']} |"
        )
    lines += [
        "",
        "状态规则：`PASS` 表示 best_gain > 0 且 last_gain >= -0.002；`WEAK` 表示 best 正向但 last 低于阈值；`FAIL` 表示 best 不正向或出现 NaN/Inf；`DIAG` 表示 det-only / A2-only / B=1 probe 等诊断项。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [summarize_run(path) for path in args.run_dirs]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.out_md, rows)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
