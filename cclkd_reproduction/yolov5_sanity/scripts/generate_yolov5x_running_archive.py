#!/usr/bin/env python3
"""Generate compact YOLOv5x CCLKD running archive tables.

The script reads archived compact CSV files under:

  <archive>/runs/<run_key>/results.csv
  <archive>/runs/<run_key>/cclkd_yolov5_diagnostics.csv

and writes:

  summary.csv
  loss_contribution_latest.csv
  loss_contribution_latest.md
  milestone_component_comparison.csv
  milestone_component_comparison.md
  running_status.md

It intentionally uses exact same-epoch comparisons only. Missing milestone rows
are written as "pending".
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ARCHIVE = Path(
    "cclkd_reproduction/yolov5_sanity/results/"
    "scalingfix_paper_components_400ep_20260613"
)
DEFAULT_MILESTONES = [80, 100, 125, 150, 185, 200, 250, 300, 350, 399]

RUNS = {
    "paper_full": {
        "dir": "paper_full",
        "label": "Full CCLKD",
        "mode": "paper_full",
    },
    "paper_atkd_only": {
        "dir": "paper_atkd_only",
        "label": "ATKD-only",
        "mode": "paper_atkd_only",
    },
    "paper_ccl_only": {
        "dir": "paper_ccl_only",
        "label": "CCL-only",
        "mode": "paper_ccl_only",
    },
    "det_only_same_trainer": {
        "dir": "det_only_same_trainer",
        "label": "Det-only baseline",
        "mode": "det_only_same_trainer",
    },
}

SUMMARY_FIELDS = [
    "run_key",
    "mode",
    "latest_epoch",
    "AP50",
    "AP",
    "precision",
    "recall",
    "detonly_same_epoch_ap50",
    "detonly_same_epoch_ap",
    "delta_ap50_vs_detonly_same_epoch",
    "delta_ap_vs_detonly_same_epoch",
    "student_box_loss",
    "student_obj_loss",
    "student_cls_loss",
    "teacher_box_loss",
    "teacher_obj_loss",
    "teacher_cls_loss",
    "kd_total_loss",
    "lld_loss",
    "fld_loss",
    "rld_loss",
    "ccl_loss",
    "cop_positive_ratio",
    "cop_positive_candidates",
    "cop_valid_candidates",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "weighted_kd_to_student_det_ratio",
    "feature_capture_ok",
    "nan_or_inf_detected",
]

LOSS_FIELDS = [
    "run",
    "epoch",
    "AP",
    "det_same_AP",
    "delta_AP",
    "student_box",
    "student_obj",
    "student_cls",
    "student_det_sum",
    "teacher_det_sum",
    "lld",
    "fld",
    "rld",
    "atkd",
    "ccl",
    "kd_total",
    "atkd_share_in_kd",
    "ccl_share_in_kd",
    "weighted_kd/det",
    "raw_kd/det",
    "cop_pos",
    "feature_ok",
    "nan",
]

MILESTONE_FIELDS = [
    "epoch",
    "det_only_ap50",
    "det_only_ap",
    "atkd_ap50",
    "atkd_ap",
    "atkd_delta_ap50",
    "atkd_delta_ap",
    "ccl_ap50",
    "ccl_ap",
    "ccl_delta_ap50",
    "ccl_delta_ap",
    "full_ap50",
    "full_ap",
    "full_delta_ap50",
    "full_delta_ap",
    "full_minus_atkd_ap",
    "full_minus_ccl_ap",
    "best_component_by_ap",
    "note",
]


def clean_row(row: dict[str, str]) -> dict[str, str]:
    return {
        (key.strip() if key is not None else key): (
            value.strip() if isinstance(value, str) else value
        )
        for key, value in row.items()
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [clean_row(row) for row in csv.DictReader(handle)]


def by_epoch(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        epoch = int_value(row.get("epoch", ""))
        if epoch is not None:
            out[epoch] = row
    return out


def float_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_value(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, str):
        return value
    return f"{float(value):.5f}"


def blank_fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.5f}"


def metric(row: dict[str, str] | None, key: str) -> float | None:
    if not row:
        return None
    return float_value(row.get(key, ""))


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def sum_values(*values: float | None) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid)


def load_archive(archive: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for run_key, cfg in RUNS.items():
        run_dir = archive / "runs" / cfg["dir"]
        results = read_csv(run_dir / "results.csv")
        diagnostics = read_csv(run_dir / "cclkd_yolov5_diagnostics.csv")
        data[run_key] = {
            "cfg": cfg,
            "results": results,
            "diagnostics": diagnostics,
            "results_by_epoch": by_epoch(results),
            "diagnostics_by_epoch": by_epoch(diagnostics),
            "latest_result": results[-1] if results else {},
            "latest_diag": diagnostics[-1] if diagnostics else {},
        }
    return data


def build_summary_rows(data: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    det_by_epoch = data["det_only_same_trainer"]["results_by_epoch"]
    rows: list[dict[str, str]] = []
    for run_key in ("paper_full", "paper_atkd_only", "paper_ccl_only", "det_only_same_trainer"):
        item = data[run_key]
        latest = item["latest_result"]
        diag = item["latest_diag"]
        epoch = int_value(latest.get("epoch", ""))
        det_same = det_by_epoch.get(epoch) if epoch is not None else None
        ap50 = metric(latest, "metrics/mAP_0.5")
        ap = metric(latest, "metrics/mAP_0.5:0.95")
        if run_key == "det_only_same_trainer":
            det_ap50: float | str | None = "baseline"
            det_ap: float | str | None = "baseline"
            delta_ap50 = None
            delta_ap = None
        else:
            det_ap50 = metric(det_same, "metrics/mAP_0.5")
            det_ap = metric(det_same, "metrics/mAP_0.5:0.95")
            delta_ap50 = delta(ap50, det_ap50 if isinstance(det_ap50, float) else None)
            delta_ap = delta(ap, det_ap if isinstance(det_ap, float) else None)
        row = {
            "run_key": run_key,
            "mode": item["cfg"]["mode"],
            "latest_epoch": str(epoch) if epoch is not None else "",
            "AP50": blank_fmt(ap50),
            "AP": blank_fmt(ap),
            "precision": blank_fmt(metric(latest, "metrics/precision")),
            "recall": blank_fmt(metric(latest, "metrics/recall")),
            "detonly_same_epoch_ap50": blank_fmt(det_ap50),
            "detonly_same_epoch_ap": blank_fmt(det_ap),
            "delta_ap50_vs_detonly_same_epoch": blank_fmt(delta_ap50),
            "delta_ap_vs_detonly_same_epoch": blank_fmt(delta_ap),
            "student_box_loss": blank_fmt(metric(diag, "student_box_loss")),
            "student_obj_loss": blank_fmt(metric(diag, "student_obj_loss")),
            "student_cls_loss": blank_fmt(metric(diag, "student_cls_loss")),
            "teacher_box_loss": blank_fmt(metric(diag, "teacher_box_loss")),
            "teacher_obj_loss": blank_fmt(metric(diag, "teacher_obj_loss")),
            "teacher_cls_loss": blank_fmt(metric(diag, "teacher_cls_loss")),
            "kd_total_loss": blank_fmt(metric(diag, "kd_total_loss")),
            "lld_loss": blank_fmt(metric(diag, "lld_loss")),
            "fld_loss": blank_fmt(metric(diag, "fld_loss")),
            "rld_loss": blank_fmt(metric(diag, "rld_loss")),
            "ccl_loss": blank_fmt(metric(diag, "ccl_loss")),
            "cop_positive_ratio": blank_fmt(metric(diag, "cop_positive_ratio")),
            "cop_positive_candidates": blank_fmt(metric(diag, "cop_positive_candidates")),
            "cop_valid_candidates": blank_fmt(metric(diag, "cop_valid_candidates")),
            "temperature_mean": blank_fmt(metric(diag, "temperature_mean")),
            "temperature_min": blank_fmt(metric(diag, "temperature_min")),
            "temperature_max": blank_fmt(metric(diag, "temperature_max")),
            "weighted_kd_to_student_det_ratio": blank_fmt(metric(diag, "weighted_kd_to_student_det_ratio")),
            "feature_capture_ok": blank_fmt(metric(diag, "feature_capture_ok")),
            "nan_or_inf_detected": blank_fmt(metric(diag, "nan_or_inf_detected")),
        }
        rows.append(row)
    return rows


def build_loss_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    labels = {
        "paper_full": "Full CCLKD",
        "paper_atkd_only": "ATKD-only",
        "paper_ccl_only": "CCL-only",
        "det_only_same_trainer": "Det-only baseline",
    }
    for row in summary_rows:
        run_key = row["run_key"]
        student_box = float_value(row["student_box_loss"])
        student_obj = float_value(row["student_obj_loss"])
        student_cls = float_value(row["student_cls_loss"])
        teacher_box = float_value(row["teacher_box_loss"])
        teacher_obj = float_value(row["teacher_obj_loss"])
        teacher_cls = float_value(row["teacher_cls_loss"])
        lld = float_value(row["lld_loss"])
        fld = float_value(row["fld_loss"])
        rld = float_value(row["rld_loss"])
        ccl = float_value(row["ccl_loss"])
        atkd = sum_values(lld, fld, rld)
        kd_total = float_value(row["kd_total_loss"])
        atkd_share = None
        ccl_share = None
        if kd_total and kd_total > 0:
            atkd_share = (atkd or 0.0) / kd_total
            ccl_share = (ccl or 0.0) / kd_total
        rows.append(
            {
                "run": labels[run_key],
                "epoch": row["latest_epoch"],
                "AP": row["AP"],
                "det_same_AP": row["detonly_same_epoch_ap"],
                "delta_AP": row["delta_ap_vs_detonly_same_epoch"],
                "student_box": row["student_box_loss"],
                "student_obj": row["student_obj_loss"],
                "student_cls": row["student_cls_loss"],
                "student_det_sum": blank_fmt(sum_values(student_box, student_obj, student_cls)),
                "teacher_det_sum": blank_fmt(sum_values(teacher_box, teacher_obj, teacher_cls)),
                "lld": row["lld_loss"],
                "fld": row["fld_loss"],
                "rld": row["rld_loss"],
                "atkd": blank_fmt(atkd),
                "ccl": row["ccl_loss"],
                "kd_total": row["kd_total_loss"],
                "atkd_share_in_kd": blank_fmt(atkd_share),
                "ccl_share_in_kd": blank_fmt(ccl_share),
                "weighted_kd/det": row["weighted_kd_to_student_det_ratio"],
                "raw_kd/det": row["weighted_kd_to_student_det_ratio"],
                "cop_pos": row["cop_positive_ratio"],
                "feature_ok": row["feature_capture_ok"],
                "nan": row["nan_or_inf_detected"],
            }
        )
    return rows


def best_component(atkd_ap: float | None, ccl_ap: float | None, full_ap: float | None) -> str:
    values = {"atkd": atkd_ap, "ccl": ccl_ap, "full": full_ap}
    available = {key: value for key, value in values.items() if value is not None}
    if not available:
        return "pending"
    return max(available, key=lambda key: available[key])


def build_milestone_rows(data: dict[str, dict[str, Any]], milestones: list[int]) -> list[dict[str, str]]:
    det = data["det_only_same_trainer"]["results_by_epoch"]
    atkd = data["paper_atkd_only"]["results_by_epoch"]
    ccl = data["paper_ccl_only"]["results_by_epoch"]
    full = data["paper_full"]["results_by_epoch"]
    rows: list[dict[str, str]] = []
    for epoch in milestones:
        det_row = det.get(epoch)
        atkd_row = atkd.get(epoch)
        ccl_row = ccl.get(epoch)
        full_row = full.get(epoch)
        det_ap50 = metric(det_row, "metrics/mAP_0.5")
        det_ap = metric(det_row, "metrics/mAP_0.5:0.95")
        atkd_ap50 = metric(atkd_row, "metrics/mAP_0.5")
        atkd_ap = metric(atkd_row, "metrics/mAP_0.5:0.95")
        ccl_ap50 = metric(ccl_row, "metrics/mAP_0.5")
        ccl_ap = metric(ccl_row, "metrics/mAP_0.5:0.95")
        full_ap50 = metric(full_row, "metrics/mAP_0.5")
        full_ap = metric(full_row, "metrics/mAP_0.5:0.95")
        complete = all(value is not None for value in (det_ap50, det_ap, atkd_ap50, atkd_ap, ccl_ap50, ccl_ap, full_ap50, full_ap))
        note = "pending"
        if complete:
            note = "pre_200_snapshot" if epoch < 200 else "aligned_snapshot"
        rows.append(
            {
                "epoch": str(epoch),
                "det_only_ap50": fmt(det_ap50),
                "det_only_ap": fmt(det_ap),
                "atkd_ap50": fmt(atkd_ap50),
                "atkd_ap": fmt(atkd_ap),
                "atkd_delta_ap50": fmt(delta(atkd_ap50, det_ap50)),
                "atkd_delta_ap": fmt(delta(atkd_ap, det_ap)),
                "ccl_ap50": fmt(ccl_ap50),
                "ccl_ap": fmt(ccl_ap),
                "ccl_delta_ap50": fmt(delta(ccl_ap50, det_ap50)),
                "ccl_delta_ap": fmt(delta(ccl_ap, det_ap)),
                "full_ap50": fmt(full_ap50),
                "full_ap": fmt(full_ap),
                "full_delta_ap50": fmt(delta(full_ap50, det_ap50)),
                "full_delta_ap": fmt(delta(full_ap, det_ap)),
                "full_minus_atkd_ap": fmt(delta(full_ap, atkd_ap)),
                "full_minus_ccl_ap": fmt(delta(full_ap, ccl_ap)),
                "best_component_by_ap": best_component(atkd_ap, ccl_ap, full_ap),
                "note": note,
            }
        )
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_markdown_table(path: Path, title: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_running_status(
    path: Path,
    summary_rows: list[dict[str, str]],
    loss_rows: list[dict[str, str]],
    milestone_rows: list[dict[str, str]],
    snapshot_time: str,
) -> None:
    lines = [
        "# CCLKD YOLOv5x 400epoch Running Status",
        "",
        f"Snapshot time: `{snapshot_time}`.",
        "",
        "This archive tracks the four YOLOv5x scaling-fix b32/s0/400ep main runs. "
        "No loss change, sweep, or stop action is implied by this generated report.",
        "",
        "## Current Results",
        "",
        "| run | epoch | AP50 | AP | same-epoch det AP | delta AP | KD/det | feature ok | NaN/Inf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {run} | {epoch} | {ap50} | {ap} | {det_ap} | {delta_ap} | {kd_det} | {feature} | {nan} |".format(
                run=RUNS[row["run_key"]]["label"],
                epoch=row["latest_epoch"],
                ap50=row["AP50"],
                ap=row["AP"],
                det_ap=row["detonly_same_epoch_ap"],
                delta_ap=row["delta_ap_vs_detonly_same_epoch"],
                kd_det=row["weighted_kd_to_student_det_ratio"],
                feature=row["feature_capture_ok"],
                nan=row["nan_or_inf_detected"],
            )
        )
    lines.extend(
        [
            "",
            "## Loss Contribution",
            "",
            "| run | epoch | student det sum | ATKD | CCL | KD total | ATKD share | CCL share | KD/det |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in loss_rows:
        lines.append(
            "| {run} | {epoch} | {student} | {atkd} | {ccl} | {kd} | {atkd_share} | {ccl_share} | {ratio} |".format(
                run=row["run"],
                epoch=row["epoch"],
                student=row["student_det_sum"],
                atkd=row["atkd"],
                ccl=row["ccl"],
                kd=row["kd_total"],
                atkd_share=row["atkd_share_in_kd"],
                ccl_share=row["ccl_share_in_kd"],
                ratio=row["weighted_kd/det"],
            )
        )
    lines.extend(
        [
            "",
            "## Milestone Readiness",
            "",
            "| epoch | det AP | ATKD AP | CCL AP | Full AP | Full - ATKD | Full - CCL | note |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in milestone_rows:
        lines.append(
            "| {epoch} | {det} | {atkd} | {ccl} | {full} | {fatkd} | {fccl} | {note} |".format(
                epoch=row["epoch"],
                det=row["det_only_ap"],
                atkd=row["atkd_ap"],
                ccl=row["ccl_ap"],
                full=row["full_ap"],
                fatkd=row["full_minus_atkd_ap"],
                fccl=row["full_minus_ccl_ap"],
                note=row["note"],
            )
        )
    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            "- Use exact same-epoch comparisons only.",
            "- If a milestone row contains `pending`, do not make aligned milestone decisions from it.",
            "- Continue the active main runs unless a documented stop condition is met.",
            "- Do not modify CCLKD loss or launch sweeps before the 200/250 milestone evidence justifies it.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--milestones", nargs="+", type=int, default=DEFAULT_MILESTONES)
    parser.add_argument("--snapshot-time", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_time = args.snapshot_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = load_archive(args.archive)
    summary_rows = build_summary_rows(data)
    loss_rows = build_loss_rows(summary_rows)
    milestone_rows = build_milestone_rows(data, args.milestones)
    write_csv(args.archive / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(args.archive / "loss_contribution_latest.csv", LOSS_FIELDS, loss_rows)
    write_markdown_table(
        args.archive / "loss_contribution_latest.md",
        "YOLOv5x CCLKD Loss Contribution",
        ["run", "epoch", "AP", "delta_AP", "student_det_sum", "atkd", "ccl", "kd_total", "atkd_share_in_kd", "ccl_share_in_kd", "weighted_kd/det", "cop_pos"],
        loss_rows,
    )
    write_csv(args.archive / "milestone_component_comparison.csv", MILESTONE_FIELDS, milestone_rows)
    write_markdown_table(
        args.archive / "milestone_component_comparison.md",
        "YOLOv5x CCLKD Milestone Component Comparison",
        ["epoch", "det_only_ap", "atkd_ap", "atkd_delta_ap", "ccl_ap", "ccl_delta_ap", "full_ap", "full_delta_ap", "full_minus_atkd_ap", "full_minus_ccl_ap", "best_component_by_ap", "note"],
        milestone_rows,
    )
    write_running_status(args.archive / "running_status.md", summary_rows, loss_rows, milestone_rows, snapshot_time)
    print(args.archive / "summary.csv")
    print(args.archive / "loss_contribution_latest.csv")
    print(args.archive / "loss_contribution_latest.md")
    print(args.archive / "milestone_component_comparison.csv")
    print(args.archive / "milestone_component_comparison.md")
    print(args.archive / "running_status.md")


if __name__ == "__main__":
    main()
