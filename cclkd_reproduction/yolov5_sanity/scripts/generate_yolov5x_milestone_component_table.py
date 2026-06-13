#!/usr/bin/env python3
"""Generate exact-epoch YOLOv5x CCLKD component comparison tables.

The script reads compact archived run CSVs and writes:
  - milestone_component_comparison.csv
  - milestone_component_comparison.md

It intentionally uses exact epoch matches only. Missing epochs are written as
"pending" rather than filled with nearest-neighbor values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_ARCHIVE = Path(
    "cclkd_reproduction/yolov5_sanity/results/"
    "scalingfix_paper_components_400ep_20260613"
)
DEFAULT_MILESTONES = [80, 100, 125, 150, 185, 200, 250, 300, 350, 399]

RUN_FILES = {
    "det_only": "det_only_same_trainer",
    "atkd": "paper_atkd_only",
    "ccl": "paper_ccl_only",
    "full": "paper_full",
}


def clean_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        (key.strip() if key is not None else key): (
            value.strip() if isinstance(value, str) else value
        )
        for key, value in row.items()
    }


def read_results(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    by_epoch: Dict[int, Dict[str, str]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = clean_row(raw)
            epoch_text = row.get("epoch", "")
            if not epoch_text:
                continue
            try:
                epoch = int(float(epoch_text))
            except ValueError:
                continue
            by_epoch[epoch] = row
    return by_epoch


def metric(row: Optional[Dict[str, str]], key: str) -> Optional[float]:
    if not row:
        return None
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "pending"
    return f"{value:.5f}"


def delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return value - baseline


def best_component(atkd_ap: Optional[float], ccl_ap: Optional[float], full_ap: Optional[float]) -> str:
    candidates = {
        "atkd": atkd_ap,
        "ccl": ccl_ap,
        "full": full_ap,
    }
    available = {key: value for key, value in candidates.items() if value is not None}
    if not available:
        return "pending"
    return max(available, key=lambda key: available[key])


def note_for(epoch: int, values: Iterable[Optional[float]]) -> str:
    if any(value is None for value in values):
        return "pending"
    if epoch < 200:
        return "pre_200_snapshot"
    return "aligned_snapshot"


def build_rows(archive: Path, milestones: List[int]) -> List[Dict[str, str]]:
    runs = {
        key: read_results(archive / "runs" / run_dir / "results.csv")
        for key, run_dir in RUN_FILES.items()
    }

    rows: List[Dict[str, str]] = []
    for epoch in milestones:
        det = runs["det_only"].get(epoch)
        atkd = runs["atkd"].get(epoch)
        ccl = runs["ccl"].get(epoch)
        full = runs["full"].get(epoch)

        det_ap50 = metric(det, "metrics/mAP_0.5")
        det_ap = metric(det, "metrics/mAP_0.5:0.95")
        atkd_ap50 = metric(atkd, "metrics/mAP_0.5")
        atkd_ap = metric(atkd, "metrics/mAP_0.5:0.95")
        ccl_ap50 = metric(ccl, "metrics/mAP_0.5")
        ccl_ap = metric(ccl, "metrics/mAP_0.5:0.95")
        full_ap50 = metric(full, "metrics/mAP_0.5")
        full_ap = metric(full, "metrics/mAP_0.5:0.95")

        row = {
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
            "note": note_for(
                epoch,
                [
                    det_ap50,
                    det_ap,
                    atkd_ap50,
                    atkd_ap,
                    ccl_ap50,
                    ccl_ap,
                    full_ap50,
                    full_ap,
                ],
            ),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    columns = [
        "epoch",
        "det_only_ap",
        "atkd_ap",
        "atkd_delta_ap",
        "ccl_ap",
        "ccl_delta_ap",
        "full_ap",
        "full_delta_ap",
        "full_minus_atkd_ap",
        "full_minus_ccl_ap",
        "best_component_by_ap",
        "note",
    ]
    lines = [
        "# YOLOv5x CCLKD Milestone Component Comparison",
        "",
        "Exact epoch matches only. Missing epochs are marked as `pending`; no nearest-epoch substitution is used.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Archive directory containing runs/<run_key>/results.csv.",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=DEFAULT_MILESTONES,
        help="Exact epochs to compare.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV path. Defaults inside --archive.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults inside --archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive
    rows = build_rows(archive, args.milestones)
    csv_out = args.csv_out or archive / "milestone_component_comparison.csv"
    md_out = args.md_out or archive / "milestone_component_comparison.md"
    write_csv(csv_out, rows)
    write_markdown(md_out, rows)
    print(csv_out)
    print(md_out)


if __name__ == "__main__":
    main()
