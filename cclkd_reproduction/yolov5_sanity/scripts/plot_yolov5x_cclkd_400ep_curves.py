#!/usr/bin/env python3
"""Plot YOLOv5x CCLKD 400ep running curves from archived CSV files.

This script is intentionally read-only with respect to experiment outputs. It
reads compact `results.csv` and `cclkd_yolov5_diagnostics.csv` files from the
400epoch scaling-fix archive and writes comparison figures under
`<archive>/figures`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


DEFAULT_ARCHIVE = Path(
    "cclkd_reproduction/yolov5_sanity/results/"
    "scalingfix_paper_components_400ep_20260613"
)

RUNS = {
    "det": {
        "dir": "det_only_same_trainer",
        "label": "Det-only",
        "color": "#4D4D4D",
        "linestyle": "-",
    },
    "atkd": {
        "dir": "paper_atkd_only",
        "label": "ATKD-only",
        "color": "#0072B2",
        "linestyle": "-",
    },
    "ccl": {
        "dir": "paper_ccl_only",
        "label": "CCL-only",
        "color": "#D55E00",
        "linestyle": "-",
    },
    "full": {
        "dir": "paper_full",
        "label": "Full CCLKD",
        "color": "#009E73",
        "linestyle": "-",
    },
}


def clean_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        (key.strip() if key is not None else key): (
            value.strip() if isinstance(value, str) else value
        )
        for key, value in row.items()
    }


def to_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_csv_by_epoch(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, str]] = {}
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
            rows[epoch] = row
    return rows


def series(
    rows: Dict[int, Dict[str, str]],
    key: str,
) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    values: List[float] = []
    for epoch in sorted(rows):
        value = to_float(rows[epoch], key)
        if value is not None:
            epochs.append(epoch)
            values.append(value)
    return epochs, values


def delta_series(
    method: Dict[int, Dict[str, str]],
    det: Dict[int, Dict[str, str]],
    metric: str,
) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    deltas: List[float] = []
    for epoch in sorted(set(method).intersection(det)):
        method_value = to_float(method[epoch], metric)
        det_value = to_float(det[epoch], metric)
        if method_value is None or det_value is None:
            continue
        epochs.append(epoch)
        deltas.append(method_value - det_value)
    return epochs, deltas


def common_diff_series(
    left: Dict[int, Dict[str, str]],
    right: Dict[int, Dict[str, str]],
    metric: str,
) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    diffs: List[float] = []
    for epoch in sorted(set(left).intersection(right)):
        left_value = to_float(left[epoch], metric)
        right_value = to_float(right[epoch], metric)
        if left_value is None or right_value is None:
            continue
        epochs.append(epoch)
        diffs.append(left_value - right_value)
    return epochs, diffs


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_all(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path)
        print(path)


def plot_metric_curves(
    results: Dict[str, Dict[int, Dict[str, str]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharex=True)
    metrics = [
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
        ("metrics/mAP_0.5", "mAP@0.5"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for key in ("det", "atkd", "ccl", "full"):
            epochs, values = series(results[key], metric)
            cfg = RUNS[key]
            ax.plot(
                epochs,
                values,
                label=cfg["label"],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=1.9,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, out_dir, "yolov5x_cclkd_ap_curves")
    plt.close(fig)


def plot_delta_curves(
    results: Dict[str, Dict[int, Dict[str, str]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharex=True)
    metrics = [
        ("metrics/mAP_0.5:0.95", "Delta mAP@0.5:0.95 vs det-only"),
        ("metrics/mAP_0.5", "Delta mAP@0.5 vs det-only"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
        for key in ("atkd", "ccl", "full"):
            epochs, values = delta_series(results[key], results["det"], metric)
            cfg = RUNS[key]
            ax.plot(
                epochs,
                values,
                label=cfg["label"],
                color=cfg["color"],
                linewidth=1.9,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, out_dir, "yolov5x_cclkd_delta_vs_det")
    plt.close(fig)


def plot_component_gaps(
    results: Dict[str, Dict[int, Dict[str, str]]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.0))
    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
    pairs = [
        ("full", "atkd", "Full - ATKD", "#CC79A7"),
        ("full", "ccl", "Full - CCL", "#009E73"),
    ]
    for left, right, label, color in pairs:
        epochs, values = common_diff_series(
            results[left],
            results[right],
            "metrics/mAP_0.5:0.95",
        )
        ax.plot(epochs, values, label=label, color=color, linewidth=1.9)
    ax.set_xlabel("common epoch")
    ax.set_ylabel("mAP@0.5:0.95 gap")
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, out_dir, "yolov5x_cclkd_component_gaps")
    plt.close(fig)


def plot_diagnostics(
    diagnostics: Dict[str, Dict[int, Dict[str, str]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharex=True)
    panels = [
        ("weighted_kd_to_student_det_ratio", "weighted KD / student det"),
        ("ccl_loss", "CCL loss"),
        ("cop_positive_ratio", "COP positive ratio"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels):
        for key in ("atkd", "ccl", "full"):
            epochs, values = series(diagnostics[key], metric)
            cfg = RUNS[key]
            ax.plot(epochs, values, label=cfg["label"], color=cfg["color"], linewidth=1.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_all(fig, out_dir, "yolov5x_cclkd_diagnostics")
    plt.close(fig)


def plot_loss_contributions(
    diagnostics: Dict[str, Dict[int, Dict[str, str]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    labels = ["ATKD-only", "CCL-only", "Full CCLKD"]
    keys = ["atkd", "ccl", "full"]
    atkd_values: List[float] = []
    ccl_values: List[float] = []
    kd_ratios: List[float] = []
    for key in keys:
        latest_epoch = max(diagnostics[key])
        latest = diagnostics[key][latest_epoch]
        atkd_values.append(to_float(latest, "weighted_atkd_loss") or 0.0)
        ccl_values.append(to_float(latest, "weighted_ccl_loss") or 0.0)
        kd_ratios.append(to_float(latest, "weighted_kd_to_student_det_ratio") or 0.0)

    x_positions = list(range(len(keys)))
    axes[0].bar(x_positions, atkd_values, label="weighted ATKD", color="#0072B2")
    axes[0].bar(
        x_positions,
        ccl_values,
        bottom=atkd_values,
        label="weighted CCL",
        color="#D55E00",
    )
    axes[0].set_xticks(x_positions, labels, rotation=12)
    axes[0].set_ylabel("raw weighted KD component")
    axes[0].legend(frameon=False)

    ax_ratio = axes[0].twinx()
    ax_ratio.plot(
        x_positions,
        kd_ratios,
        color="#333333",
        marker="o",
        linewidth=1.6,
        label="weighted KD/det",
    )
    ax_ratio.set_ylabel("weighted KD / student det")
    ax_ratio.legend(frameon=False, loc="upper right")

    full_epochs = sorted(diagnostics["full"])
    full_atkd_share: List[float] = []
    full_ccl_share: List[float] = []
    share_epochs: List[int] = []
    for epoch in full_epochs:
        row = diagnostics["full"][epoch]
        atkd = to_float(row, "weighted_atkd_loss") or 0.0
        ccl = to_float(row, "weighted_ccl_loss") or 0.0
        total = atkd + ccl
        if total <= 0:
            continue
        share_epochs.append(epoch)
        full_atkd_share.append(atkd / total)
        full_ccl_share.append(ccl / total)

    axes[1].plot(share_epochs, full_atkd_share, label="Full: ATKD share", color="#0072B2")
    axes[1].plot(share_epochs, full_ccl_share, label="Full: CCL share", color="#D55E00")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("share inside Full KD total")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save_all(fig, out_dir, "yolov5x_cclkd_loss_contributions")
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    lines = [
        "# YOLOv5x CCLKD 400ep Curve Figures",
        "",
        "Generated from archived `results.csv` and `cclkd_yolov5_diagnostics.csv` files.",
        "",
        "- `yolov5x_cclkd_ap_curves`: raw AP/AP50 curves.",
        "- `yolov5x_cclkd_delta_vs_det`: exact same-epoch deltas vs det-only.",
        "- `yolov5x_cclkd_component_gaps`: Full minus ATKD/CCL at common epochs.",
        "- `yolov5x_cclkd_diagnostics`: KD pressure, CCL loss, and COP positive ratio.",
        "- `yolov5x_cclkd_loss_contributions`: raw weighted ATKD/CCL contributions.",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive
    out_dir = args.out_dir or archive / "figures"

    apply_style()
    results = {
        key: read_csv_by_epoch(archive / "runs" / cfg["dir"] / "results.csv")
        for key, cfg in RUNS.items()
    }
    diagnostics = {
        key: read_csv_by_epoch(archive / "runs" / cfg["dir"] / "cclkd_yolov5_diagnostics.csv")
        for key, cfg in RUNS.items()
    }

    plot_metric_curves(results, out_dir)
    plot_delta_curves(results, out_dir)
    plot_component_gaps(results, out_dir)
    plot_diagnostics(diagnostics, out_dir)
    plot_loss_contributions(diagnostics, out_dir)
    write_readme(out_dir)


if __name__ == "__main__":
    main()
