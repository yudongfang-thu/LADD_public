#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


RUNS = {
    "n": [
        ("SAR baseline n", SOURCE / "sar_yolo11n_baseline_results.csv", "#4c78a8", "-", 1.8),
        ("RGB teacher n", SOURCE / "rgb_yolo11n_baseline_results.csv", "#9ecae9", "--", 1.5),
        ("Prev LADD cap2 n", SOURCE / "prev_ladd_cap2_yolo11n_b800_s42_results.csv", "#b279a2", "--", 1.9),
        ("Current LADD cap2 n", SOURCE / "ladd_cap2_yolo11n_b800_results.csv", "#f58518", "-", 2.1),
        ("CMDistill n", SOURCE / "cmdistill_yolo11n_formal800_results.csv", "#54a24b", "-", 2.1),
    ],
    "s": [
        ("SAR baseline s", SOURCE / "sar_yolo11s_baseline_results.csv", "#4c78a8", "-", 1.8),
        ("RGB teacher s", SOURCE / "rgb_yolo11s_baseline_results.csv", "#9ecae9", "--", 1.5),
        ("Prev LADD cap2 s", SOURCE / "prev_ladd_cap2_yolo11s_b800_s0_results.csv", "#b279a2", "--", 1.9),
        ("CMDistill s", SOURCE / "cmdistill_yolo11s_formal800_results.csv", "#54a24b", "-", 2.1),
    ],
}


METRICS = {
    "ap": ("metrics/mAP50-95(B)", "mAP50-95", "mainline_progress_ap"),
    "ap50": ("metrics/mAP50(B)", "mAP50", "mainline_progress_ap50"),
}


def load_curve(path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise KeyError(f"{metric} not found in {path}")
    epoch = pd.to_numeric(df["epoch"], errors="coerce") + 1
    y = pd.to_numeric(df[metric], errors="coerce")
    return pd.DataFrame({"epoch": epoch, "value": y}).dropna()


def summarize(path: Path, metric: str) -> tuple[int, float, int, float]:
    curve = load_curve(path, metric)
    latest_epoch = int(curve["epoch"].iloc[-1])
    latest = float(curve["value"].iloc[-1])
    best_idx = curve["value"].idxmax()
    best_epoch = int(curve.loc[best_idx, "epoch"])
    best = float(curve.loc[best_idx, "value"])
    return latest_epoch, latest, best_epoch, best


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def plot_metric(metric_key: str, metric_col: str, ylabel: str, out_stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=False)
    for ax, size in zip(axes, ["n", "s"], strict=True):
        for label, path, color, linestyle, linewidth in RUNS[size]:
            if not path.exists():
                continue
            curve = load_curve(path, metric_col)
            ax.plot(
                curve["epoch"],
                curve["value"],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            latest_epoch, latest, best_epoch, best = summarize(path, metric_col)
            if "baseline" in label or "teacher" in label:
                ax.axhline(best, color=color, linestyle=":", linewidth=1.0, alpha=0.45)
            ax.scatter([latest_epoch], [latest], color=color, s=14, zorder=3)
        ax.set_xlabel("Completed epoch")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        ax.set_xlim(0, 800)
        ax.legend(frameon=False, loc="lower right")
        ax.text(
            0.02,
            0.96,
            f"YOLO11{size}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{out_stem}.{ext}")
    plt.close(fig)


def write_summary() -> None:
    rows = []
    for size, specs in RUNS.items():
        for label, path, *_ in specs:
            if not path.exists():
                continue
            ap_latest_epoch, ap_latest, ap_best_epoch, ap_best = summarize(path, METRICS["ap"][0])
            ap50_latest_epoch, ap50_latest, ap50_best_epoch, ap50_best = summarize(path, METRICS["ap50"][0])
            rows.append(
                {
                    "model_size": size,
                    "run": label,
                    "rows": len(pd.read_csv(path)),
                    "latest_epoch": ap_latest_epoch,
                    "latest_ap": f"{ap_latest:.5f}",
                    "best_ap": f"{ap_best:.5f}",
                    "best_ap_epoch": ap_best_epoch,
                    "latest_ap50": f"{ap50_latest:.5f}",
                    "best_ap50": f"{ap50_best:.5f}",
                    "best_ap50_epoch": ap50_best_epoch,
                    "source_csv": str(path.relative_to(ROOT)),
                }
            )
    with (ROOT / "mainline_progress_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    style()
    for metric_key, (metric_col, ylabel, out_stem) in METRICS.items():
        plot_metric(metric_key, metric_col, ylabel, out_stem)
    write_summary()


if __name__ == "__main__":
    main()
