#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def first_existing(pattern: str) -> Path:
    matches = sorted(SOURCE.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


RUNS_N = [
    (
        "SAR baseline n",
        first_existing("mnt/dataY/ydf/projects/LADD_og/**/sar_yolo11n_hbb_800ep_cos_closeAt100_pat80_s0_gpu4/results.csv"),
        "#4c78a8",
        ":",
        1.3,
    ),
    (
        "Old cap2 full B s0",
        first_existing("docs/experiments/ladd_mosaic90_20260528_artifacts/ladd_b_runs/ladd_hbb_ogsod11n_ladd800r2_cap2_s0_b_e800_b64_s0_gpu4/results.csv"),
        "#f58518",
        "-",
        1.8,
    ),
    (
        "Old cap2 full B s42",
        first_existing("docs/experiments/ladd_mosaic90_20260528_artifacts/ladd_b_runs/ladd_hbb_ogsod11n_ladd800r2_cap2_s42_b_e800_b64_s42_gpu3/results.csv"),
        "#f58518",
        "--",
        1.5,
    ),
    (
        "Old cap2 full B s123",
        first_existing("docs/experiments/ladd_mosaic90_20260528_artifacts/ladd_b_runs/ladd_hbb_ogsod11n_ladd800r2_cap2_s123_b_e800_b64_s123_gpu5/results.csv"),
        "#f58518",
        "-.",
        1.5,
    ),
    (
        "Current A2last s0",
        first_existing("runs_public/ogsod/hbb/ladd_mosaic_a2last_20260615/ladd_hbb_ogsod11n_mosaic_a2last_cap2_s0_b_e800_b64_s0_gpu1/results.csv"),
        "#54a24b",
        "-",
        2.0,
    ),
    (
        "Current A2last s123",
        first_existing("runs_public/ogsod/hbb/ladd_mosaic_a2last_20260615/ladd_hbb_ogsod11n_mosaic_a2last_cap2_s123_b_e800_b64_s123_gpu3/results.csv"),
        "#54a24b",
        "--",
        1.8,
    ),
    (
        "Current skipA2 from A1 s42",
        first_existing("runs_public/ogsod/hbb/ladd_mosaic_a2last_20260615/ladd_hbb_ogsod11n_mosaic_a1best_skipa2_cap2_s42_b_e800_b64_s42_gpu1/results.csv"),
        "#e45756",
        "-",
        2.2,
    ),
]

RUNS_S = [
    (
        "SAR baseline s",
        first_existing("runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/sar_yolo11s_hbb_mosaicE800_closeAt100_s0_gpu1_20260615/results.csv"),
        "#4c78a8",
        ":",
        1.3,
    ),
    (
        "RGB teacher s",
        first_existing("runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/rgb_yolo11s_hbb_mosaicE800_closeAt100_s0_gpu1_20260615/results.csv"),
        "#9ecae9",
        ":",
        1.3,
    ),
    (
        "Current skipA2 from A1 s42",
        first_existing("runs_public/ogsod/hbb/ladd_mosaic_s_20260616/ladd_hbb_ogsod11s_mosaic_yolo11s_cap2_s42_A1best_skipA2_B_20260616_b_e800_b64_s42_gpu1/results.csv"),
        "#e45756",
        "-",
        2.2,
    ),
    (
        "Current A1A2 s0 A2 phase",
        first_existing("runs_public/ogsod/hbb/ladd_mosaic_s_20260616/ladd_hbb_ogsod11s_mosaic_yolo11s_cap2_s0_mainline_A1A2B_20260616_a2_e50_b64_s0_gpu1/results.csv"),
        "#54a24b",
        "--",
        1.7,
    ),
    (
        "Current A1A2 s42 A2 phase",
        first_existing("runs_public/ogsod/hbb/ladd_mosaic_s_20260616/ladd_hbb_ogsod11s_mosaic_yolo11s_cap2_s42_mainline_A1A2B_20260616_a2_e50_b64_s42_gpu5/results.csv"),
        "#54a24b",
        "-.",
        1.7,
    ),
]

METRICS = {
    "ap": ("metrics/mAP50-95(B)", "mAP50-95"),
    "ap50": ("metrics/mAP50(B)", "mAP50"),
}


def curve(path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "epoch": pd.to_numeric(df["epoch"], errors="coerce") + 1,
            "value": pd.to_numeric(df[metric], errors="coerce"),
        }
    ).dropna()


def summarize(path: Path, metric: str) -> dict[str, object]:
    c = curve(path, metric)
    best_idx = c["value"].idxmax()
    return {
        "rows": len(c),
        "latest_epoch": int(c["epoch"].iloc[-1]),
        "latest": float(c["value"].iloc[-1]),
        "best_epoch": int(c.loc[best_idx, "epoch"]),
        "best": float(c.loc[best_idx, "value"]),
    }


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


def plot_group(runs, out_stem: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS.values(), strict=True):
        for label, path, color, linestyle, linewidth in runs:
            c = curve(path, metric)
            if "baseline" in label.lower() or "teacher" in label.lower():
                best = c["value"].max()
                ax.axhline(best, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9, label=f"{label} best")
            else:
                ax.plot(c["epoch"], c["value"], label=label, color=color, linestyle=linestyle, linewidth=linewidth)
                ax.scatter([c["epoch"].iloc[-1]], [c["value"].iloc[-1]], color=color, s=12, zorder=3)
        ax.set_xlabel("Completed epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 800)
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{out_stem}.{ext}")
    plt.close(fig)


def write_summary() -> None:
    rows = []
    for group, runs in [("yolo11n", RUNS_N), ("yolo11s", RUNS_S)]:
        for label, path, *_ in runs:
            ap = summarize(path, METRICS["ap"][0])
            ap50 = summarize(path, METRICS["ap50"][0])
            rows.append(
                {
                    "group": group,
                    "run": label,
                    "rows": ap["rows"],
                    "latest_epoch": ap["latest_epoch"],
                    "latest_ap50": f"{ap50['latest']:.5f}",
                    "latest_ap": f"{ap['latest']:.5f}",
                    "best_ap50": f"{ap50['best']:.5f}",
                    "best_ap50_epoch": ap50["best_epoch"],
                    "best_ap": f"{ap['best']:.5f}",
                    "best_ap_epoch": ap["best_epoch"],
                    "source_csv": str(path.relative_to(ROOT)),
                }
            )
    with (ROOT / "skipa2_compare_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    style()
    plot_group(RUNS_N, "yolo11n_skipa2_vs_full_b", "YOLO11n mosaic_first100_close700: skipA2 vs full B")
    plot_group(RUNS_S, "yolo11s_current_skipa2_early", "YOLO11s current early curves")
    write_summary()


if __name__ == "__main__":
    main()
