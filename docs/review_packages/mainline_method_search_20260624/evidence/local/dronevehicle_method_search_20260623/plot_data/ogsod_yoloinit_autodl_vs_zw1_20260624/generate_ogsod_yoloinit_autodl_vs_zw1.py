#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/Users/yudongfang/Desktop/光sar/LADD_public")
EXP = ROOT / "docs/experiments/dronevehicle_method_search_20260623"
AUTODL = EXP / "autodl_probeA_evidence_20260624"
ZW1 = EXP / "plot_data/ogsod_e800_current_curves_20260624/data_latest_ssh"
FIG_DIR = EXP / "figures"
OUT_DIR = EXP / "plot_data/ogsod_yoloinit_autodl_vs_zw1_20260624"

RUNS = {
    "autodl_probe": {
        "label": "AutoDL2 ProbeA",
        "path": AUTODL / "yolo_probeA_results.csv",
        "color": "#0072B2",
        "style": "-",
    },
    "zw1_probe": {
        "label": "ZW1 ProbeA",
        "path": ZW1 / "ogsod_e800_probeA_results.csv",
        "color": "#0072B2",
        "style": "--",
    },
    "autodl_dynamic": {
        "label": "AutoDL2 dynamic",
        "path": AUTODL / "yolo_dynamic_results.csv",
        "color": "#009E73",
        "style": "-",
    },
    "zw1_dynamic": {
        "label": "ZW1 dynamic",
        "path": ZW1 / "ogsod_e800_dynamic_results.csv",
        "color": "#009E73",
        "style": "--",
    },
}

SAR_BASELINE = ZW1 / "ogsod_sar_baseline_results.csv"


def read_curve(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]
    map_col = [c for c in df.columns if "metrics/mAP50-95" in c][0]
    return df[epoch_col].astype(int), df[map_col].astype(float)


def align_delta(y: pd.Series, baseline: pd.Series) -> pd.Series:
    n = min(len(y), len(baseline))
    return y.iloc[:n].reset_index(drop=True) - baseline.iloc[:n].reset_index(drop=True)


def plot_curves(xlim: tuple[int, int] | None, suffix: str) -> tuple[Path, Path]:
    sar_epoch, sar_map = read_curve(SAR_BASELINE)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 260,
        }
    )
    fig, (ax, axd) = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )

    ax.plot(
        sar_epoch,
        sar_map,
        color="#111111",
        linewidth=2.4,
        alpha=0.88,
        label="SAR baseline curve",
        zorder=2,
    )
    ax.axhline(
        sar_map.max(),
        color="#111111",
        linewidth=1.3,
        linestyle=":",
        alpha=0.9,
        label=f"SAR baseline best {sar_map.max():.5f}",
    )

    for info in RUNS.values():
        epoch, y = read_curve(info["path"])
        ax.plot(
            epoch,
            y,
            color=info["color"],
            linestyle=info["style"],
            linewidth=2.0,
            label=info["label"],
            zorder=4,
        )
        if len(y):
            ax.scatter(
                [int(epoch.iloc[-1])],
                [float(y.iloc[-1])],
                s=24,
                color=info["color"],
                zorder=5,
            )
            ax.annotate(
                f"{float(y.iloc[-1]):.3f}",
                (int(epoch.iloc[-1]), float(y.iloc[-1])),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=10,
                color=info["color"],
            )

        delta = align_delta(y, sar_map)
        axd.plot(
            epoch.iloc[: len(delta)],
            delta,
            color=info["color"],
            linestyle=info["style"],
            linewidth=2.0,
            label=info["label"],
        )

    axd.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_ylabel("AP50-95")
    axd.set_ylabel("Delta vs SAR")
    axd.set_xlabel("Epoch")
    ax.grid(alpha=0.18)
    axd.grid(alpha=0.18)

    if xlim is not None:
        ax.set_xlim(*xlim)
        axd.set_xlim(*xlim)
        axd.set_ylim(-0.012, 0.024)
    else:
        ax.set_xlim(1, 560)
        axd.set_xlim(1, 560)
        axd.set_ylim(-0.012, 0.026)

    ax.legend(loc="lower right", ncol=2, frameon=False)
    axd.legend(loc="upper right", ncol=2, frameon=False)

    png = FIG_DIR / f"ogsod_yoloinit_autodl_vs_zw1_{suffix}_20260624.png"
    pdf = FIG_DIR / f"ogsod_yoloinit_autodl_vs_zw1_{suffix}_20260624.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_summary() -> Path:
    _, sar_map = read_curve(SAR_BASELINE)
    rows = []
    epochs = [20, 30, 50, 72, 80, 100, 123, 130, 150, 200, 300, 500]
    curves = {name: read_curve(info["path"]) for name, info in RUNS.items()}
    for epoch_id in epochs:
        if epoch_id > len(sar_map):
            continue
        row = {"epoch": epoch_id, "sar_baseline": float(sar_map.iloc[epoch_id - 1])}
        for name, (_, y) in curves.items():
            if epoch_id <= len(y):
                row[f"{name}_ap5095"] = float(y.iloc[epoch_id - 1])
                row[f"{name}_delta_vs_sar"] = float(y.iloc[epoch_id - 1] - sar_map.iloc[epoch_id - 1])
        rows.append(row)
    out = OUT_DIR / "ogsod_yoloinit_autodl_vs_zw1_epoch_delta_summary_20260624.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = write_summary()
    early_png, early_pdf = plot_curves((1, 150), "early150")
    full_png, full_pdf = plot_curves(None, "full560")
    full800_png, full800_pdf = plot_curves((1, 800), "full800")
    for path in (early_png, early_pdf, full_png, full_pdf, full800_png, full800_pdf, summary):
        print(path)


if __name__ == "__main__":
    main()
