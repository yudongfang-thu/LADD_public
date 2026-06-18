#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "plots"

CMDISTILL_TABLE_I_RGB_BASELINE = 0.702


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key.strip()] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(parsed)
    return rows


def get(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row]


def best(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    return max(rows, key=lambda row: row.get(key, float("-inf")))


def det_loss(rows: list[dict[str, float]], prefix: str) -> list[float]:
    return [
        row.get(f"{prefix}/box_loss", 0.0)
        + row.get(f"{prefix}/obj_loss", 0.0)
        + row.get(f"{prefix}/cls_loss", 0.0)
        for row in rows
    ]


def setup() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Serif",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def add_best_marker(ax, rows: list[dict[str, float]], key: str, color, dy: float = 0.008) -> None:
    b = best(rows, key)
    ax.scatter([b["epoch"]], [b[key]], color=color, s=35, zorder=4)
    ax.annotate(
        f"{b[key]:.3f}@{int(b['epoch'])}",
        (b["epoch"], b[key]),
        xytext=(7, dy * 1000),
        textcoords="offset points",
        fontsize=8,
        color=color,
    )


def plot_baseline_curves(rgb: list[dict[str, float]], ir: list[dict[str, float]]) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))

    ax = axes[0, 0]
    ax.plot(get(rgb, "epoch"), get(rgb, "metrics/mAP_0.5"), color=colors[0], linewidth=1.5, label="RGB")
    ax.plot(get(ir, "epoch"), get(ir, "metrics/mAP_0.5"), color=colors[2], linewidth=1.5, label="IR")
    ax.axhline(CMDISTILL_TABLE_I_RGB_BASELINE, color="0.2", linestyle=":", linewidth=1.2, label="Table I RGB baseline 0.702")
    add_best_marker(ax, rgb, "metrics/mAP_0.5", colors[0])
    add_best_marker(ax, ir, "metrics/mAP_0.5", colors[2])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0.0, 0.75)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.plot(get(rgb, "epoch"), get(rgb, "metrics/mAP_0.5:0.95"), color=colors[0], linewidth=1.5, label="RGB")
    ax.plot(get(ir, "epoch"), get(ir, "metrics/mAP_0.5:0.95"), color=colors[2], linewidth=1.5, label="IR")
    add_best_marker(ax, rgb, "metrics/mAP_0.5:0.95", colors[0], dy=-0.012)
    add_best_marker(ax, ir, "metrics/mAP_0.5:0.95", colors[2], dy=-0.012)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_ylim(0.0, 0.43)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 2]
    ax.plot(get(rgb, "epoch"), get(rgb, "metrics/precision"), color=colors[0], linewidth=1.3, label="RGB P")
    ax.plot(get(rgb, "epoch"), get(rgb, "metrics/recall"), color=colors[0], linestyle="--", linewidth=1.3, label="RGB R")
    ax.plot(get(ir, "epoch"), get(ir, "metrics/precision"), color=colors[2], linewidth=1.3, label="IR P")
    ax.plot(get(ir, "epoch"), get(ir, "metrics/recall"), color=colors[2], linestyle="--", linewidth=1.3, label="IR R")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision / Recall")
    ax.set_ylim(0.0, 0.85)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[1, 0]
    ax.plot(get(rgb, "epoch"), det_loss(rgb, "train"), color=colors[0], linewidth=1.5, label="RGB train")
    ax.plot(get(ir, "epoch"), det_loss(ir, "train"), color=colors[2], linewidth=1.5, label="IR train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train detection loss")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(get(rgb, "epoch"), det_loss(rgb, "val"), color=colors[0], linewidth=1.5, label="RGB val")
    ax.plot(get(ir, "epoch"), det_loss(ir, "val"), color=colors[2], linewidth=1.5, label="IR val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val detection loss")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    ax.plot(get(rgb, "epoch"), get(rgb, "x/lr0"), color=colors[0], linewidth=1.4, label="RGB lr0")
    ax.plot(get(ir, "epoch"), get(ir, "x/lr0"), color=colors[2], linewidth=1.4, label="IR lr0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.legend(frameon=False)

    fig.tight_layout()
    save(fig, "vedai_baseline_training_curves")


def write_summary(rgb: list[dict[str, float]], ir: list[dict[str, float]]) -> None:
    lines = [
        "| baseline | epochs logged | latest mAP50 | best mAP50 | best mAP50 epoch | latest mAP50-95 | best mAP50-95 | best mAP50-95 epoch | latest val det loss | best val det loss | best val det epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in [("RGB", rgb), ("IR", ir)]:
        b50 = best(rows, "metrics/mAP_0.5")
        b95 = best(rows, "metrics/mAP_0.5:0.95")
        vloss = det_loss(rows, "val")
        min_vloss_idx = min(range(len(vloss)), key=vloss.__getitem__)
        lines.append(
            f"| {name} | {len(rows)} | {rows[-1]['metrics/mAP_0.5']:.4f} | "
            f"{b50['metrics/mAP_0.5']:.4f} | {int(b50['epoch'])} | "
            f"{rows[-1]['metrics/mAP_0.5:0.95']:.4f} | {b95['metrics/mAP_0.5:0.95']:.4f} | "
            f"{int(b95['epoch'])} | {vloss[-1]:.4f} | {vloss[min_vloss_idx]:.4f} | "
            f"{int(rows[min_vloss_idx]['epoch'])} |"
        )
    (OUT / "vedai_baseline_curve_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup()
    rgb = read_rows(DATA / "baseline_rgb_e300_results.csv")
    ir = read_rows(DATA / "baseline_ir_e300_results.csv")
    plot_baseline_curves(rgb, ir)
    write_summary(rgb, ir)


if __name__ == "__main__":
    main()
