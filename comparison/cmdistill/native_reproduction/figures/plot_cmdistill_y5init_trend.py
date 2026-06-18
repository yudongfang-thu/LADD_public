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

RGB_BASELINE_BEST = 0.6919
CMDISTILL_TABLE_I = 0.740


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


def moving_average(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def setup_style() -> None:
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


def det_loss(rows: list[dict[str, float]], prefix: str) -> list[float]:
    return [
        row.get(f"{prefix}/box_loss", 0.0)
        + row.get(f"{prefix}/obj_loss", 0.0)
        + row.get(f"{prefix}/cls_loss", 0.0)
        for row in rows
    ]


def plot_trend(y5init: list[dict[str, float]], old: list[dict[str, float]], rgb: list[dict[str, float]]) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))

    ax = axes[0, 0]
    ax.plot(get(rgb, "epoch"), get(rgb, "metrics/mAP_0.5"), color=colors[0], linewidth=1.3, alpha=0.7, label="RGB baseline")
    ax.plot(get(old, "epoch"), get(old, "metrics/mAP_0.5"), color=colors[3], linewidth=1.2, alpha=0.45, label="old: baseline-init")
    ax.plot(get(y5init, "epoch"), get(y5init, "metrics/mAP_0.5"), color=colors[2], linewidth=1.8, label="corrected: yolov5s-init")
    ax.plot(get(y5init, "epoch"), moving_average(get(y5init, "metrics/mAP_0.5")), color=colors[2], linewidth=2.5, alpha=0.35, label="corrected 5-epoch avg")
    ax.axhline(RGB_BASELINE_BEST, color="0.35", linestyle="--", linewidth=1.0, label="RGB baseline best")
    ax.axhline(CMDISTILL_TABLE_I, color="0.1", linestyle=":", linewidth=1.3, label="Table I target")
    b = best(y5init, "metrics/mAP_0.5")
    ax.scatter([b["epoch"]], [b["metrics/mAP_0.5"]], color=colors[2], s=35, zorder=4)
    ax.annotate(f"{b['metrics/mAP_0.5']:.3f}@{int(b['epoch'])}", (b["epoch"], b["metrics/mAP_0.5"]), xytext=(7, 8), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0.0, 0.76)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.plot(get(y5init, "epoch"), get(y5init, "metrics/mAP_0.5"), color=colors[2], linewidth=1.8, label="mAP@0.5")
    ax.plot(get(y5init, "epoch"), get(y5init, "metrics/mAP_0.5:0.95"), color=colors[1], linewidth=1.5, label="mAP@0.5:0.95")
    ax.plot(get(y5init, "epoch"), moving_average(get(y5init, "metrics/mAP_0.5")), color=colors[2], linewidth=2.5, alpha=0.3, label="mAP@0.5 avg")
    ax.axhline(RGB_BASELINE_BEST, color="0.35", linestyle="--", linewidth=1.0)
    ax.axhline(CMDISTILL_TABLE_I, color="0.1", linestyle=":", linewidth=1.3)
    ax.set_xlabel("Corrected CMDistill epoch")
    ax.set_ylabel("Validation metric")
    ax.set_ylim(0.0, 0.76)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    ax.plot(get(y5init, "epoch"), det_loss(y5init, "train"), color=colors[4], linewidth=1.4, label="det loss")
    ax.plot(get(y5init, "epoch"), get(y5init, "train/feat_loss"), color=colors[0], linewidth=1.4, label="feature loss")
    ax.plot(get(y5init, "epoch"), get(y5init, "train/rel_loss"), color=colors[2], linewidth=1.4, label="relation loss")
    ax.set_xlabel("Corrected CMDistill epoch")
    ax.set_ylabel("Train loss")
    ax.legend(frameon=False, ncol=3)

    ax = axes[1, 1]
    ax.plot(get(y5init, "epoch"), get(y5init, "train/out_loss"), color=colors[5], linewidth=1.4, label="output KD loss")
    ax2 = ax.twinx()
    ax2.plot(get(y5init, "epoch"), get(y5init, "x/lr0"), color="0.25", linestyle="--", linewidth=1.2, label="lr0")
    ax.set_xlabel("Corrected CMDistill epoch")
    ax.set_ylabel("output KD loss")
    ax2.set_ylabel("lr0")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, "cmdistill_y5init_trend")


def write_summary(y5init: list[dict[str, float]], old: list[dict[str, float]], rgb: list[dict[str, float]]) -> None:
    lines = [
        "| run | epochs logged | latest mAP50 | best mAP50 | best epoch | latest mAP50-95 | best mAP50-95 | best mAP50-95 epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in [
        ("RGB baseline", rgb),
        ("old CMDistill baseline-init", old),
        ("corrected CMDistill yolov5s-init", y5init),
    ]:
        b50 = best(rows, "metrics/mAP_0.5")
        b95 = best(rows, "metrics/mAP_0.5:0.95")
        lines.append(
            f"| {name} | {len(rows)} | {rows[-1]['metrics/mAP_0.5']:.4f} | "
            f"{b50['metrics/mAP_0.5']:.4f} | {int(b50['epoch'])} | "
            f"{rows[-1]['metrics/mAP_0.5:0.95']:.4f} | {b95['metrics/mAP_0.5:0.95']:.4f} | {int(b95['epoch'])} |"
        )
    (OUT / "cmdistill_y5init_trend_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_style()
    y5init = read_rows(DATA / "cmdistill_rgb_ir_y5init_e300_aligned_nogeo_results.csv")
    old = read_rows(DATA / "cmdistill_rgb_ir_e300_aligned_nogeo_results.csv")
    rgb = read_rows(DATA / "baseline_rgb_e300_results.csv")
    plot_trend(y5init, old, rgb)
    write_summary(y5init, old, rgb)


if __name__ == "__main__":
    main()
