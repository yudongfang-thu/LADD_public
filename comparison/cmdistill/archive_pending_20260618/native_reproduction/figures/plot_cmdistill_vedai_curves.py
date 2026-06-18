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

RGB_BASELINE = 0.695
CMDISTILL_TABLE_I = 0.740


def read_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned: dict[str, float] = {}
            for key, value in row.items():
                k = key.strip()
                try:
                    cleaned[k] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(cleaned)
    return rows


def col(rows: list[dict[str, float]], name: str) -> list[float]:
    return [r[name] for r in rows if name in r]


def best_row(rows: list[dict[str, float]], metric: str) -> dict[str, float]:
    return max(rows, key=lambda r: r.get(metric, float("-inf")))


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


def plot_overview(cmd: list[dict[str, float]], rgb: list[dict[str, float]], ir: list[dict[str, float]]) -> None:
    c = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    ax = axes[0, 0]

    for rows, label, color in [
        (rgb, "RGB baseline train", c[0]),
        (ir, "IR baseline train", c[2]),
        (cmd, "CMDistill IR -> RGB", c[3]),
    ]:
        ax.plot(col(rows, "epoch"), col(rows, "metrics/mAP_0.5"), label=label, color=color, linewidth=1.6)
    ax.axhline(RGB_BASELINE, color="0.35", linestyle="--", linewidth=1.1, label="RGB baseline best 0.695")
    ax.axhline(CMDISTILL_TABLE_I, color="0.1", linestyle=":", linewidth=1.4, label="CMDistill Table I 0.740")
    b = best_row(cmd, "metrics/mAP_0.5")
    ax.scatter([b["epoch"]], [b["metrics/mAP_0.5"]], color=c[3], s=35, zorder=5)
    ax.annotate(
        f"best {b['metrics/mAP_0.5']:.3f}@{int(b['epoch'])}",
        (b["epoch"], b["metrics/mAP_0.5"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0.55, 0.76)
    ax.legend(frameon=False, ncol=1)

    ax = axes[0, 1]
    ax.plot(col(cmd, "epoch"), col(cmd, "metrics/mAP_0.5"), color=c[3], linewidth=1.8)
    ax.axhline(RGB_BASELINE, color="0.35", linestyle="--", linewidth=1.1)
    ax.axhline(CMDISTILL_TABLE_I, color="0.1", linestyle=":", linewidth=1.4)
    ax.scatter([b["epoch"]], [b["metrics/mAP_0.5"]], color=c[3], s=35, zorder=5)
    ax.set_xlim(-1, 35)
    ax.set_ylim(0.65, 0.75)
    ax.set_xlabel("CMDistill epoch")
    ax.set_ylabel("mAP@0.5")
    ax.text(0.02, 0.05, "early peak, then degradation", transform=ax.transAxes, fontsize=9)

    ax = axes[1, 0]
    for rows, label, color in [
        (rgb, "RGB baseline train", c[0]),
        (ir, "IR baseline train", c[2]),
        (cmd, "CMDistill IR -> RGB", c[3]),
    ]:
        ax.plot(col(rows, "epoch"), col(rows, "metrics/mAP_0.5:0.95"), label=label, color=color, linewidth=1.6)
    b95 = best_row(cmd, "metrics/mAP_0.5:0.95")
    ax.scatter([b95["epoch"]], [b95["metrics/mAP_0.5:0.95"]], color=c[3], s=35, zorder=5)
    ax.annotate(
        f"best {b95['metrics/mAP_0.5:0.95']:.3f}@{int(b95['epoch'])}",
        (b95["epoch"], b95["metrics/mAP_0.5:0.95"]),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_ylim(0.30, 0.43)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    det_loss = [
        r.get("train/box_loss", 0.0) + r.get("train/obj_loss", 0.0) + r.get("train/cls_loss", 0.0)
        for r in cmd
    ]
    epochs = col(cmd, "epoch")
    ax.plot(epochs, det_loss, color=c[4], linewidth=1.6, label="det loss")
    ax.set_xlabel("CMDistill epoch")
    ax.set_ylabel("train det loss")
    ax2 = ax.twinx()
    ax2.plot(epochs, col(cmd, "metrics/mAP_0.5"), color=c[3], linewidth=1.4, label="mAP@0.5")
    ax2.set_ylabel("mAP@0.5")
    ax2.axhline(RGB_BASELINE, color="0.35", linestyle="--", linewidth=1.0)
    lines = [l for l in ax.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, "cmdistill_vedai_overview")


def plot_diagnostics(cmd: list[dict[str, float]]) -> None:
    c = plt.cm.tab10.colors
    epochs = col(cmd, "epoch")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)

    ax = axes[0]
    ax.plot(epochs, col(cmd, "metrics/mAP_0.5"), color=c[3], linewidth=1.8, label="mAP@0.5")
    ax.plot(epochs, col(cmd, "metrics/mAP_0.5:0.95"), color=c[1], linewidth=1.5, label="mAP@0.5:0.95")
    ax.axhline(RGB_BASELINE, color="0.35", linestyle="--", linewidth=1.0, label="RGB baseline best")
    ax.axhline(CMDISTILL_TABLE_I, color="0.1", linestyle=":", linewidth=1.3, label="Table I target")
    ax.set_ylabel("Validation metric")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    det_loss = [
        r.get("train/box_loss", 0.0) + r.get("train/obj_loss", 0.0) + r.get("train/cls_loss", 0.0)
        for r in cmd
    ]
    ax.plot(epochs, det_loss, color=c[4], linewidth=1.5, label="det loss")
    ax.plot(epochs, col(cmd, "train/feat_loss"), color=c[0], linewidth=1.4, label="feature loss")
    ax.plot(epochs, col(cmd, "train/rel_loss"), color=c[2], linewidth=1.4, label="relation loss")
    ax.set_ylabel("Train loss")
    ax.legend(frameon=False, ncol=3)

    ax = axes[2]
    ax.plot(epochs, col(cmd, "train/out_loss"), color=c[5], linewidth=1.4, label="output KD loss")
    ax.set_ylabel("output KD loss")
    ax2 = ax.twinx()
    ax2.plot(epochs, col(cmd, "x/lr0"), color="0.25", linewidth=1.3, linestyle="--", label="lr0")
    ax2.set_ylabel("lr0")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right")
    ax.set_xlabel("CMDistill epoch")

    fig.tight_layout()
    save(fig, "cmdistill_vedai_diagnostics")


def write_summary(cmd: list[dict[str, float]], rgb: list[dict[str, float]], ir: list[dict[str, float]]) -> None:
    rows = []
    for name, data in [("RGB baseline", rgb), ("IR baseline", ir), ("CMDistill IR->RGB", cmd)]:
        b50 = best_row(data, "metrics/mAP_0.5")
        b95 = best_row(data, "metrics/mAP_0.5:0.95")
        latest = data[-1]
        rows.append(
            {
                "name": name,
                "epochs_done": int(latest["epoch"]) + 1,
                "latest_mAP50": latest["metrics/mAP_0.5"],
                "latest_mAP5095": latest["metrics/mAP_0.5:0.95"],
                "best_mAP50": b50["metrics/mAP_0.5"],
                "best_mAP50_epoch": int(b50["epoch"]),
                "best_mAP5095": b95["metrics/mAP_0.5:0.95"],
                "best_mAP5095_epoch": int(b95["epoch"]),
            }
        )
    lines = [
        "| run | epochs done | latest mAP50 | latest mAP50-95 | best mAP50 | best mAP50 epoch | best mAP50-95 | best mAP50-95 epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['epochs_done']} | {r['latest_mAP50']:.4f} | "
            f"{r['latest_mAP5095']:.4f} | {r['best_mAP50']:.4f} | {r['best_mAP50_epoch']} | "
            f"{r['best_mAP5095']:.4f} | {r['best_mAP5095_epoch']} |"
        )
    (OUT / "cmdistill_vedai_curve_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_style()
    cmd = read_csv(DATA / "cmdistill_rgb_ir_e300_aligned_nogeo_results.csv")
    rgb = read_csv(DATA / "baseline_rgb_e300_results.csv")
    ir = read_csv(DATA / "baseline_ir_e300_results.csv")
    plot_overview(cmd, rgb, ir)
    plot_diagnostics(cmd)
    write_summary(cmd, rgb, ir)


if __name__ == "__main__":
    main()
