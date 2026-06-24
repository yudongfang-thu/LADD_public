#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_PNG = HERE / "dronevehicle_cmdistill_vs_dual_baselines_map5095.png"
OUT_PDF = HERE / "dronevehicle_cmdistill_vs_dual_baselines_map5095.pdf"


RUNS = [
    (
        "RGB baseline (student)",
        DATA_DIR / "dronevehicle_rgb_student_baseline_results.csv",
        "#4C78A8",
        "-",
    ),
    (
        "IR baseline (teacher)",
        DATA_DIR / "dronevehicle_ir_teacher_baseline_results.csv",
        "#F58518",
        "-",
    ),
    (
        "CMDistill IR->RGB",
        DATA_DIR / "dronevehicle_cmdistill_from_yolo_results.csv",
        "#54A24B",
        "-",
    ),
]


def metric_columns(fieldnames: list[str]) -> tuple[str, str]:
    normalized = {name.strip(): name for name in fieldnames}
    return normalized["metrics/mAP50(B)"], normalized["metrics/mAP50-95(B)"]


def read_curve(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    ap50: list[float] = []
    ap5095: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        col50, col5095 = metric_columns(reader.fieldnames or [])
        for i, row in enumerate(reader, start=1):
            epochs.append(int(float(row.get("epoch", i))))
            ap50.append(float(row[col50]))
            ap5095.append(float(row[col5095]))
    return epochs, ap50, ap5095


def best_point(xs: list[int], ys: list[float]) -> tuple[int, float]:
    idx = max(range(len(ys)), key=ys.__getitem__)
    return xs[idx], ys[idx]


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.7))

    curves: dict[str, tuple[list[int], list[float]]] = {}
    for label, path, color, linestyle in RUNS:
        xs, _ap50, ys = read_curve(path)
        curves[label] = (xs, ys)
        ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=2.0)

    # Mark the CMDistill best epoch because this is the positive-result claim point.
    cmd_xs, cmd_ys = curves["CMDistill IR->RGB"]
    bx, by = best_point(cmd_xs, cmd_ys)
    ax.scatter([bx], [by], color="#54A24B", edgecolor="white", linewidth=0.8, s=45, zorder=5)
    ax.annotate(
        f"best {by:.5f} @ ep{bx}",
        xy=(bx, by),
        xytext=(-78, 15),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#54A24B", "lw": 1.0},
        color="#2F6B2F",
        fontsize=8.5,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AP50-95")
    ax.set_xlim(1, 200)
    ax.set_ylim(0.0, 0.46)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.05)

    for label, (xs, ys) in curves.items():
        bx, by = best_point(xs, ys)
        print(f"{label}: best AP50-95={by:.5f} @ epoch {bx}, final={ys[-1]:.5f}")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_PDF}")


if __name__ == "__main__":
    main()
