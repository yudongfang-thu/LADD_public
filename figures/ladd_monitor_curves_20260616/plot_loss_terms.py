from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

RUNS = [
    ("s skipA2 s0 (90)", "90_yolo11s_mosaic_skipA2_s0_b.csv"),
    ("s A1A2B s123 (AutoDL)", "autodl_yolo11s_mosaic_A1A2B_s123_b.csv"),
    ("n mosaic a2last s123 (90)", "90_yolo11n_mosaic_a2last_s123_b.csv"),
    ("n nomosaic yolo-init s0 (AutoDL)", "autodl_yolo11n_nomosaic_ladd_yoloinit_s0_b.csv"),
]


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "epoch": float(row["epoch"]),
                        "s_sep": float(row["train/s_sep_loss"]),
                        "r_aux": float(row["train/r_aux_loss"]),
                        "ap": float(row["metrics/mAP50-95(B)"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=False)
    for label, filename in RUNS:
        rows = read_rows(DATA / filename)
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["s_sep"] for r in rows], linewidth=1.8, label=label)
        axes[1].plot(epochs, [r["r_aux"] for r in rows], linewidth=1.8, label=label)
        axes[2].plot(epochs, [r["ap"] for r in rows], linewidth=1.8, label=label)

    axes[0].set_ylabel("train/s_sep_loss")
    axes[1].set_ylabel("train/r_aux_loss")
    axes[2].set_ylabel("AP50-95")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[1].set_yscale("symlog", linthresh=1e-5)
    axes[2].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"ladd_aux_loss_curves.{ext}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
