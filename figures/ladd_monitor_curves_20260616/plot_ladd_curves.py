from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

RUNS = [
    {
        "group": "yolo11n",
        "label": "SAR baseline n mosaic close100 s0",
        "file": "90_yolo11n_sar_baseline_mosaic_close100_s0.csv",
        "style": ":",
        "color": "0.25",
        "baseline": True,
    },
    {
        "group": "yolo11n",
        "label": "SAR baseline n nomosaic s0",
        "file": "autodl_yolo11n_sar_baseline_nomosaic_s0.csv",
        "style": "-.",
        "color": "0.50",
        "baseline": True,
    },
    {
        "group": "yolo11n",
        "label": "LADD n mosaic a2last s123 (90)",
        "file": "90_yolo11n_mosaic_a2last_s123_b.csv",
        "style": "-",
    },
    {
        "group": "yolo11n",
        "label": "LADD n nomosaic yolo-init s0 (AutoDL)",
        "file": "autodl_yolo11n_nomosaic_ladd_yoloinit_s0_b.csv",
        "style": "-",
    },
    {
        "group": "yolo11n",
        "label": "CMDistill n nomosaic s0 (AutoDL)",
        "file": "autodl_yolo11n_nomosaic_cmdistill_s0_b.csv",
        "style": "--",
    },
    {
        "group": "yolo11s",
        "label": "SAR baseline s mosaic close100 s0",
        "file": "90_yolo11s_sar_baseline_mosaic_close100_s0.csv",
        "style": ":",
        "color": "0.25",
        "baseline": True,
    },
    {
        "group": "yolo11s",
        "label": "SAR baseline s nomosaic s0",
        "file": "autodl_yolo11s_sar_baseline_nomosaic_s0.csv",
        "style": "-.",
        "color": "0.50",
        "baseline": True,
    },
    {
        "group": "yolo11s",
        "label": "LADD s mosaic A1->B skipA2 s0 (90)",
        "file": "90_yolo11s_mosaic_skipA2_s0_b.csv",
        "style": "-",
    },
    {
        "group": "yolo11s",
        "label": "LADD s mosaic A1A2B s123 (AutoDL)",
        "file": "autodl_yolo11s_mosaic_A1A2B_s123_b.csv",
        "style": "-",
    },
    {
        "group": "yolo11s",
        "label": "LADD s nomosaic yolo-init s0 (AutoDL)",
        "file": "autodl_yolo11s_nomosaic_ladd_yoloinit_s0_b.csv",
        "style": "-",
    },
    {
        "group": "yolo11s",
        "label": "CMDistill s nomosaic s0 (AutoDL)",
        "file": "autodl_yolo11s_nomosaic_cmdistill_s0_b.csv",
        "style": "--",
    },
    {
        "group": "yolo11m",
        "label": "SAR baseline m mosaic close100 s0 (90)",
        "file": "90_yolo11m_sar_baseline_mosaic_close100_s0.csv",
        "style": ":",
        "color": "0.25",
        "baseline": True,
    },
    {
        "group": "yolo11m",
        "label": "RGB teacher m mosaic close100 s0 (90)",
        "file": "90_yolo11m_rgb_baseline_mosaic_close100_s0.csv",
        "style": ":",
        "color": "0.55",
        "baseline": True,
    },
    {
        "group": "yolo11m",
        "label": "LADD m nomosaic yolo-init s0 (90)",
        "file": "90_yolo11m_nomosaic_ladd_yoloinit_s0_b.csv",
        "style": "-",
    },
]


def read_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "epoch": float(row["epoch"]),
                        "ap50": float(row["metrics/mAP50(B)"]),
                        "ap": float(row["metrics/mAP50-95(B)"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    return max(rows, key=lambda r: r["ap"])


def plot_group(group: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    summary_rows = []

    for run in [r for r in RUNS if r["group"] == group]:
        rows = read_curve(DATA / run["file"])
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        ap50 = [r["ap50"] for r in rows]
        ap = [r["ap"] for r in rows]
        latest = rows[-1]
        best = best_row(rows)
        color = run.get("color")
        alpha = 0.85 if run.get("baseline") else 1.0

        line0 = axes[0].plot(
            epochs,
            ap,
            run["style"],
            linewidth=1.7 if run.get("baseline") else 1.9,
            label=run["label"],
            color=color,
            alpha=alpha,
        )[0]
        line1 = axes[1].plot(
            epochs,
            ap50,
            run["style"],
            linewidth=1.7 if run.get("baseline") else 1.9,
            label=run["label"],
            color=color,
            alpha=alpha,
        )[0]
        axes[0].scatter([best["epoch"]], [best["ap"]], s=22, color=line0.get_color(), alpha=alpha)
        axes[1].scatter([best["epoch"]], [best["ap50"]], s=22, color=line1.get_color(), alpha=alpha)

        summary_rows.append(
            {
                "group": group,
                "label": run["label"],
                "latest_epoch": int(latest["epoch"]),
                "latest_ap50": f"{latest['ap50']:.5f}",
                "latest_ap": f"{latest['ap']:.5f}",
                "best_epoch": int(best["epoch"]),
                "best_ap50": f"{best['ap50']:.5f}",
                "best_ap": f"{best['ap']:.5f}",
            }
        )

    axes[0].set_ylabel("AP50-95")
    axes[1].set_ylabel("AP50")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"{group}_curves.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with (ROOT / f"{group}_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "label",
                "latest_epoch",
                "latest_ap50",
                "latest_ap",
                "best_epoch",
                "best_ap50",
                "best_ap",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    for group in ("yolo11n", "yolo11s", "yolo11m"):
        plot_group(group)


if __name__ == "__main__":
    main()
