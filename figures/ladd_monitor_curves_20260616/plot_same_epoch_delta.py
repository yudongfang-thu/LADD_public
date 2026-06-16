from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PAIRS = [
    {
        "group": "yolo11n",
        "label": "LADD n mosaic a2last s123",
        "method": "90_yolo11n_mosaic_a2last_s123_b.csv",
        "baseline": "90_yolo11n_sar_baseline_mosaic_close100_s0.csv",
        "note": "same protocol, different seed baseline",
    },
    {
        "group": "yolo11n",
        "label": "LADD n nomosaic yolo-init s0",
        "method": "autodl_yolo11n_nomosaic_ladd_yoloinit_s0_b.csv",
        "baseline": "autodl_yolo11n_sar_baseline_nomosaic_s0.csv",
        "note": "same protocol, same seed",
    },
    {
        "group": "yolo11n",
        "label": "CMDistill n nomosaic s0",
        "method": "autodl_yolo11n_nomosaic_cmdistill_s0_b.csv",
        "baseline": "autodl_yolo11n_sar_baseline_nomosaic_s0.csv",
        "note": "same protocol, same seed",
    },
    {
        "group": "yolo11s",
        "label": "LADD s mosaic A1B skipA2 s0",
        "method": "90_yolo11s_mosaic_skipA2_s0_b.csv",
        "baseline": "90_yolo11s_sar_baseline_mosaic_close100_s0.csv",
        "note": "same protocol, same seed",
    },
    {
        "group": "yolo11s",
        "label": "LADD s mosaic A1A2B s123",
        "method": "autodl_yolo11s_mosaic_A1A2B_s123_b.csv",
        "baseline": "90_yolo11s_sar_baseline_mosaic_close100_s0.csv",
        "note": "same protocol, different seed baseline",
    },
    {
        "group": "yolo11s",
        "label": "LADD s nomosaic yolo-init s0",
        "method": "autodl_yolo11s_nomosaic_ladd_yoloinit_s0_b.csv",
        "baseline": "autodl_yolo11s_sar_baseline_nomosaic_s0.csv",
        "note": "same protocol, same seed",
    },
    {
        "group": "yolo11s",
        "label": "CMDistill s nomosaic s0",
        "method": "autodl_yolo11s_nomosaic_cmdistill_s0_b.csv",
        "baseline": "autodl_yolo11s_sar_baseline_nomosaic_s0.csv",
        "note": "same protocol, same seed",
    },
]


def read_curve(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            try:
                epoch = int(float(row["epoch"]))
                rows[epoch] = {
                    "epoch": epoch,
                    "ap50": float(row["metrics/mAP50(B)"]),
                    "ap": float(row["metrics/mAP50-95(B)"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def main() -> None:
    summary = []
    curves: dict[str, list[tuple[str, list[int], list[float]]]] = {}

    for pair in PAIRS:
        method = read_curve(DATA / pair["method"])
        baseline = read_curve(DATA / pair["baseline"])
        common_epochs = sorted(set(method) & set(baseline))
        if not common_epochs:
            continue
        deltas = [method[e]["ap"] - baseline[e]["ap"] for e in common_epochs]
        curves.setdefault(pair["group"], []).append((pair["label"], common_epochs, deltas))
        latest_epoch = max(common_epochs)
        mr = method[latest_epoch]
        br = baseline[latest_epoch]
        best_delta_epoch = max(common_epochs, key=lambda e: method[e]["ap"] - baseline[e]["ap"])
        summary.append(
            {
                "group": pair["group"],
                "label": pair["label"],
                "latest_common_epoch": latest_epoch,
                "method_ap50": f"{mr['ap50']:.5f}",
                "method_ap": f"{mr['ap']:.5f}",
                "baseline_ap50_same_epoch": f"{br['ap50']:.5f}",
                "baseline_ap_same_epoch": f"{br['ap']:.5f}",
                "delta_ap50": f"{mr['ap50'] - br['ap50']:+.5f}",
                "delta_ap": f"{mr['ap'] - br['ap']:+.5f}",
                "best_delta_epoch": best_delta_epoch,
                "best_delta_ap": f"{method[best_delta_epoch]['ap'] - baseline[best_delta_epoch]['ap']:+.5f}",
                "note": pair["note"],
            }
        )

    with (ROOT / "same_epoch_delta_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "label",
                "latest_common_epoch",
                "method_ap50",
                "method_ap",
                "baseline_ap50_same_epoch",
                "baseline_ap_same_epoch",
                "delta_ap50",
                "delta_ap",
                "best_delta_epoch",
                "best_delta_ap",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, group in zip(axes, ("yolo11n", "yolo11s")):
        for label, epochs, deltas in curves.get(group, []):
            ax.plot(epochs, deltas, linewidth=1.8, label=label)
        ax.axhline(0.0, color="0.25", linestyle=":", linewidth=1.0)
        ax.set_title(group)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel("AP50-95 delta vs same-epoch SAR baseline")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"same_epoch_delta_curves.{ext}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
