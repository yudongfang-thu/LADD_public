from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = REPO / "figures" / "ladd_monitor_curves_20260616" / "data"
AP_COL = "metrics/mAP50-95(B)"


def read_curve(filename: str) -> list[dict[str, float]]:
    path = DATA / filename
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            try:
                rows.append({"epoch": float(row["epoch"]), "ap": float(row[AP_COL])})
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def best(rows: list[dict[str, float]]) -> dict[str, float]:
    return max(rows, key=lambda r: r["ap"])


def plot_curve(ax: plt.Axes, rows: list[dict[str, float]], label: str, *, color: str, linestyle: str = "-") -> None:
    ax.plot([r["epoch"] for r in rows], [r["ap"] for r in rows], label=label, color=color, linestyle=linestyle, linewidth=1.9)
    b = best(rows)
    ax.scatter([b["epoch"]], [b["ap"]], color=color, s=28, zorder=4)


def main() -> None:
    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 7.5, "savefig.dpi": 240})
    baseline = read_curve("90_yolo11s_sar_baseline_mosaic_close100_s0.csv")
    skipa2 = read_curve("90_yolo11s_mosaic_skipA2_s0_b.csv")
    autodl = read_curve("autodl_yolo11s_mosaic_A1A2B_s123_b.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), sharey=True)

    plot_curve(axes[0], baseline, "SAR baseline s mosaic100", color="black", linestyle="--")
    plot_curve(axes[0], skipa2, "LADD-like s skipA2 s0 running (mosaic100, sep/aux)", color="#1f77b4")
    axes[0].set_title("Strict mosaic100 / close@100")

    plot_curve(axes[1], baseline, "SAR baseline s mosaic100 (reference only)", color="black", linestyle="--")
    plot_curve(axes[1], autodl, "LADD-like s A1A2B s123 running (close=0 caveat, sep/aux)", color="#d62728")
    axes[1].set_title("AutoDL run: mosaic=1.0, close=0 caveat")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("AP50-95")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"s_mosaic_running_two_runs_vs_baseline.{ext}", bbox_inches="tight")

    rows = []
    for name, curve in [
        ("SAR baseline s mosaic100", baseline),
        ("90 s skipA2 s0 mosaic100 running", skipa2),
        ("AutoDL s A1A2B s123 close0 running", autodl),
    ]:
        b = best(curve)
        last = curve[-1]
        rows.append(
            {
                "run": name,
                "rows": str(len(curve)),
                "best_epoch": str(int(b["epoch"])),
                "best_ap": f"{b['ap']:.5f}",
                "last_epoch": str(int(last["epoch"])),
                "last_ap": f"{last['ap']:.5f}",
            }
        )
    with (OUT / "s_mosaic_running_two_runs_summary_20260617.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
