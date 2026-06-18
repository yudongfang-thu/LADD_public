from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = REPO / "figures" / "ladd_monitor_curves_20260616" / "data"

AP_COL = "metrics/mAP50-95(B)"
AP50_COL = "metrics/mAP50(B)"


GROUPS = {
    "n_nomosaic": {
        "model": "YOLO11n",
        "protocol": "formal no-mosaic",
        "baseline": {
            "label": "SAR baseline n no-mosaic",
            "file": "autodl_yolo11n_sar_baseline_nomosaic_s0.csv",
        },
        "runs": [
            {
                "label": "Historical LADD-like n original s0 (sep/aux)",
                "file": "90_yolo11n_nomosaic_ladd_original_s0_b.csv",
                "style": "-",
            },
            {
                "label": "Historical LADD-like n cap2 s0 (sep/aux)",
                "file": "90_yolo11n_nomosaic_ladd_cap2_s0_b.csv",
                "style": "-",
            },
            {
                "label": "Historical LADD-like n cap2 s42 (sep/aux)",
                "file": "90_yolo11n_nomosaic_ladd_cap2_s42_b.csv",
                "style": "-",
            },
            {
                "label": "Historical LADD-like n cap2 s0 BNfreeze (sep/aux)",
                "file": "90_yolo11n_nomosaic_ladd_cap2_s0_bnfreeze_b.csv",
                "style": "-.",
            },
            {
                "label": "Historical LADD-like n cap2 s123 BNfreeze (sep/aux)",
                "file": "90_yolo11n_nomosaic_ladd_cap2_s123_bnfreeze_b.csv",
                "style": "-.",
            },
            {
                "label": "Current diagnostic n yolo-init A1->B_A2core (sep/aux)",
                "file": "autodl_yolo11n_nomosaic_ladd_yoloinit_s0_b.csv",
                "style": ":",
            },
            {
                "label": "CMDistill n no-mosaic",
                "file": "autodl_yolo11n_nomosaic_cmdistill_s0_b.csv",
                "style": "--",
            },
        ],
    },
    "n_mosaic100": {
        "model": "YOLO11n",
        "protocol": "mosaic100 / close@100",
        "baseline": {
            "label": "SAR baseline n mosaic100",
            "file": "90_yolo11n_sar_baseline_mosaic_close100_s0.csv",
        },
        "runs": [
            {
                "label": "LADD-like n A2last s0 (sep/aux)",
                "file": "90_yolo11n_mosaic_a2last_s0_b.csv",
                "style": "-",
            },
            {
                "label": "LADD-like n skipA2 s42 (sep/aux)",
                "file": "90_yolo11n_mosaic_skipA2_s42_b.csv",
                "style": "-",
            },
            {
                "label": "LADD-like n A2last s123 running (sep/aux)",
                "file": "90_yolo11n_mosaic_a2last_s123_b.csv",
                "style": ":",
            },
        ],
    },
    "s_nomosaic": {
        "model": "YOLO11s",
        "protocol": "formal no-mosaic",
        "baseline": {
            "label": "SAR baseline s no-mosaic",
            "file": "autodl_yolo11s_sar_baseline_nomosaic_s0.csv",
        },
        "runs": [
            {
                "label": "LADD-like s yolo-init (sep/aux)",
                "file": "autodl_yolo11s_nomosaic_ladd_yoloinit_s0_b.csv",
                "style": "-",
            },
            {
                "label": "CMDistill s no-mosaic",
                "file": "autodl_yolo11s_nomosaic_cmdistill_s0_b.csv",
                "style": "--",
            },
        ],
    },
    "s_mosaic100": {
        "model": "YOLO11s",
        "protocol": "mosaic100 / close@100",
        "baseline": {
            "label": "SAR baseline s mosaic100",
            "file": "90_yolo11s_sar_baseline_mosaic_close100_s0.csv",
        },
        "runs": [
            {
                "label": "LADD-like s skipA2 s0 running (sep/aux)",
                "file": "90_yolo11s_mosaic_skipA2_s0_b.csv",
                "style": "-",
            },
        ],
    },
}


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def read_curve(filename: str) -> list[dict[str, float]]:
    path = DATA / filename
    rows: list[dict[str, float]] = []
    if not path.exists():
        print(f"missing: {path}")
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            epoch = _float(row, "epoch")
            ap = _float(row, AP_COL)
            ap50 = _float(row, AP50_COL)
            box = _float(row, "train/box_loss")
            cls = _float(row, "train/cls_loss")
            dfl = _float(row, "train/dfl_loss")
            if epoch is None or ap is None:
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "ap": ap,
                    "ap50": ap50 if ap50 is not None else float("nan"),
                    "box": box if box is not None else float("nan"),
                    "cls": cls if cls is not None else float("nan"),
                    "dfl": dfl if dfl is not None else float("nan"),
                }
            )
    return rows


def best(rows: list[dict[str, float]]) -> dict[str, float]:
    return max(rows, key=lambda row: row["ap"])


def baseline_lookup(rows: list[dict[str, float]]) -> dict[int, float]:
    return {int(row["epoch"]): row["ap"] for row in rows}


def same_epoch_delta(rows: list[dict[str, float]], baseline: dict[int, float]) -> tuple[list[float], list[float]]:
    epochs: list[float] = []
    deltas: list[float] = []
    for row in rows:
        epoch = int(row["epoch"])
        if epoch in baseline:
            epochs.append(row["epoch"])
            deltas.append(row["ap"] - baseline[epoch])
    return epochs, deltas


def summarize_run(group_key: str, label: str, rows: list[dict[str, float]], baseline_rows: list[dict[str, float]]) -> dict[str, str]:
    b = best(rows)
    last = rows[-1]
    bl_best = best(baseline_rows)
    bl_last_same = baseline_lookup(baseline_rows).get(int(last["epoch"]))
    return {
        "group": group_key,
        "label": label,
        "rows": str(len(rows)),
        "best_epoch": str(int(b["epoch"])),
        "best_ap": f"{b['ap']:.5f}",
        "last_epoch": str(int(last["epoch"])),
        "last_ap": f"{last['ap']:.5f}",
        "baseline_best_ap": f"{bl_best['ap']:.5f}",
        "last_minus_baseline_same_epoch": "" if bl_last_same is None else f"{last['ap'] - bl_last_same:.5f}",
        "last_minus_baseline_best": f"{last['ap'] - bl_best['ap']:.5f}",
    }


def plot_group(key: str, cfg: dict[str, object]) -> list[dict[str, str]]:
    baseline_cfg = cfg["baseline"]  # type: ignore[index]
    baseline_rows = read_curve(baseline_cfg["file"])  # type: ignore[index]
    baseline_map = baseline_lookup(baseline_rows)
    fig_width = 12.2 if key == "n_nomosaic" else 10.8
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.2), sharex=False)

    axes[0].plot(
        [r["epoch"] for r in baseline_rows],
        [r["ap"] for r in baseline_rows],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=baseline_cfg["label"],  # type: ignore[index]
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.4, label="SAR baseline")

    summary = [summarize_run(key, baseline_cfg["label"], baseline_rows, baseline_rows)]  # type: ignore[index]
    colors = plt.cm.tab10.colors
    for idx, run in enumerate(cfg["runs"]):  # type: ignore[index]
        rows = read_curve(run["file"])
        if not rows:
            continue
        color = colors[idx % len(colors)]
        style = run.get("style", "-")
        label = run["label"]
        axes[0].plot(
            [r["epoch"] for r in rows],
            [r["ap"] for r in rows],
            linestyle=style,
            color=color,
            linewidth=1.9,
            label=label,
        )
        b = best(rows)
        axes[0].scatter([b["epoch"]], [b["ap"]], color=color, s=26, zorder=4)
        epochs, deltas = same_epoch_delta(rows, baseline_map)
        if epochs:
            axes[1].plot(epochs, deltas, linestyle=style, color=color, linewidth=1.9, label=label)
        summary.append(summarize_run(key, label, rows, baseline_rows))

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AP50-95")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AP50-95 - same-protocol SAR baseline")
    axes[1].set_ylim(-0.18, 0.04)
    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        legend_size = 6.4 if key == "n_nomosaic" else 7.4
        ax.legend(frameon=False, fontsize=legend_size, loc="best")

    fig.suptitle(f"{cfg['model']} | {cfg['protocol']}", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{key}_ap_with_same_protocol_baseline.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return summary


def write_summary(rows: list[dict[str, str]]) -> None:
    fields = [
        "group",
        "label",
        "rows",
        "best_epoch",
        "best_ap",
        "last_epoch",
        "last_ap",
        "baseline_best_ap",
        "last_minus_baseline_same_epoch",
        "last_minus_baseline_best",
    ]
    with (OUT / "capacity_protocol_split_summary_20260617.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.4,
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "axes.unicode_minus": False,
        }
    )
    all_rows: list[dict[str, str]] = []
    for key, cfg in GROUPS.items():
        all_rows.extend(plot_group(key, cfg))
    write_summary(all_rows)


if __name__ == "__main__":
    main()
