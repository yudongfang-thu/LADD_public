#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ABLATION_DIR = DATA / "table3_ablation_20260618_165900"
OUT = ROOT / "plots"

RGB_BASELINE_BEST = 0.6919
CMDISTILL_TABLE_I = 0.740
CMDISTILL_TABLE_III_NO_KD = 0.702

RUN_ORDER = ["no_kd", "log_only", "feature_only", "relation_only", "all"]
RUN_LABELS = {
    "no_kd": "no KD",
    "log_only": "logit only",
    "feature_only": "feature only",
    "relation_only": "relation only",
    "all": "all KD",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Serif",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


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
            if parsed:
                rows.append(parsed)
    return rows


def get(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row]


def available_runs() -> dict[str, list[dict[str, float]]]:
    runs: dict[str, list[dict[str, float]]] = {}
    for name in RUN_ORDER:
        path = ABLATION_DIR / f"{name}_results.csv"
        if path.exists():
            runs[name] = read_rows(path)
    for path in sorted(ABLATION_DIR.glob("*_results.csv")):
        name = path.name.removesuffix("_results.csv")
        if name not in runs:
            runs[name] = read_rows(path)
    return {name: rows for name, rows in runs.items() if rows}


def moving_average(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def best(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    return max(rows, key=lambda row: row.get(key, float("-inf")))


def det_loss(rows: list[dict[str, float]], prefix: str) -> list[float]:
    return [
        row.get(f"{prefix}/box_loss", 0.0)
        + row.get(f"{prefix}/obj_loss", 0.0)
        + row.get(f"{prefix}/cls_loss", 0.0)
        for row in rows
    ]


def plot_trend(runs: dict[str, list[dict[str, float]]]) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.7))

    rgb_path = DATA / "baseline_rgb_e300_results.csv"
    rgb_rows = read_rows(rgb_path) if rgb_path.exists() else []

    ax = axes[0, 0]
    if rgb_rows:
        ax.plot(
            get(rgb_rows, "epoch"),
            get(rgb_rows, "metrics/mAP_0.5"),
            color="0.55",
            linewidth=1.1,
            alpha=0.65,
            label="RGB baseline curve",
        )
    for idx, (name, rows) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        epochs = get(rows, "epoch")
        map50 = get(rows, "metrics/mAP_0.5")
        ax.plot(epochs, map50, color=color, linewidth=1.6, alpha=0.75, label=RUN_LABELS.get(name, name))
        ax.plot(epochs, moving_average(map50), color=color, linewidth=2.4, alpha=0.28)
        b = best(rows, "metrics/mAP_0.5")
        ax.scatter([b["epoch"]], [b["metrics/mAP_0.5"]], color=color, s=30, zorder=4)
        ax.annotate(
            f"{b['metrics/mAP_0.5']:.3f}@{int(b['epoch'])}",
            (b["epoch"], b["metrics/mAP_0.5"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(RGB_BASELINE_BEST, color="0.35", linestyle="--", linewidth=1.0, label="our RGB best")
    ax.axhline(CMDISTILL_TABLE_III_NO_KD, color=colors[3], linestyle="-.", linewidth=1.1, label="paper Table III no KD")
    ax.axhline(CMDISTILL_TABLE_I, color="0.1", linestyle=":", linewidth=1.2, label="paper Table I CMDistill")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0.0, 0.78)
    ax.legend(frameon=False, loc="lower right", ncol=2)

    ax = axes[0, 1]
    for idx, (name, rows) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        epochs = get(rows, "epoch")
        ax.plot(
            epochs,
            get(rows, "metrics/mAP_0.5:0.95"),
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=RUN_LABELS.get(name, name),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    for idx, (name, rows) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        ax.plot(
            get(rows, "epoch"),
            det_loss(rows, "train"),
            color=color,
            linewidth=1.4,
            label=f"{RUN_LABELS.get(name, name)} train",
        )
        ax.plot(
            get(rows, "epoch"),
            det_loss(rows, "val"),
            color=color,
            linewidth=1.1,
            linestyle="--",
            alpha=0.65,
            label=f"{RUN_LABELS.get(name, name)} val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detection loss")
    ax.legend(frameon=False, loc="upper right", ncol=2)

    ax = axes[1, 1]
    for idx, (name, rows) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        epochs = get(rows, "epoch")
        if any(row.get("train/kd_loss", 0.0) != 0.0 for row in rows):
            ax.plot(epochs, get(rows, "train/kd_loss"), color=color, linewidth=1.4, label=f"{RUN_LABELS.get(name, name)} KD")
        else:
            ax.plot(epochs, get(rows, "x/lr0"), color=color, linewidth=1.2, label=f"{RUN_LABELS.get(name, name)} lr0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KD loss or lr0")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cmdistill_table3_ablation_20260618_165900_trend.pdf")
    fig.savefig(OUT / "cmdistill_table3_ablation_20260618_165900_trend.png")
    plt.close(fig)


def write_summary(runs: dict[str, list[dict[str, float]]]) -> None:
    lines = [
        "| run | epochs logged | last epoch | latest mAP50 | best mAP50 | best epoch | latest mAP50-95 | best mAP50-95 | best mAP50-95 epoch | latest train det loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in runs.items():
        b50 = best(rows, "metrics/mAP_0.5")
        b95 = best(rows, "metrics/mAP_0.5:0.95")
        lines.append(
            f"| {RUN_LABELS.get(name, name)} | {len(rows)} | {int(rows[-1]['epoch'])} | "
            f"{rows[-1]['metrics/mAP_0.5']:.4f} | {b50['metrics/mAP_0.5']:.4f} | {int(b50['epoch'])} | "
            f"{rows[-1]['metrics/mAP_0.5:0.95']:.4f} | {b95['metrics/mAP_0.5:0.95']:.4f} | {int(b95['epoch'])} | "
            f"{det_loss(rows, 'train')[-1]:.5f} |"
        )
    lines += [
        "",
        f"References: our RGB best={RGB_BASELINE_BEST:.4f}; CMDistill Table I={CMDISTILL_TABLE_I:.3f}; CMDistill Table III no-KD={CMDISTILL_TABLE_III_NO_KD:.3f}.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cmdistill_table3_ablation_20260618_165900_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_style()
    runs = available_runs()
    if not runs:
        raise FileNotFoundError(f"No *_results.csv files found in {ABLATION_DIR}")
    plot_trend(runs)
    write_summary(runs)


if __name__ == "__main__":
    main()
