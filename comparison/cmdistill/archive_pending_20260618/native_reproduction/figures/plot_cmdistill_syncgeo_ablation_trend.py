#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "syncgeo_ablation_20260618"
OUT = ROOT / "plots"

RGB_BASELINE_BEST = 0.6919
CMDISTILL_TABLE_I = 0.740
CMDISTILL_TABLE_III_NO_KD = 0.702

RUNS = [
    ("syncgeo_nokd", "sync geo no KD", "syncgeo_nokd_results.csv"),
    ("syncgeo_featrel", "sync geo feat+rel KD", "syncgeo_featrel_partial_results.csv"),
    ("syncgeo_allkd_fix", "sync geo all KD fix", "syncgeo_allkd_fix_results.csv"),
    ("syncgeo_allkd_warm10_fix", "sync geo all KD warm10 fix", "syncgeo_allkd_warm10_fix_results.csv"),
    ("syncgeo_logitonly_fix", "sync geo logit-only KD", "syncgeo_logitonly_fix_results.csv"),
    ("syncgeo_featureonly_fix", "sync geo feature-only KD", "syncgeo_featureonly_fix_results.csv"),
    ("syncgeo_relationonly_fix", "sync geo relation-only KD", "syncgeo_relationonly_fix_results.csv"),
]


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
        for row in csv.DictReader(f):
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


def load_runs() -> dict[str, tuple[str, list[dict[str, float]]]]:
    loaded: dict[str, tuple[str, list[dict[str, float]]]] = {}
    for key, label, filename in RUNS:
        path = DATA / filename
        if path.exists():
            rows = read_rows(path)
            if rows:
                loaded[key] = (label, rows)
    return loaded


def plot(runs: dict[str, tuple[str, list[dict[str, float]]]]) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8))

    ax = axes[0, 0]
    for idx, (key, (label, rows)) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        epochs = get(rows, "epoch")
        m50 = get(rows, "metrics/mAP_0.5")
        ax.plot(epochs, m50, color=color, linewidth=1.5, alpha=0.72, label=label)
        ax.plot(epochs, moving_average(m50), color=color, linewidth=2.4, alpha=0.28)
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
    ax.axhline(CMDISTILL_TABLE_III_NO_KD, color="0.25", linestyle="-.", linewidth=1.0, label="paper Table III no KD")
    ax.axhline(CMDISTILL_TABLE_I, color="0.1", linestyle=":", linewidth=1.2, label="paper Table I CMDistill")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0.0, 0.78)
    ax.legend(frameon=False, loc="lower right", ncol=2)

    ax = axes[0, 1]
    for idx, (key, (label, rows)) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        ax.plot(get(rows, "epoch"), get(rows, "metrics/mAP_0.5:0.95"), color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    for idx, (key, (label, rows)) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        ax.plot(get(rows, "epoch"), det_loss(rows, "train"), color=color, linewidth=1.3, label=f"{label} train")
        ax.plot(get(rows, "epoch"), det_loss(rows, "val"), color=color, linestyle="--", linewidth=1.1, alpha=0.65, label=f"{label} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detection loss")
    ax.legend(frameon=False, loc="upper right", ncol=2)

    ax = axes[1, 1]
    for idx, (key, (label, rows)) in enumerate(runs.items()):
        color = colors[idx % len(colors)]
        epochs = get(rows, "epoch")
        if any(row.get("train/kd_loss", 0.0) != 0.0 for row in rows):
            ax.plot(epochs, get(rows, "train/kd_loss"), color=color, linewidth=1.3, label=f"{label} KD")
            ax.plot(epochs, get(rows, "train/feat_loss"), color=color, linestyle="--", linewidth=1.0, alpha=0.65, label=f"{label} feat")
        else:
            ax.plot(epochs, get(rows, "x/lr0"), color=color, linewidth=1.2, label=f"{label} lr0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KD loss components or lr0")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "cmdistill_syncgeo_ablation_20260618_trend.pdf")
    fig.savefig(OUT / "cmdistill_syncgeo_ablation_20260618_trend.png")
    plt.close(fig)


def write_summary(runs: dict[str, tuple[str, list[dict[str, float]]]]) -> None:
    lines = [
        "| run | rows | last epoch | latest mAP50 | best mAP50 | best epoch | latest mAP50-95 | best mAP50-95 | best mAP50-95 epoch | latest KD | latest feat | latest rel | latest out |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, (label, rows) in runs.items():
        b50 = best(rows, "metrics/mAP_0.5")
        b95 = best(rows, "metrics/mAP_0.5:0.95")
        last = rows[-1]
        lines.append(
            f"| {label} | {len(rows)} | {int(last['epoch'])} | "
            f"{last['metrics/mAP_0.5']:.4f} | {b50['metrics/mAP_0.5']:.4f} | {int(b50['epoch'])} | "
            f"{last['metrics/mAP_0.5:0.95']:.4f} | {b95['metrics/mAP_0.5:0.95']:.4f} | {int(b95['epoch'])} | "
            f"{last.get('train/kd_loss', 0.0):.4f} | {last.get('train/feat_loss', 0.0):.4f} | "
            f"{last.get('train/rel_loss', 0.0):.4f} | {last.get('train/out_loss', 0.0):.4f} |"
        )
    lines += [
        "",
        "Note: rows from active runs are partial snapshots and should be refreshed after the corresponding screen exits.",
        f"References: our RGB best={RGB_BASELINE_BEST:.4f}; CMDistill Table I={CMDISTILL_TABLE_I:.3f}; CMDistill Table III no-KD={CMDISTILL_TABLE_III_NO_KD:.3f}.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cmdistill_syncgeo_ablation_20260618_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup_style()
    runs = load_runs()
    if not runs:
        raise FileNotFoundError(f"No syncgeo CSVs found in {DATA}")
    plot(runs)
    write_summary(runs)


if __name__ == "__main__":
    main()
