#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


MAP = "metrics/mAP50-95(B)"
MAP50 = "metrics/mAP50(B)"
LOSS_COLS = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]


plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class RunSpec:
    protocol: str
    model: str
    modality: str
    seed: int
    batch: int
    path: Path
    note: str = ""

    @property
    def label(self) -> str:
        return self.modality.upper()


def nomosaic_path(modality: str, model: str, batch: int) -> Path:
    return (
        ROOT
        / "ladd/results/ladd90_formal_baselines_20260612/results"
        / f"{modality}_yolo11{model}_s0_b{batch}.csv"
    )


RUNS = [
    *[
        RunSpec("nomosaic", model, modality, 0, batch, nomosaic_path(modality, model, batch))
        for model, batch in [("n", 64), ("s", 64), ("m", 32), ("l", 32), ("x", 16)]
        for modality in ["sar", "rgb"]
    ],
    RunSpec(
        "mosaic100",
        "n",
        "sar",
        0,
        64,
        ROOT / "docs/experiments/ladd_mosaic100_mainline_curves_20260618/data/n_sar_baseline_results.csv",
    ),
    RunSpec(
        "mosaic100",
        "n",
        "rgb",
        0,
        64,
        ROOT / "docs/experiments/ladd_mosaic100_mainline_curves_20260618/data/n_rgb_baseline_results.csv",
    ),
    RunSpec(
        "mosaic100",
        "s",
        "sar",
        0,
        64,
        ROOT / "docs/experiments/ladd_mosaic100_mainline_curves_20260618/data/s_sar_baseline_results.csv",
    ),
    RunSpec(
        "mosaic100",
        "s",
        "rgb",
        0,
        64,
        ROOT / "docs/experiments/ladd_mosaic100_mainline_curves_20260618/data/s_rgb_baseline_results.csv",
        "90 remote confirmed 759 epochs",
    ),
    RunSpec(
        "mosaic100",
        "m",
        "sar",
        0,
        64,
        ROOT
        / "docs/experiments/archive_legacy_ladd_20260618/a1a2b_and_bstage/current_progress_curves_20260617_0831/raw/90/m_sar_mosaic_baseline_running.csv",
        "90 remote confirmed latest snapshot; 793 epochs",
    ),
    RunSpec(
        "mosaic100",
        "m",
        "rgb",
        0,
        64,
        ROOT
        / "docs/experiments/archive_legacy_ladd_20260618/a1a2b_and_bstage/current_progress_curves_20260617_0831/raw/90/m_rgb_mosaic_baseline_stopped.csv",
        "90 remote confirmed stopped snapshot; 680 epochs",
    ),
]


PROTOCOL_MODELS = {
    "nomosaic": ["n", "s", "m", "l", "x"],
    "mosaic100": ["n", "s", "m"],
}


STYLE = {
    "sar": {"color": "#2f6f9f", "linestyle": "-", "linewidth": 1.8},
    "rgb": {"color": "#b55d2a", "linestyle": "--", "linewidth": 1.8},
}


MODEL_COLORS = {
    "n": "#1f77b4",
    "s": "#ff7f0e",
    "m": "#2ca02c",
    "l": "#d62728",
    "x": "#9467bd",
}


def read_run(spec: RunSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    df = pd.read_csv(spec.path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" not in df:
        df.insert(0, "epoch", range(1, len(df) + 1))
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    if df["epoch"].min(skipna=True) == 0:
        df["epoch_plot"] = df["epoch"] + 1
    else:
        df["epoch_plot"] = df["epoch"]
    for col in df.columns:
        if col != "epoch_plot":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
    missing = [col for col in [MAP, MAP50] if col not in df.columns]
    if missing:
        raise RuntimeError(f"{spec.path} misses required columns: {missing}")
    loss_cols = [col for col in LOSS_COLS if col in df.columns]
    df["train/det_loss"] = df[loss_cols].sum(axis=1, min_count=1) if loss_cols else pd.NA
    return df


def smooth(s: pd.Series, window: int = 11) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(window=window, min_periods=1, center=True).mean()


def best_row(df: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(df[metric], errors="coerce")
    if values.notna().sum() == 0:
        raise RuntimeError(f"No finite values for {metric}")
    return df.loc[values.idxmax()]


def summarize(spec: RunSpec, df: pd.DataFrame) -> dict[str, object]:
    best_ap = best_row(df, MAP)
    best_ap50 = best_row(df, MAP50)
    last = df.iloc[-1]
    complete = int(last["epoch_plot"]) >= 800
    return {
        "protocol": spec.protocol,
        "model": spec.model,
        "modality": spec.modality,
        "seed": spec.seed,
        "batch": spec.batch,
        "epochs_recorded": int(len(df)),
        "last_epoch": int(last["epoch_plot"]),
        "complete_800": bool(complete),
        "best_map50_95": float(best_ap[MAP]),
        "best_map50_95_epoch": int(best_ap["epoch_plot"]),
        "last_map50_95": float(last[MAP]),
        "best_map50": float(best_ap50[MAP50]),
        "best_map50_epoch": int(best_ap50["epoch_plot"]),
        "last_map50": float(last[MAP50]),
        "last_det_loss": float(last["train/det_loss"]) if pd.notna(last["train/det_loss"]) else "",
        "source": str(spec.path.relative_to(ROOT)),
        "note": spec.note,
    }


def collect() -> tuple[dict[tuple[str, str, str], pd.DataFrame], list[dict[str, object]]]:
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for spec in RUNS:
        df = read_run(spec)
        frames[(spec.protocol, spec.model, spec.modality)] = df
        rows.append(summarize(spec, df))
    return frames, rows


def make_gap_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(r["protocol"], r["model"], r["modality"]): r for r in rows}
    gap_rows = []
    for protocol, models in PROTOCOL_MODELS.items():
        for model in models:
            sar = by_key.get((protocol, model, "sar"))
            rgb = by_key.get((protocol, model, "rgb"))
            if not sar or not rgb:
                continue
            gap_rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "sar_best_map50_95": sar["best_map50_95"],
                    "sar_best_epoch": sar["best_map50_95_epoch"],
                    "sar_last_map50_95": sar["last_map50_95"],
                    "rgb_best_map50_95": rgb["best_map50_95"],
                    "rgb_best_epoch": rgb["best_map50_95_epoch"],
                    "rgb_last_map50_95": rgb["last_map50_95"],
                    "rgb_minus_sar_gap_map50_95": float(rgb["best_map50_95"]) - float(sar["best_map50_95"]),
                    "sar_best_map50": sar["best_map50"],
                    "rgb_best_map50": rgb["best_map50"],
                    "rgb_minus_sar_gap_map50": float(rgb["best_map50"]) - float(sar["best_map50"]),
                    "sar_epochs": sar["last_epoch"],
                    "rgb_epochs": rgb["last_epoch"],
                    "status": "complete" if sar["complete_800"] and rgb["complete_800"] else "partial",
                    "note": "; ".join(n for n in [str(sar.get("note", "")), str(rgb.get("note", ""))] if n),
                }
            )
    return gap_rows


def plot_protocol_curves(protocol: str, frames: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    models = PROTOCOL_MODELS[protocol]
    fig, axes = plt.subplots(2, len(models), figsize=(3.15 * len(models), 6.2), sharex=False)
    if len(models) == 1:
        axes = axes.reshape(2, 1)
    for col_idx, model in enumerate(models):
        for modality in ["sar", "rgb"]:
            df = frames[(protocol, model, modality)]
            style = STYLE[modality]
            label = modality.upper()
            axes[0, col_idx].plot(
                df["epoch_plot"],
                smooth(df[MAP]),
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
            best = best_row(df, MAP)
            axes[0, col_idx].scatter(
                [best["epoch_plot"]],
                [best[MAP]],
                s=22,
                color=style["color"],
                zorder=4,
            )
            axes[1, col_idx].plot(
                df["epoch_plot"],
                smooth(df["train/det_loss"]),
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
        axes[0, col_idx].set_title(f"YOLO11{model}")
        axes[0, col_idx].set_ylabel("mAP50-95(B)" if col_idx == 0 else "")
        axes[1, col_idx].set_ylabel("train box+cls+dfl" if col_idx == 0 else "")
        axes[1, col_idx].set_xlabel("epoch")
        axes[0, col_idx].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"baseline_{protocol}_dual_modality_curves.png")
    fig.savefig(FIG / f"baseline_{protocol}_dual_modality_curves.pdf")
    plt.close(fig)


def plot_capacity_overlay(protocol: str, frames: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    models = PROTOCOL_MODELS[protocol]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex=False)
    panels = [
        (axes[0, 0], "sar", MAP, "SAR baseline mAP50-95(B)"),
        (axes[0, 1], "rgb", MAP, "RGB baseline mAP50-95(B)"),
        (axes[1, 0], "sar", "train/det_loss", "SAR train box+cls+dfl"),
        (axes[1, 1], "rgb", "train/det_loss", "RGB train box+cls+dfl"),
    ]
    for ax, modality, metric, ylabel in panels:
        for model in models:
            df = frames[(protocol, model, modality)]
            color = MODEL_COLORS[model]
            label = f"YOLO11{model}"
            ax.plot(
                df["epoch_plot"],
                smooth(df[metric]),
                label=label,
                color=color,
                linewidth=1.8,
            )
            if metric == MAP:
                best = best_row(df, MAP)
                ax.scatter([best["epoch_plot"]], [best[MAP]], color=color, s=20, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("epoch")
        ax.legend(loc="lower right" if metric == MAP else "upper right", ncols=1)
    fig.tight_layout()
    fig.savefig(FIG / f"baseline_{protocol}_capacity_overlay.png")
    fig.savefig(FIG / f"baseline_{protocol}_capacity_overlay.pdf")
    plt.close(fig)


def plot_gap(gap_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, protocol in zip(axes, ["nomosaic", "mosaic100"]):
        rows = [r for r in gap_rows if r["protocol"] == protocol]
        x = list(range(len(rows)))
        width = 0.35
        sar = [float(r["sar_best_map50_95"]) for r in rows]
        rgb = [float(r["rgb_best_map50_95"]) for r in rows]
        ax.bar([i - width / 2 for i in x], sar, width, label="SAR", color=STYLE["sar"]["color"], alpha=0.85)
        ax.bar([i + width / 2 for i in x], rgb, width, label="RGB", color=STYLE["rgb"]["color"], alpha=0.85)
        for i, r in enumerate(rows):
            gap = float(r["rgb_minus_sar_gap_map50_95"])
            ymax = max(sar[i], rgb[i])
            ax.text(i, ymax + 0.006, f"gap {gap:+.3f}", ha="center", va="bottom", fontsize=8)
            if r["status"] != "complete":
                ax.text(i, min(sar[i], rgb[i]) - 0.018, "partial", ha="center", va="top", fontsize=8, color="#a33")
        ax.set_xticks(x)
        ax.set_xticklabels([f"YOLO11{r['model']}" for r in rows])
        ax.set_ylabel("best mAP50-95(B)" if protocol == "nomosaic" else "")
        ax.set_title(protocol)
        ax.legend(loc="lower right")
        ax.set_ylim(0.45, max(rgb + sar) + 0.05)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_dual_modality_gap_by_protocol.png")
    fig.savefig(FIG / "baseline_dual_modality_gap_by_protocol.pdf")
    plt.close(fig)


def fmt(v: object, digits: int = 5) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def write_outputs(rows: list[dict[str, object]], gap_rows: list[dict[str, object]]) -> None:
    summary_path = OUT / "baseline_dual_modality_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gap_path = OUT / "baseline_dual_modality_gap_table.csv"
    with gap_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gap_rows[0]))
        writer.writeheader()
        writer.writerows(gap_rows)

    lines = [
        "# Baseline Dual-Modality Gap Table",
        "",
        "Interpretation: `gap` is `RGB best mAP50-95(B) - SAR best mAP50-95(B)` under the same capacity and protocol. "
        "The curves compare the two modality-specific single-modality baselines, not a fused two-input detector.",
        "",
    ]
    for protocol in ["nomosaic", "mosaic100"]:
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| Model | SAR best AP @ epoch | RGB best AP @ epoch | RGB-SAR gap | SAR last AP | RGB last AP | SAR/RGB epochs | Status | Note |",
                "|---|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for r in [x for x in gap_rows if x["protocol"] == protocol]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"YOLO11{r['model']}",
                        f"{fmt(r['sar_best_map50_95'])} @{r['sar_best_epoch']}",
                        f"{fmt(r['rgb_best_map50_95'])} @{r['rgb_best_epoch']}",
                        fmt(r["rgb_minus_sar_gap_map50_95"]),
                        fmt(r["sar_last_map50_95"]),
                        fmt(r["rgb_last_map50_95"]),
                        f"{r['sar_epochs']}/{r['rgb_epochs']}",
                        str(r["status"]),
                        str(r.get("note", "")).replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Outputs",
            "",
            "- `figures/baseline_nomosaic_dual_modality_curves.png`",
            "- `figures/baseline_mosaic100_dual_modality_curves.png`",
            "- `figures/baseline_nomosaic_capacity_overlay.png`",
            "- `figures/baseline_mosaic100_capacity_overlay.png`",
            "- `figures/baseline_dual_modality_gap_by_protocol.png`",
            "- `baseline_dual_modality_summary.csv`",
            "- `baseline_dual_modality_gap_table.csv`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    frames, rows = collect()
    gap_rows = make_gap_rows(rows)
    for protocol in PROTOCOL_MODELS:
        plot_protocol_curves(protocol, frames)
        plot_capacity_overlay(protocol, frames)
    plot_gap(gap_rows)
    write_outputs(rows, gap_rows)
    print(f"Wrote baseline curves and tables to {OUT}")


if __name__ == "__main__":
    main()
