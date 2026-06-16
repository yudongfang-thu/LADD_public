#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    filename: str
    group: str
    model: str
    seed: int | None = None
    color: str = "#1f77b4"
    linestyle: str = "-"


RUNS = [
    Run("old_n_s0", "old n LADD s0 A2-best", "old_n_ladd_seed0_a2best_B800_results.csv", "old_ladd", "n", 0, "#1f77b4", "-"),
    Run("old_n_s42", "old n LADD s42 A2-best", "old_n_ladd_seed42_a2best_B800_results.csv", "old_ladd", "n", 42, "#1f77b4", "-"),
    Run("old_n_s123", "old n LADD s123 A2-best", "old_n_ladd_seed123_a2best_B800_results.csv", "old_ladd", "n", 123, "#1f77b4", "-"),
    Run("cur_n_s0", "current n s0 A2-last", "n_ladd_seed0_a2last_B800_results.csv", "current_ladd", "n", 0, "#d62728", "--"),
    Run("cur_n_s42", "current n s42 A1-best skipA2", "n_ladd_seed42_a1best_skipa2_B800_results.csv", "current_ladd", "n", 42, "#d62728", "--"),
    Run("cur_n_s123", "current n s123 A2-last", "n_ladd_seed123_a2last_B800_results.csv", "current_ladd", "n", 123, "#d62728", "--"),
    Run("n_sar_base", "n SAR baseline", "n_sar_baseline_mosaic_results.csv", "baseline", "n", None, "#333333", "-"),
    Run("n_rgb_base", "n RGB baseline", "n_rgb_baseline_mosaic_results.csv", "baseline", "n", None, "#9467bd", "-"),
    Run("s_sar_base", "s SAR baseline", "s_sar_baseline_mosaic_results.csv", "baseline", "s", None, "#333333", "-"),
    Run("s_rgb_base", "s RGB baseline", "s_rgb_baseline_mosaic_results.csv", "baseline", "s", None, "#9467bd", "-"),
    Run("m_sar_base", "m SAR baseline b64", "m_sar_baseline_mosaic_b64_results.csv", "baseline", "m", None, "#2ca02c", "-"),
    Run("m_rgb_base", "m RGB baseline b64", "m_rgb_baseline_mosaic_b64_results.csv", "baseline", "m", None, "#ff7f0e", "-"),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )


def read_run(run: Run) -> pd.DataFrame:
    path = DATA / run.filename
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["epoch_idx"] = range(1, len(df) + 1)
    df["run_key"] = run.key
    return df


def metric_col(df: pd.DataFrame, needle: str) -> str:
    matches = [c for c in df.columns if needle in c]
    if not matches:
        raise KeyError(f"missing {needle}; columns={list(df.columns)}")
    return matches[-1]


def ap(df: pd.DataFrame) -> pd.Series:
    return df[metric_col(df, "mAP50-95")].astype(float)


def det_loss(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ("train/box_loss", "train/cls_loss", "train/dfl_loss") if c in df.columns]
    return df[cols].astype(float).sum(axis=1)


def maybe(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    s = df[col].astype(float)
    if s.isna().all() or s.abs().sum() == 0:
        return None
    return s


def roll(s: pd.Series, w: int = 13) -> pd.Series:
    return s.rolling(window=w, min_periods=1, center=True).mean()


def plot_n_ap(data: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    seeds = [0, 42, 123]
    for ax, seed in zip(axes, seeds):
        for base_key in ["n_sar_base", "n_rgb_base"]:
            run = next(r for r in RUNS if r.key == base_key)
            df = data[run.key]
            y = ap(df)
            ax.plot(df["epoch_idx"], roll(y), color=run.color, lw=1.5, alpha=0.55, label=run.label)
        for run in [r for r in RUNS if r.model == "n" and r.seed == seed and r.group in {"old_ladd", "current_ladd"}]:
            df = data[run.key]
            y = ap(df)
            ax.plot(df["epoch_idx"], y, color=run.color, lw=0.8, alpha=0.14)
            ax.plot(df["epoch_idx"], roll(y), color=run.color, lw=2.0, ls=run.linestyle, label=run.label)
            best_i = int(y.idxmax())
            ax.scatter(df["epoch_idx"].iloc[best_i], y.iloc[best_i], s=20, color=run.color, zorder=4)
        ax.axvline(700, color="#777777", lw=1.0, ls=":", alpha=0.8)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("epoch")
        ax.set_xlim(1, 800)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
    axes[0].set_ylabel("AP50-95")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path = FIG / "fig1_n_ladd_current_vs_old_ap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_n_detector_loss(data: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), sharey=True)
    for ax, seed in zip(axes, [0, 42, 123]):
        for run in [r for r in RUNS if r.model == "n" and r.seed == seed and r.group in {"old_ladd", "current_ladd"}]:
            df = data[run.key]
            ax.plot(df["epoch_idx"], roll(det_loss(df)), color=run.color, lw=1.9, ls=run.linestyle, label=run.label)
        base = next(r for r in RUNS if r.key == "n_sar_base")
        df = data[base.key]
        ax.plot(df["epoch_idx"], roll(det_loss(df)), color=base.color, lw=1.4, alpha=0.55, label=base.label)
        ax.axvline(700, color="#777777", lw=1.0, ls=":", alpha=0.8)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("epoch")
        ax.set_xlim(1, 800)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
    axes[0].set_ylabel("train box+cls+dfl")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path = FIG / "fig2_n_ladd_detector_loss.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aux_losses(data: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex=True)
    cols = [
        ("train/kd_loss", "KD"),
        ("train/s_rec_loss", "student rec"),
        ("train/r_aux_loss", "residual aux"),
        ("train/u_aux_loss", "teacher private aux"),
    ]
    selected = [r for r in RUNS if r.group in {"old_ladd", "current_ladd"} and r.seed in {0, 42, 123}]
    for ax, (col, title) in zip(axes.flat, cols):
        plotted = False
        for run in selected:
            df = data[run.key]
            y = maybe(df, col)
            if y is None:
                continue
            alpha = 0.95 if run.group == "current_ladd" else 0.55
            ax.plot(df["epoch_idx"], roll(y), color=run.color, lw=1.5, ls=run.linestyle, alpha=alpha, label=run.label)
            plotted = True
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.set_xlim(1, 800)
        if not plotted:
            ax.text(0.5, 0.5, "unavailable / all zero", transform=ax.transAxes, ha="center", va="center")
    axes[1, 0].set_xlabel("B epoch")
    axes[1, 1].set_xlabel("B epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    if seen:
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    path = FIG / "fig3_n_ladd_aux_losses.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_baselines(data: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, keys, title in [
        (axes[0], ["s_sar_base", "s_rgb_base"], "YOLO11s mosaic baseline"),
        (axes[1], ["m_sar_base", "m_rgb_base"], "YOLO11m mosaic baseline, running"),
    ]:
        for key in keys:
            run = next(r for r in RUNS if r.key == key)
            df = data[key]
            y = ap(df)
            ax.plot(df["epoch_idx"], y, color=run.color, lw=0.7, alpha=0.18)
            ax.plot(df["epoch_idx"], roll(y), color=run.color, lw=2.0, label=run.label)
            best_i = int(y.idxmax())
            ax.scatter(df["epoch_idx"].iloc[best_i], y.iloc[best_i], s=20, color=run.color, zorder=4)
        ax.axvline(700, color="#777777", lw=1.0, ls=":", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_xlim(1, 800)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.legend(frameon=False)
    axes[0].set_ylabel("AP50-95")
    fig.tight_layout()
    path = FIG / "fig4_s_m_baseline_progress.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(data: dict[str, pd.DataFrame]) -> Path:
    rows = []
    for run in RUNS:
        df = data[run.key]
        y = ap(df)
        best_i = int(y.idxmax())
        rows.append(
            {
                "run_key": run.key,
                "label": run.label,
                "group": run.group,
                "model": run.model,
                "seed": run.seed,
                "epochs_available": int(df["epoch_idx"].iloc[-1]),
                "best_ap50_95": float(y.iloc[best_i]),
                "best_epoch": int(df["epoch_idx"].iloc[best_i]),
                "last_ap50_95": float(y.iloc[-1]),
                "det_loss_last": float(det_loss(df).iloc[-1]),
            }
        )
    path = ROOT / "mosaic_progress_summary_20260616.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    data = {run.key: read_run(run) for run in RUNS}
    figures = [
        plot_n_ap(data),
        plot_n_detector_loss(data),
        plot_aux_losses(data),
        plot_baselines(data),
    ]
    summary = write_summary(data)
    print(f"summary={summary}")
    for figure in figures:
        print(f"figure={figure}")


if __name__ == "__main__":
    main()
