#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    rel: str
    group: str
    seed: int | None = None
    entry: str = ""
    color: str = ""
    linestyle: str = "-"


RUNS = [
    RunSpec(
        "old_ladd_s0",
        "old LADD cap2 s0, A2-best -> B",
        "mnt/dataY/ydf/projects/LADD_og/legacy_results_archive/pre_formal_nomosaic_20260528/runs_public/ogsod/hbb/ladd_converged_20260524/ladd_hbb_ogsod11n_ladd800r2_cap2_s0_b_e800_b64_s0_gpu4/results.csv",
        "old_ladd",
        0,
        "A2 best",
        "#1f77b4",
    ),
    RunSpec(
        "old_ladd_s42",
        "old LADD cap2 s42, A2-best -> B",
        "mnt/dataY/ydf/projects/LADD_og/legacy_results_archive/pre_formal_nomosaic_20260528/runs_public/ogsod/hbb/ladd_converged_20260524/ladd_hbb_ogsod11n_ladd800r2_cap2_s42_b_e800_b64_s42_gpu3/results.csv",
        "old_ladd",
        42,
        "A2 best",
        "#1f77b4",
    ),
    RunSpec(
        "old_ladd_s123",
        "old LADD cap2 s123, A2-best -> B",
        "mnt/dataY/ydf/projects/LADD_og/legacy_results_archive/pre_formal_nomosaic_20260528/runs_public/ogsod/hbb/ladd_converged_20260524/ladd_hbb_ogsod11n_ladd800r2_cap2_s123_b_e800_b64_s123_gpu5/results.csv",
        "old_ladd",
        123,
        "A2 best",
        "#1f77b4",
    ),
    RunSpec(
        "new_ladd_s0",
        "current s0, A2-last -> B",
        "mnt/dataY/ydf/projects/LADD_public/runs_public/ogsod/hbb/ladd_mosaic_a2last_20260615/ladd_hbb_ogsod11n_mosaic_a2last_cap2_s0_b_e800_b64_s0_gpu1/results.csv",
        "new_ladd",
        0,
        "A2 last",
        "#d62728",
    ),
    RunSpec(
        "new_ladd_s42",
        "current s42, A1-best skip A2 -> B",
        "mnt/dataY/ydf/projects/LADD_public/runs_public/ogsod/hbb/ladd_mosaic_a2last_20260615/ladd_hbb_ogsod11n_mosaic_a1best_skipa2_cap2_s42_b_e800_b64_s42_gpu1/results.csv",
        "new_ladd",
        42,
        "A1 best, skip A2",
        "#d62728",
    ),
    RunSpec(
        "new_ladd_s123",
        "current s123, A2-last -> B",
        "mnt/dataY/ydf/projects/LADD_public/runs_public/ogsod/hbb/ladd_mosaic_a2last_20260615/ladd_hbb_ogsod11n_mosaic_a2last_cap2_s123_b_e800_b64_s123_gpu3/results.csv",
        "new_ladd",
        123,
        "A2 last",
        "#d62728",
    ),
    RunSpec(
        "baseline_n_sar",
        "YOLO11n SAR baseline",
        "mnt/dataY/ydf/projects/LADD_og/legacy_results_archive/pre_formal_nomosaic_20260528/runs_public/ogsod/hbb/baseline_controls/cos_closeAt100_E800_20260524/sar_yolo11n_hbb_800ep_cos_closeAt100_pat80_s0_gpu4/results.csv",
        "baseline",
        None,
        "detector from scratch",
        "#333333",
        "--",
    ),
    RunSpec(
        "baseline_n_rgb",
        "YOLO11n RGB baseline",
        "mnt/dataY/ydf/projects/LADD_og/legacy_results_archive/pre_formal_nomosaic_20260528/runs_public/ogsod/hbb/baseline_controls/cos_closeAt100_E800_20260524/rgb_yolo11n_hbb_800ep_cos_closeAt100_pat80_s0_gpu5/results.csv",
        "baseline",
        None,
        "detector from scratch",
        "#9467bd",
        "--",
    ),
    RunSpec(
        "baseline_s_sar",
        "YOLO11s SAR baseline, current running",
        "mnt/dataY/ydf/projects/LADD_public/runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/sar_yolo11s_hbb_mosaicE800_closeAt100_s0_gpu1_20260615/results.csv",
        "baseline_s",
        None,
        "detector from scratch",
        "#333333",
        "-",
    ),
    RunSpec(
        "baseline_s_rgb",
        "YOLO11s RGB baseline, current running",
        "mnt/dataY/ydf/projects/LADD_public/runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/rgb_yolo11s_hbb_mosaicE800_closeAt100_s0_gpu1_20260615/results.csv",
        "baseline_s",
        None,
        "detector from scratch",
        "#9467bd",
        "-",
    ),
]


def read_run(raw_root: Path, spec: RunSpec) -> pd.DataFrame:
    path = raw_root / spec.rel
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["epoch_idx"] = range(1, len(df) + 1)
    df["run_key"] = spec.key
    df["label"] = spec.label
    return df


def metric_col(df: pd.DataFrame, needle: str) -> str:
    matches = [c for c in df.columns if needle in c]
    if not matches:
        raise KeyError(f"Missing column containing {needle}; columns={list(df.columns)}")
    return matches[-1]


def ap_series(df: pd.DataFrame) -> pd.Series:
    return df[metric_col(df, "mAP50-95")].astype(float)


def det_loss(df: pd.DataFrame) -> pd.Series:
    cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    existing = [c for c in cols if c in df.columns]
    return df[existing].astype(float).sum(axis=1)


def maybe_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    s = df[col].astype(float)
    if (s.abs().sum() == 0) or s.isna().all():
        return None
    return s


def rolling(y: pd.Series, window: int = 11) -> pd.Series:
    return y.rolling(window=window, min_periods=1, center=True).mean()


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


def plot_seedwise_ap(data: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4), sharey=True)
    seeds = [0, 42, 123]
    baseline_specs = [s for s in RUNS if s.key in {"baseline_n_sar", "baseline_n_rgb"}]
    for ax, seed in zip(axes, seeds):
        for spec in baseline_specs:
            df = data[spec.key]
            y = ap_series(df)
            ax.plot(df["epoch_idx"], rolling(y, 13), color=spec.color, linestyle=spec.linestyle, lw=1.4, alpha=0.65, label=spec.label)
        for spec in [s for s in RUNS if s.seed == seed and s.group in {"old_ladd", "new_ladd"}]:
            df = data[spec.key]
            y = ap_series(df)
            ax.plot(df["epoch_idx"], y, color=spec.color, alpha=0.18, lw=0.8)
            ax.plot(df["epoch_idx"], rolling(y, 13), color=spec.color, linestyle=spec.linestyle, lw=2.0, label=spec.label)
            best_idx = int(y.idxmax())
            ax.scatter([df["epoch_idx"].iloc[best_idx]], [y.iloc[best_idx]], color=spec.color, s=22, zorder=4)
        ax.axvline(700, color="#777777", ls=":", lw=1.0, alpha=0.75)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("epoch (LADD: B-stage epoch; baseline: detector epoch)")
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.set_xlim(1, 800)
    axes[0].set_ylabel("AP50-95")
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    handles3, labels3 = axes[2].get_legend_handles_labels()
    all_h = handles + handles2 + handles3
    all_l = labels + labels2 + labels3
    seen = {}
    for h, l in zip(all_h, all_l):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("YOLO11n mosaic protocol: old LADD mainline vs current B-entry variants with baselines", y=1.03)
    fig.tight_layout()
    path = out_dir / "fig_n_seedwise_ap_with_baselines.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overview(data: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for spec in RUNS:
        if spec.group not in {"old_ladd", "new_ladd", "baseline"}:
            continue
        df = data[spec.key]
        y = ap_series(df)
        alpha = 0.9 if spec.group != "baseline" else 0.55
        lw = 2.0 if spec.group != "baseline" else 1.7
        label = spec.label
        if spec.group == "old_ladd":
            ls = "-"
        elif spec.group == "new_ladd":
            ls = "--"
        else:
            ls = spec.linestyle
        ax.plot(df["epoch_idx"], rolling(y, 13), color=spec.color, linestyle=ls, lw=lw, alpha=alpha, label=label)
    ax.axvline(700, color="#777777", ls=":", lw=1.0, label="close_mosaic=700")
    ax.set_xlabel("epoch")
    ax.set_ylabel("AP50-95")
    ax.set_title("Full overview, smoothed AP curves (11-epoch centered rolling mean)")
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_xlim(1, 800)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = out_dir / "fig_n_all_ap_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_detection_loss(data: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4), sharey=True)
    for ax, seed in zip(axes, [0, 42, 123]):
        for spec in [s for s in RUNS if s.seed == seed and s.group in {"old_ladd", "new_ladd"}]:
            df = data[spec.key]
            ax.plot(df["epoch_idx"], rolling(det_loss(df), 13), color=spec.color, lw=2.0, label=spec.label)
        df_sar = data["baseline_n_sar"]
        ax.plot(df_sar["epoch_idx"], rolling(det_loss(df_sar), 13), color="#333333", lw=1.5, ls="--", alpha=0.65, label="YOLO11n SAR baseline")
        ax.axvline(700, color="#777777", ls=":", lw=1.0, alpha=0.75)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("epoch")
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.set_xlim(1, 800)
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
    fig.suptitle("Detection loss comparison under the same mosaic schedule", y=1.03)
    fig.tight_layout()
    path = out_dir / "fig_n_detection_loss_compare.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aux(data: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    aux_cols = [
        ("train/kd_loss", "KD loss"),
        ("train/s_rec_loss", "student reconstruction loss"),
        ("train/r_aux_loss", "residual auxiliary loss"),
        ("train/u_aux_loss", "teacher private auxiliary loss"),
    ]
    specs = [s for s in RUNS if s.group in {"old_ladd", "new_ladd"} and s.seed in {0, 42}]
    for ax, (col, title) in zip(axes.flat, aux_cols):
        plotted = False
        for spec in specs:
            df = data[spec.key]
            y = maybe_series(df, col)
            if y is None:
                continue
            ls = "-" if spec.group == "old_ladd" else "--"
            ax.plot(df["epoch_idx"], rolling(y, 13), color=spec.color, ls=ls, lw=1.8, alpha=0.9, label=spec.label)
            plotted = True
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.set_xlim(1, 800)
        if not plotted:
            ax.text(0.5, 0.5, "all zero / unavailable", transform=ax.transAxes, ha="center", va="center")
    axes[1, 0].set_xlabel("B-stage epoch")
    axes[1, 1].set_xlabel("B-stage epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("LADD auxiliary losses, old mainline vs current B-entry variants", y=1.01)
    fig.tight_layout()
    path = out_dir / "fig_n_ladd_aux_losses.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_s_baseline(data: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color in [("baseline_s_sar", "#333333"), ("baseline_s_rgb", "#9467bd")]:
        spec = next(s for s in RUNS if s.key == key)
        df = data[key]
        y = ap_series(df)
        ax.plot(df["epoch_idx"], y, color=color, alpha=0.2, lw=0.8)
        ax.plot(df["epoch_idx"], rolling(y, 13), color=color, lw=2.0, label=spec.label)
        best_idx = int(y.idxmax())
        ax.scatter([df["epoch_idx"].iloc[best_idx]], [y.iloc[best_idx]], color=color, s=24, zorder=4)
    ax.axvline(700, color="#777777", ls=":", lw=1.0, label="close_mosaic=700")
    ax.set_xlabel("epoch")
    ax.set_ylabel("AP50-95")
    ax.set_title("YOLO11s current mosaic baseline reference")
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_xlim(1, 800)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "fig_s_baseline_reference.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(data: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    rows = []
    for spec in RUNS:
        df = data[spec.key]
        y = ap_series(df)
        best_idx = int(y.idxmax())
        rows.append(
            {
                "run_key": spec.key,
                "label": spec.label,
                "group": spec.group,
                "seed": spec.seed,
                "entry": spec.entry,
                "epochs_available": len(df),
                "best_ap50_95": y.iloc[best_idx],
                "best_epoch": int(df["epoch_idx"].iloc[best_idx]),
                "last_ap50_95": y.iloc[-1],
                "last_epoch": int(df["epoch_idx"].iloc[-1]),
                "det_loss_last": det_loss(df).iloc[-1],
            }
        )
    path = out_dir / "ladd_mosaic_protocol_compare_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_report(out_dir: Path, figures: list[Path], summary_path: Path) -> Path:
    summary = pd.read_csv(summary_path)
    report = out_dir / "LADD_MOSAIC_PROTOCOL_CURVE_COMPARE_20260615_CN.md"
    table = summary[
        [
            "run_key",
            "group",
            "seed",
            "entry",
            "epochs_available",
            "best_ap50_95",
            "best_epoch",
            "last_ap50_95",
        ]
    ].copy()
    for col in ["best_ap50_95", "last_ap50_95"]:
        table[col] = table[col].map(lambda x: f"{x:.5f}")
    lines = [
        "# LADD mosaic protocol curve comparison, 2026-06-15",
        "",
        "协议匹配项：epochs=800, batch=64, imgsz=256, mosaic=1.0, close_mosaic=700, optimizer=auto, lr0=0.01, lrf=0.01, cos_lr=true, warmup_epochs=3.0, warmup_bias_lr=0.1。",
        "",
        "注意：baseline 曲线是 detector 从头训练；LADD 曲线是 B 阶段训练。因此同图主要用于观察同一 mosaic/cosine 协议下的收敛形态，不应把横轴直接解释为相同训练阶段。",
        "",
        "## Summary",
        "",
        table.to_markdown(index=False),
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        lines.append(f"![{fig.stem}](figures/{fig.name})")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    style()
    data = {spec.key: read_run(args.raw_root, spec) for spec in RUNS}
    figs = [
        plot_seedwise_ap(data, fig_dir),
        plot_overview(data, fig_dir),
        plot_detection_loss(data, fig_dir),
        plot_aux(data, fig_dir),
        plot_s_baseline(data, fig_dir),
    ]
    summary = write_summary(data, args.out_dir)
    report = write_report(args.out_dir, figs, summary)
    print(f"summary={summary}")
    print(f"report={report}")
    for fig in figs:
        print(f"figure={fig}")


if __name__ == "__main__":
    main()
