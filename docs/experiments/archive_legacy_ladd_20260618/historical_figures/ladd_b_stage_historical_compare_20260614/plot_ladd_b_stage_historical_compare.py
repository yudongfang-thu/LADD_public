#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
AP_COL = "metrics/mAP50-95(B)"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    model: str
    family: str
    path: str
    color: str
    linestyle: str = "-"
    linewidth: float = 1.8
    alpha: float = 1.0


CURRENT_N = [
    RunSpec(
        "N1_current_base_cont",
        "N1 current: SAR baseline cont. B100",
        "n",
        "current",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/N1_basecontinue_b100/run_files/results.csv",
        "#1f77b4",
        "-",
        2.2,
    ),
    RunSpec(
        "N2_current_a2best_cont",
        "N2 current: A2-best cont. B100",
        "n",
        "current",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/N2_a2best_continue_b100/run_files/results.csv",
        "#ff7f0e",
        "-",
        2.2,
    ),
    RunSpec(
        "N3_current_sarbase_a2last_decomp",
        "N3 current: SAR-base + A2-last decomp B100",
        "n",
        "current",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/N3_base_a2last_decomp_b100/run_files/results.csv",
        "#2ca02c",
        "-",
        2.2,
    ),
    RunSpec(
        "N4_current_kd_ramp",
        "N4 current: N3 + KD ramp B120",
        "n",
        "current",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/N4_base_a2last_kdramp_b120/run_files/results.csv",
        "#d62728",
        "-",
        2.2,
    ),
]

CURRENT_S = [
    RunSpec(
        "S1_current_base_cont",
        "S1 current: SAR baseline cont. B100",
        "s",
        "current",
        "ladd/results/b_entrance_20260613/evidence/autodl/S1_basecontinue_b100_autodl/run_files/results.csv",
        "#1f77b4",
        "-",
        2.2,
    ),
    RunSpec(
        "S2_current_a2best_cont",
        "S2 current: A2-best cont. B100",
        "s",
        "current",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/S2_a2best_continue_b100_retry2_running/run_files/results.csv",
        "#ff7f0e",
        "-",
        2.2,
    ),
    RunSpec(
        "S3_current_sarbase_a2last_decomp",
        "S3 current: SAR-base + A2-last decomp B100",
        "s",
        "current",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/S3_base_a2last_decomp_b100/run_files/results.csv",
        "#2ca02c",
        "-",
        2.2,
    ),
    RunSpec(
        "S4_current_kd_ramp",
        "S4 current: S3 + KD ramp B120",
        "s",
        "current",
        "docs/experiments/figures/ladd_b_entrance_trends_20260614/source/S4_latest_results.csv",
        "#d62728",
        "-",
        2.2,
    ),
]

HISTORICAL_N = [
    RunSpec(
        "n_mosaic100_cap2_s0",
        "old n: mosaic100 cap2 s0 B800",
        "n",
        "old_mosaic",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/ladd_hbb_ogsod11n_ladd800r2_cap2_s0_b_e800_b64_s0_gpu4/results.csv",
        "#17becf",
        "--",
    ),
    RunSpec(
        "n_mosaic100_legacy_s0",
        "old n: mosaic100 legacy s0 B755",
        "n",
        "old_mosaic",
        "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/ladd_hbb_ogsod11n_ladd800r2_legacy_s0_b_e800_b64_s0_gpu2/results.csv",
        "#bcbd22",
        "--",
    ),
    RunSpec(
        "n_nomosaic_cap2_s0_no_bnfreeze",
        "old n: no-mosaic cap2 s0 no-BN-freeze B800",
        "n",
        "old_nomosaic",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu6/results.csv",
        "#9467bd",
        "-.",
    ),
    RunSpec(
        "n_nomosaic_cap2_s0_bnfreeze",
        "old n: no-mosaic cap2 s0 BN-freeze B800",
        "n",
        "old_nomosaic",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_bnfreeze1e3_90_gpu7_b_e800_b64_s0_gpu7/results.csv",
        "#8c564b",
        "-.",
    ),
    RunSpec(
        "n_nomosaic_s123_old_crash",
        "old n: no-mosaic s123 old-B crash",
        "n",
        "old_crash",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_b_e800_b64_s123_gpu5/results.csv",
        "#7f7f7f",
        ":",
        1.4,
        0.8,
    ),
    RunSpec(
        "n_nomosaic_s123_bstable_late_reg",
        "old n: no-mosaic s123 B-lr1e-3 late-reg.",
        "n",
        "old_late_regression",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_bstable1e3_b_e800_b64_s123_gpu2/results.csv",
        "#e377c2",
        ":",
        1.5,
        0.85,
    ),
]

HISTORICAL_S = [
    RunSpec(
        "s_nomosaic_cap2_s0_no_bnfreeze",
        "old s: no-mosaic cap2 s0 no-BN-freeze B608",
        "s",
        "old_nomosaic",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/s_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu5/results.csv",
        "#9467bd",
        "-.",
    ),
    RunSpec(
        "s_nomosaic_cap2_s0_bnfreeze",
        "old s: no-mosaic cap2 s0 BN-freeze B800",
        "s",
        "old_nomosaic",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/s_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s0_gpu1/results.csv",
        "#8c564b",
        "-.",
    ),
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_run(spec: RunSpec) -> pd.DataFrame:
    path = REPO / spec.path
    if not path.exists():
        raise FileNotFoundError(path)
    df = clean_columns(pd.read_csv(path))
    if AP_COL not in df.columns:
        raise KeyError(f"{AP_COL} missing from {path}")
    df = df[["epoch", AP_COL, "train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"]].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df[AP_COL] = pd.to_numeric(df[AP_COL], errors="coerce")
    df = df.dropna(subset=["epoch", AP_COL])
    df["run"] = spec.key
    df["label"] = spec.label
    df["model"] = spec.model
    df["family"] = spec.family
    df["source"] = spec.path
    return df


def linear_slope(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y.to_numpy(dtype=float), deg=1)[0])


def summarize(spec: RunSpec, df: pd.DataFrame) -> dict[str, object]:
    ap = pd.to_numeric(df[AP_COL], errors="coerce")
    best_idx = int(ap.idxmax())
    best = float(ap.loc[best_idx])
    best_epoch = int(df.loc[best_idx, "epoch"])
    last = float(ap.iloc[-1])
    last_epoch = int(df["epoch"].iloc[-1])
    first = float(ap.iloc[0])
    first120 = df[df["epoch"] <= 120]
    first120_ap = pd.to_numeric(first120[AP_COL], errors="coerce")
    best120 = float(first120_ap.max()) if len(first120_ap) else float("nan")
    best120_epoch = int(first120.loc[first120_ap.idxmax(), "epoch"]) if len(first120_ap) else -1
    return {
        "key": spec.key,
        "label": spec.label,
        "model": spec.model,
        "family": spec.family,
        "epochs": len(df),
        "first_epoch_ap": first,
        "best": best,
        "best_epoch": best_epoch,
        "last": last,
        "last_epoch": last_epoch,
        "best_final_drop": best - last,
        "best_first120": best120,
        "best_first120_epoch": best120_epoch,
        "last20_slope": linear_slope(ap.tail(20)),
        "last50_slope": linear_slope(ap.tail(50)),
        "last100_slope": linear_slope(ap.tail(100)),
        "source": spec.path,
    }


def plot_ap_panels(specs: list[RunSpec], panel_prefix: str, filename: str) -> None:
    data = {spec.key: load_run(spec) for spec in specs}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4), sharey=False)
    for ax, xmax, label_suffix in [(axes[0], 820, "full B stage"), (axes[1], 125, "first 120 epochs")]:
        for spec in specs:
            df = data[spec.key]
            sub = df[df["epoch"] <= xmax]
            if sub.empty:
                continue
            ax.plot(
                sub["epoch"],
                sub[AP_COL],
                label=spec.label,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )
        ax.set_xlabel("B-stage epoch")
        ax.set_ylabel("AP50-95")
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
        ax.set_xlim(1, xmax)
        ax.text(0.01, 0.98, f"{panel_prefix}: {label_suffix}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close(fig)


def plot_loss_zoom(specs: list[RunSpec], filename: str) -> None:
    selected = [
        spec
        for spec in specs
        if spec.key
        in {
            "N1_current_base_cont",
            "N3_current_sarbase_a2last_decomp",
            "N4_current_kd_ramp",
            "n_nomosaic_cap2_s0_no_bnfreeze",
            "n_mosaic100_cap2_s0",
            "s_nomosaic_cap2_s0_bnfreeze",
            "S3_current_sarbase_a2last_decomp",
            "S4_current_kd_ramp",
        }
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.1))
    for ax, loss_prefix in [(axes[0], "train"), (axes[1], "val")]:
        for spec in selected:
            df = load_run(spec)
            sub = df[df["epoch"] <= 120].copy()
            for col in [f"{loss_prefix}/box_loss", f"{loss_prefix}/cls_loss", f"{loss_prefix}/dfl_loss"]:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub["det_loss"] = sub[[f"{loss_prefix}/box_loss", f"{loss_prefix}/cls_loss", f"{loss_prefix}/dfl_loss"]].sum(axis=1)
            ax.plot(
                sub["epoch"],
                sub["det_loss"],
                label=spec.label,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )
        ax.set_xlabel("B-stage epoch")
        ax.set_ylabel(f"{loss_prefix} box+cls+dfl")
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
        ax.set_xlim(1, 120)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close(fig)


def write_summary(specs: list[RunSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        df = load_run(spec)
        rows.append(summarize(spec, df))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "b_stage_historical_compare_summary_20260614.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return out


def write_markdown(summary: pd.DataFrame) -> None:
    lines = [
        "# LADD B 阶段历史曲线对比（2026-06-14）",
        "",
        "本地报告对比当前 B-entrance 诊断实验和之前 LADD 主线 B 阶段记录。",
        "",
        "关键读法：",
        "- 当前 B-entrance 实验从收敛 baseline 或 selected checkpoint 出发，所以 epoch 1 AP 已经接近 SAR baseline 区间。",
        "- 历史 mosaic100 LADD B 从更低的 A2 末端开始，在 B 阶段长期恢复；它的上升曲线不能直接等价于从收敛 detector 做 B-only continuation。",
        "- 历史 no-mosaic n 主线说明完整 A1/A2/B 链条下 B 仍有长程增长，尤其和当前 split-load 设置不是同一个实验。",
        "- 历史 s BN-freeze 的 best 为正，但 best-final drop 很大；这和当前 s 入口实验的担心一致：不崩溃也可能 late-regress。",
        "",
        "生成图：",
        "- `fig1_n_current_vs_historical_b_ap.png/pdf`",
        "- `fig2_s_current_vs_historical_b_ap.png/pdf`",
        "- `fig3_selected_b_loss_zoom_120.png/pdf`",
        "",
        "汇总表：",
        "",
        summary[
            [
                "label",
                "model",
                "family",
                "epochs",
                "first_epoch_ap",
                "best",
                "best_epoch",
                "last",
                "best_final_drop",
                "best_first120",
                "best_first120_epoch",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
    ]
    (OUT_DIR / "README_LADD_B_STAGE_HISTORICAL_COMPARE_20260614_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_n = CURRENT_N + HISTORICAL_N
    all_s = CURRENT_S + HISTORICAL_S
    summary = write_summary(all_n + all_s)
    plot_ap_panels(all_n, "YOLO11n", "fig1_n_current_vs_historical_b_ap")
    plot_ap_panels(all_s, "YOLO11s", "fig2_s_current_vs_historical_b_ap")
    plot_loss_zoom(all_n + all_s, "fig3_selected_b_loss_zoom_120")
    write_markdown(summary)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
