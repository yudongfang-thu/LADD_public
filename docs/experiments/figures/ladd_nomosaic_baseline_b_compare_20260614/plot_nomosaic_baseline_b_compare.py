#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
AP_COL = "metrics/mAP50-95(B)"


@dataclass(frozen=True)
class Curve:
    key: str
    label: str
    model: str
    kind: str
    path: str
    color: str
    linestyle: str = "-"
    linewidth: float = 1.9
    alpha: float = 1.0


BASELINES = [
    Curve(
        "sar_n_s0_baseline",
        "SAR n seed0 baseline 800ep",
        "n",
        "baseline",
        "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv",
        "#1f77b4",
        "-",
        2.4,
    ),
    Curve(
        "sar_s_s0_baseline",
        "SAR s seed0 baseline 800ep",
        "s",
        "baseline",
        "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s0_b64.csv",
        "#1f77b4",
        "-",
        2.4,
    ),
]

N_CURVES = [
    Curve(
        "n_baseline",
        "SAR n baseline train 800ep",
        "n",
        "baseline",
        "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv",
        "#1f77b4",
        "-",
        2.5,
    ),
    Curve(
        "n_old_cap2_s0_no_bnfreeze",
        "old no-mosaic n cap2 s0 no-BN-freeze B800",
        "n",
        "old_ladd_b",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu6/results.csv",
        "#9467bd",
        "-.",
    ),
    Curve(
        "n_old_cap2_s0_bnfreeze",
        "old no-mosaic n cap2 s0 BN-freeze B800",
        "n",
        "old_ladd_b",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_bnfreeze1e3_90_gpu7_b_e800_b64_s0_gpu7/results.csv",
        "#8c564b",
        "--",
    ),
    Curve(
        "n_old_cap2_s123_crash",
        "old no-mosaic n s123 old-B crash",
        "n",
        "old_ladd_b",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_b_e800_b64_s123_gpu5/results.csv",
        "#7f7f7f",
        ":",
        1.4,
        0.75,
    ),
    Curve(
        "n_current_n1",
        "current N1 SAR baseline cont. B100",
        "n",
        "current_b_entrance",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/N1_basecontinue_b100/run_files/results.csv",
        "#2ca02c",
        "-",
        2.2,
    ),
    Curve(
        "n_current_n3",
        "current N3 SAR-base + A2-last decomp B100",
        "n",
        "current_b_entrance",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/N3_base_a2last_decomp_b100/run_files/results.csv",
        "#ff7f0e",
        "-",
        2.2,
    ),
    Curve(
        "n_current_n4",
        "current N4 N3 + KD ramp B120",
        "n",
        "current_b_entrance",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/N4_base_a2last_kdramp_b120/run_files/results.csv",
        "#d62728",
        "-",
        2.2,
    ),
]

S_CURVES = [
    Curve(
        "s_baseline",
        "SAR s baseline train 800ep",
        "s",
        "baseline",
        "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s0_b64.csv",
        "#1f77b4",
        "-",
        2.5,
    ),
    Curve(
        "s_old_cap2_s0_no_bnfreeze",
        "old no-mosaic s cap2 s0 no-BN-freeze B608",
        "s",
        "old_ladd_b",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/s_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu5/results.csv",
        "#9467bd",
        "-.",
    ),
    Curve(
        "s_old_cap2_s0_bnfreeze",
        "old no-mosaic s cap2 s0 BN-freeze B800",
        "s",
        "old_ladd_b",
        "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/s_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s0_gpu1/results.csv",
        "#8c564b",
        "--",
    ),
    Curve(
        "s_current_s1",
        "current S1 SAR baseline cont. B100",
        "s",
        "current_b_entrance",
        "ladd/results/b_entrance_20260613/evidence/autodl/S1_basecontinue_b100_autodl/run_files/results.csv",
        "#2ca02c",
        "-",
        2.2,
    ),
    Curve(
        "s_current_s3",
        "current S3 SAR-base + A2-last decomp B100",
        "s",
        "current_b_entrance",
        "ladd/results/b_entrance_20260613/evidence/ladd4090/S3_base_a2last_decomp_b100/run_files/results.csv",
        "#ff7f0e",
        "-",
        2.2,
    ),
    Curve(
        "s_current_s4",
        "current S4 S3 + KD ramp B120",
        "s",
        "current_b_entrance",
        "docs/experiments/figures/ladd_b_entrance_trends_20260614/source/S4_latest_results.csv",
        "#d62728",
        "-",
        2.2,
    ),
]


def read_curve(curve: Curve) -> pd.DataFrame:
    path = REPO / curve.path
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if AP_COL not in df:
        raise KeyError(f"{AP_COL} missing in {path}")
    keep = ["epoch", AP_COL, "train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df[AP_COL] = pd.to_numeric(df[AP_COL], errors="coerce")
    df = df.dropna(subset=["epoch", AP_COL])
    df["key"] = curve.key
    df["label"] = curve.label
    df["model"] = curve.model
    df["kind"] = curve.kind
    df["source"] = curve.path
    return df


def slope(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return float("nan")
    x = np.arange(len(series), dtype=float)
    return float(np.polyfit(x, series.to_numpy(dtype=float), 1)[0])


def summarize(curve: Curve, df: pd.DataFrame) -> dict[str, object]:
    ap = pd.to_numeric(df[AP_COL], errors="coerce")
    best_idx = int(ap.idxmax())
    last = float(ap.iloc[-1])
    first120 = df[df["epoch"] <= 120]
    first120_ap = pd.to_numeric(first120[AP_COL], errors="coerce")
    return {
        "key": curve.key,
        "label": curve.label,
        "model": curve.model,
        "kind": curve.kind,
        "epochs": len(df),
        "first_ap": float(ap.iloc[0]),
        "best": float(ap.loc[best_idx]),
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "last": last,
        "last_epoch": int(df["epoch"].iloc[-1]),
        "best_final_drop": float(ap.loc[best_idx] - last),
        "best_first120": float(first120_ap.max()) if len(first120_ap) else float("nan"),
        "best_first120_epoch": int(first120.loc[first120_ap.idxmax(), "epoch"]) if len(first120_ap) else -1,
        "last100_slope": slope(ap.tail(100)),
        "source": curve.path,
    }


def add_baseline_guides(ax: plt.Axes, model: str) -> None:
    if model == "n":
        best, final = 0.55654, 0.55127
    elif model == "s":
        best, final = 0.62897, 0.62233
    else:
        return
    ax.axhline(best, color="#222222", linewidth=1.0, linestyle=":", alpha=0.65)
    ax.axhline(final, color="#555555", linewidth=1.0, linestyle=(0, (3, 2)), alpha=0.55)
    ax.text(0.99, best, "SAR best", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    ax.text(0.99, final, "SAR final", transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=8)


def plot_model(curves: list[Curve], model: str, out_name: str) -> None:
    loaded = {curve.key: read_curve(curve) for curve in curves}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4), sharey=False)
    for ax, xmax, panel in [(axes[0], 820, "full available record"), (axes[1], 125, "first 120 epochs")]:
        for curve in curves:
            df = loaded[curve.key]
            sub = df[df["epoch"] <= xmax]
            if sub.empty:
                continue
            ax.plot(
                sub["epoch"],
                sub[AP_COL],
                label=curve.label,
                color=curve.color,
                linestyle=curve.linestyle,
                linewidth=curve.linewidth,
                alpha=curve.alpha,
            )
        add_baseline_guides(ax, model)
        ax.set_xlim(1, xmax)
        ax.set_xlabel("epoch within own stage/protocol")
        ax.set_ylabel("AP50-95")
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
        ax.text(0.01, 0.98, f"YOLO11{model}: {panel}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(OUT_DIR / f"{out_name}.png", dpi=260)
    fig.savefig(OUT_DIR / f"{out_name}.pdf")
    plt.close(fig)


def plot_baseline_only() -> None:
    curves = [
        Curve("sar_n_s0", "SAR n seed0", "n", "baseline", "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv", "#1f77b4"),
        Curve("sar_n_s42", "SAR n seed42", "n", "baseline", "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s42_b64.csv", "#17becf", "--"),
        Curve("sar_n_s123", "SAR n seed123", "n", "baseline", "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s123_b64.csv", "#9467bd", "-."),
        Curve("sar_s_s0", "SAR s seed0", "s", "baseline", "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s0_b64.csv", "#2ca02c"),
        Curve("sar_s_s42", "SAR s seed42", "s", "baseline", "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s42_b64.csv", "#bcbd22", "--"),
        Curve("sar_s_s123", "SAR s seed123", "s", "baseline", "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s123_b64.csv", "#8c564b", "-."),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
    for curve in curves:
        df = read_curve(curve)
        ax.plot(df["epoch"], df[AP_COL], label=curve.label, color=curve.color, linestyle=curve.linestyle, linewidth=1.8)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("AP50-95")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.text(0.01, 0.98, "formal no-mosaic SAR baselines", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig0_formal_nomosaic_sar_baseline_curves.png", dpi=260)
    fig.savefig(OUT_DIR / "fig0_formal_nomosaic_sar_baseline_curves.pdf")
    plt.close(fig)


def write_summary(curves: list[Curve]) -> pd.DataFrame:
    rows = []
    for curve in curves:
        rows.append(summarize(curve, read_curve(curve)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "nomosaic_baseline_b_compare_summary_20260614.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def write_readme(summary: pd.DataFrame) -> None:
    body = [
        "# Formal no-mosaic baseline 与 LADD B 阶段对比（2026-06-14）",
        "",
        "本报告只看 formal no-mosaic 记录，不纳入 mosaic100 历史实验。",
        "",
        "关键读法：",
        "- formal no-mosaic SAR baseline 本身在 800ep 后段存在 best-final gap，因此 final 低于 best 是协议现象的一部分。",
        "- 当前 B-entrance 从收敛/selected checkpoint 出发，epoch1 AP 已经高；它不是从 YOLO 初始权重开始的 baseline 曲线。",
        "- no-mosaic 历史 LADD full-chain B 证明 B 阶段可以在完整链条中继续获得长程增益；当前 split-load B-only 目前没有复现这种增益。",
        "- s 模型历史 BN-freeze B800 与当前 S 曲线共同提示：早期平台期后仍可能出现 late-regression，B100/B120 只看入口不够完整。",
        "",
        "生成图：",
        "- `fig0_formal_nomosaic_sar_baseline_curves.png/pdf`",
        "- `fig1_nomosaic_n_baseline_ladd_b_compare.png/pdf`",
        "- `fig2_nomosaic_s_baseline_ladd_b_compare.png/pdf`",
        "",
        "汇总表：",
        "",
        summary[["label", "model", "kind", "epochs", "first_ap", "best", "best_epoch", "last", "best_final_drop", "best_first120", "best_first120_epoch"]].to_markdown(index=False, floatfmt=".5f"),
        "",
    ]
    (OUT_DIR / "README_NOMOSAIC_BASELINE_B_COMPARE_20260614_CN.md").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_baseline_only()
    plot_model(N_CURVES, "n", "fig1_nomosaic_n_baseline_ladd_b_compare")
    plot_model(S_CURVES, "s", "fig2_nomosaic_s_baseline_ladd_b_compare")
    summary = write_summary(N_CURVES + S_CURVES)
    write_readme(summary)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
