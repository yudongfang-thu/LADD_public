#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = Path(__file__).resolve().parent
REPORT = ROOT / "docs/experiments/LADD_CURRENT_BASELINE_OVERLAY_ANALYSIS_20260614_CN.md"
SUMMARY_CSV = ROOT / "docs/experiments/ladd_current_baseline_overlay_summary_20260614.csv"
MAP_COL = "metrics/mAP50-95(B)"

N_BASE_BEST = 0.55654
N_BASE_FINAL = 0.55127


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    family: str
    b_epochs: int
    source: Path
    note: str = ""


RUNS = [
    RunSpec(
        "current_N1_basebest_B800",
        "B800 N1 SAR-best det-only",
        "current_b800_det",
        800,
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N1_basebest_continue_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "Current baseline-best detector continuation; detection-only.",
    ),
    RunSpec(
        "current_N1_baselast_B800",
        "B800 N1 SAR-last det-only",
        "current_b800_det",
        800,
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N1_baselast_continue_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "Current baseline-last detector continuation; detection-only.",
    ),
    RunSpec(
        "current_N2_a2best_B800",
        "B800 N2 A2-best full LADD",
        "current_b800_ladd",
        800,
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2best_continue_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "A2-best detector+decomposition checkpoint, full LADD B; NaN at 229.",
    ),
    RunSpec(
        "current_N2_a2last_B800",
        "B800 N2 A2-last full LADD",
        "current_b800_ladd",
        800,
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2last_continue_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "A2-last detector+decomposition checkpoint, full LADD B; NaN at 319.",
    ),
    RunSpec(
        "current_N3_yoloinit_decomp_B800",
        "B800 N3 YOLO-init + A2 decomp",
        "current_b800_yoloinit",
        800,
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N3_yoloinit_a2last_decomp_20260614_cfgfix_retry2_b_e800_b64_s0_gpu1/results.csv",
        "Detector from YOLO init, teacher-side decomp from A2.",
    ),
    RunSpec(
        "current_N4_yoloinit_decomp_kdwarm_B800",
        "B800 N4 YOLO-init + decomp KD-warmup",
        "current_b800_yoloinit",
        800,
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N4_yoloinit_a2last_decomp_kdwarmup_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "Detector from YOLO init, teacher-side decomp from A2, KD-only warmup.",
    ),
    RunSpec(
        "prev_N1_base_B100",
        "prev B100 N1 SAR-base det-only",
        "previous_current_baseline",
        100,
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N1_basecontinue_b100/run_files/results.csv",
        "Previous current-baseline continuation, compressed B100 schedule.",
    ),
    RunSpec(
        "prev_N2_a2best_B100",
        "prev B100 N2 A2-best continue",
        "previous_current_baseline",
        100,
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N2_a2best_continue_b100/run_files/results.csv",
        "Previous A2-best continuation, compressed B100 schedule.",
    ),
    RunSpec(
        "prev_N3_base_decomp_B100",
        "prev B100 N3 SAR-base + A2 decomp",
        "previous_current_baseline",
        100,
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N3_base_a2last_decomp_b100/run_files/results.csv",
        "Previous clean split-load target: SAR baseline detector + A2 decomposition.",
    ),
    RunSpec(
        "prev_N4_base_decomp_kdramp_B120",
        "prev B120 N4 SAR-base + A2 decomp KD-ramp",
        "previous_current_baseline",
        120,
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N4_base_a2last_kdramp_b120/run_files/results.csv",
        "Previous split-load plus KD-only ramp, compressed B120 schedule.",
    ),
    RunSpec(
        "repair_N_weakkd025_B200",
        "repair B200 N A2-short13 + weakKD0.25",
        "repair_current_baseline",
        200,
        ROOT
        / "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/n_weakkd0p25_b200/unknown/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_fix_n_s0_a2lr3e4_short13_bweakKD0p25_b200_b_e200_b64_s0_gpu1/results.csv",
        "Repair branch with selected short A2 and weaker B KD.",
    ),
    RunSpec(
        "old90_nomosaic_cap2_s0_noBN_B800",
        "old 90 B800 no-mosaic cap2 s0 no-BN-freeze",
        "old_same_protocol",
        800,
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu6/results.csv",
        "Historical healthy no-mosaic LADD mainline on server 90.",
    ),
    RunSpec(
        "old90_nomosaic_cap2_s0_BNfreeze_B800",
        "old 90 B800 no-mosaic cap2 s0 BN-freeze",
        "old_same_protocol",
        800,
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_bnfreeze1e3_90_gpu7_b_e800_b64_s0_gpu7/results.csv",
        "Historical healthy no-mosaic LADD with BN freeze.",
    ),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "lines.linewidth": 1.45,
        }
    )


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for spec in RUNS:
        if spec.source.exists():
            out[spec.key] = read_results(spec.source)
        else:
            print("missing", spec.key, spec.source)
    return out


def finite_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=["epoch", col])
    sub = df[["epoch", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    return sub[np.isfinite(sub[col])]


def best_last(df: pd.DataFrame, col: str = MAP_COL) -> tuple[float, int, float, int]:
    s = finite_series(df, col)
    if s.empty:
        return math.nan, -1, math.nan, -1
    idx = s[col].idxmax()
    best = float(s.loc[idx, col])
    best_ep = int(s.loc[idx, "epoch"])
    last = float(s.iloc[-1][col])
    last_ep = int(s.iloc[-1]["epoch"])
    return best, best_ep, last, last_ep


def first_bad(df: pd.DataFrame) -> int | None:
    cols = [c for c in [MAP_COL, "train/box_loss", "train/cls_loss", "train/dfl_loss"] if c in df.columns]
    if not cols:
        return None
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(vals)
    if not bad.any():
        return None
    return int(df.iloc[np.where(bad.any(axis=1))[0][0]]["epoch"])


def summarize(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        df = runs.get(spec.key)
        if df is None:
            continue
        best, best_ep, last, last_ep = best_last(df)
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "family": spec.family,
                "b_epochs": spec.b_epochs,
                "recorded_epochs": len(df),
                "last_epoch": int(df["epoch"].dropna().iloc[-1]),
                "first_ap": float(finite_series(df, MAP_COL).iloc[0][MAP_COL]),
                "best_ap": best,
                "best_epoch": best_ep,
                "last_ap": last,
                "last_epoch_finite": last_ep,
                "best_final_drop": best - last if np.isfinite(best) and np.isfinite(last) else math.nan,
                "first_nonfinite_epoch": first_bad(df),
                "source": str(spec.source.relative_to(ROOT)),
                "note": spec.note,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    return df


def style_for(spec: RunSpec) -> tuple[str, str, float, float]:
    colors = {
        "current_b800_det": "#1f77b4",
        "current_b800_ladd": "#d62728",
        "current_b800_yoloinit": "#9467bd",
        "previous_current_baseline": "#2ca02c",
        "repair_current_baseline": "#ff7f0e",
        "old_same_protocol": "#4c4c4c",
    }
    key_linestyles = {
        "current_N1_basebest_B800": "-",
        "current_N1_baselast_B800": (0, (5, 2)),
        "current_N2_a2best_B800": "-",
        "current_N2_a2last_B800": (0, (5, 2)),
        "current_N3_yoloinit_decomp_B800": "-",
        "current_N4_yoloinit_decomp_kdwarm_B800": (0, (5, 2)),
        "prev_N1_base_B100": "--",
        "prev_N2_a2best_B100": (0, (3, 2, 1, 2)),
        "prev_N3_base_decomp_B100": (0, (1, 1)),
        "prev_N4_base_decomp_kdramp_B120": "-.",
        "repair_N_weakkd025_B200": "-.",
        "old90_nomosaic_cap2_s0_noBN_B800": ":",
        "old90_nomosaic_cap2_s0_BNfreeze_B800": (0, (1, 1)),
    }
    alphas = {
        "current_b800_det": 0.95,
        "current_b800_ladd": 0.95,
        "current_b800_yoloinit": 0.82,
        "previous_current_baseline": 0.86,
        "repair_current_baseline": 0.88,
        "old_same_protocol": 0.80,
    }
    linewidths = {
        "current_b800_det": 1.9,
        "current_b800_ladd": 1.9,
        "current_b800_yoloinit": 1.65,
        "previous_current_baseline": 1.55,
        "repair_current_baseline": 1.65,
        "old_same_protocol": 1.8,
    }
    return colors[spec.family], key_linestyles.get(spec.key, "-"), alphas[spec.family], linewidths[spec.family]


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_epoch_overlay(runs: dict[str, pd.DataFrame], specs: list[RunSpec], stem: str, xlim: tuple[int, int] | None = None, include_yoloinit: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for spec in specs:
        if not include_yoloinit and spec.family == "current_b800_yoloinit":
            continue
        df = runs.get(spec.key)
        if df is None:
            continue
        color, ls, alpha, lw = style_for(spec)
        label = spec.label
        ax.plot(
            df["epoch"],
            pd.to_numeric(df[MAP_COL], errors="coerce"),
            label=label,
            color=color,
            linestyle=ls,
            alpha=alpha,
            linewidth=lw,
        )
        bad = first_bad(df)
        if bad is not None and (xlim is None or xlim[0] <= bad <= xlim[1]):
            ax.axvline(bad, color=color, linestyle=":", alpha=0.55)
    ax.axhline(N_BASE_BEST, color="black", linestyle="--", linewidth=1.0, label="SAR n baseline best 0.55654")
    ax.axhline(N_BASE_FINAL, color="gray", linestyle="--", linewidth=1.0, label="SAR n baseline final 0.55127")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("B-stage epoch")
    ax.set_ylabel("AP50-95")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save(fig, stem)


def plot_normalized_overlay(runs: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    for spec in RUNS:
        if spec.family == "current_b800_yoloinit":
            continue
        df = runs.get(spec.key)
        if df is None:
            continue
        color, ls, alpha, lw = style_for(spec)
        x = pd.to_numeric(df["epoch"], errors="coerce") / float(spec.b_epochs)
        y = pd.to_numeric(df[MAP_COL], errors="coerce")
        ax.plot(x, y, label=spec.label, color=color, linestyle=ls, alpha=alpha, linewidth=lw)
    ax.axhline(N_BASE_BEST, color="black", linestyle="--", linewidth=1.0, label="SAR n baseline best")
    ax.axhline(N_BASE_FINAL, color="gray", linestyle="--", linewidth=1.0, label="SAR n baseline final")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("B-stage normalized progress (epoch / configured B epochs)")
    ax.set_ylabel("AP50-95")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save(fig, "n_current_baseline_overlay_normalized_progress")


def plot_losses(runs: dict[str, pd.DataFrame]) -> None:
    keys = [
        "current_N1_baselast_B800",
        "current_N2_a2last_B800",
        "prev_N2_a2best_B100",
        "prev_N3_base_decomp_B100",
        "repair_N_weakkd025_B200",
        "old90_nomosaic_cap2_s0_BNfreeze_B800",
    ]
    labels = {spec.key: spec.label for spec in RUNS}
    spec_by_key = {spec.key: spec for spec in RUNS}
    cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.8), sharex=True)
    for ax, col in zip(axes, cols):
        for key in keys:
            df = runs.get(key)
            spec = spec_by_key[key]
            if df is None or col not in df.columns:
                continue
            color, ls, alpha, lw = style_for(spec)
            ax.plot(
                df["epoch"],
                pd.to_numeric(df[col], errors="coerce"),
                label=labels[key],
                color=color,
                linestyle=ls,
                alpha=alpha,
                linewidth=lw,
            )
        ax.set_ylabel(col.replace("train/", ""))
    axes[-1].set_xlabel("B-stage epoch")
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.38))
    save(fig, "n_current_baseline_overlay_detector_losses")


def markdown_table(df: pd.DataFrame) -> str:
    view = df[
        [
            "key",
            "family",
            "b_epochs",
            "recorded_epochs",
            "best_ap",
            "best_epoch",
            "last_ap",
            "last_epoch_finite",
            "first_nonfinite_epoch",
        ]
    ].copy()
    for col in ["best_ap", "last_ap"]:
        view[col] = view[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.5f}")
    return view.to_markdown(index=False)


def write_report(summary: pd.DataFrame) -> None:
    text = f"""# LADD 当前 baseline 叠加曲线分析（2026-06-14）

本报告把当前 B800 重启批次、之前在当前 SAR baseline/A2 入口上做过的 LADD B 入口实验，以及 90 服务器 no-mosaic 健康主线叠加在一起。所有曲线都来自轻量 `results.csv`，未使用 checkpoint。

## 可比性说明

可以比较的部分：

- 数据集、模型容量、seed、formal no-mosaic 协议整体一致，n 模型曲线可以放在同一张图中观察。
- 当前 N2 A2-best/A2-last 和之前 N2 A2-best 的 B 起点完全一致或非常接近，说明入口语义是可对齐的。
- 90 服务器 no-mosaic B800 是重要参照：它说明同协议下 LADD 曾经能在 700+ epoch 继续涨。

需要谨慎的部分：

- B100/B120/B200 与 B800 的 cosine LR schedule 不同。它们能比较早期趋势，但不能把 B100 的 epoch 100 直接等价为 B800 的 epoch 100 或最终结论。
- 当前 N3/N4 已改成 YOLO-init detector + A2 decomposition；之前 N3/N4 是 SAR baseline detector + A2 decomposition。因此 N3/N4 新旧不是同一个入口，只能作为“入口改变”的对照。

## 汇总表

{markdown_table(summary)}

## 图 1：当前 B800 + 之前 current-baseline LADD + 90 健康主线

![n_current_baseline_overlay_full](figures/ladd_current_baseline_overlay_20260614/n_current_baseline_overlay_full.png)

这张图说明：当前 N1 det-only 已经非常接近/超过历史健康主线的中期区间；当前 N2 能到 baseline best 附近但会 NaN；之前 B100/B120 的 current-baseline LADD 在短程内看起来不差，但因为 schedule 短，不能代替 B800 长程判断。

## 图 2：前 160 epoch 放大

![n_current_baseline_overlay_early160](figures/ladd_current_baseline_overlay_20260614/n_current_baseline_overlay_early160.png)

早期曲线可以看出，之前 current-baseline LADD 与当前 B800 前缀在 50-120 epoch 区间确实具有趋势可比性；但当前 B800 的 LR 下降更慢，所以后续仍有空间。

## 图 3：归一化进度轴

![n_current_baseline_overlay_normalized_progress](figures/ladd_current_baseline_overlay_20260614/n_current_baseline_overlay_normalized_progress.png)

归一化后，B100/B120/B200 的“末尾”其实对应完整 schedule 的末段，而 B800 当前只跑到约 40%-65%。这解释了为什么短程实验看起来更快平台：它们在 schedule 语义上已经走到后段。

## 图 4：只看 current-baseline / A2 入口，不混入 YOLO-init N3/N4

![n_current_baseline_overlay_no_yoloinit](figures/ladd_current_baseline_overlay_20260614/n_current_baseline_overlay_no_yoloinit.png)

不看 YOLO-init 后，核心矛盾更清楚：det-only baseline continuation 很强；full LADD A2 入口有收益苗头但数值不稳定；历史健康主线说明长程 full LADD 本来可以冲到更高，因此下一步更该定位当前 full B 的稳定性和入口差异，而不是简单否定 LADD。

## 图 5：detector loss 对比

![n_current_baseline_overlay_detector_losses](figures/ladd_current_baseline_overlay_20260614/n_current_baseline_overlay_detector_losses.png)

N2 A2-last 的 loss 在 NaN 前明显出现 cls loss 抬升；det-only 和历史健康主线更平稳。这个图支持“异常不是普通平台，而是 full LADD B 数值/优化稳定性问题”。

## 结论

1. 你的理解基本对：formal no-mosaic 协议没有变，所以 n 模型主线曲线有可比性。
2. 但 B100/B120/B200 和 B800 的 schedule 不同，所以更适合比较“入口趋势”，不适合比较最终能力。
3. 当前最强的安全结果仍是 N1 SAR-last det-only B800 前缀，best `0.57687@337`；它已经接近/超过历史 no-mosaic cap2 健康主线的最终量级。
4. full LADD 的 N2 不是完全没信号：A2-last B800 到过 `0.56073@271`，但随后 NaN；如果能解决稳定性，它仍可能有空间。
5. 新 N3/N4 从 YOLO-init 出发，不应和旧 N3/N4 直接当成同一实验；目前它们主要说明“detector 从 YOLO init 开始太慢/偏弱”。
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_all()
    summary = summarize(runs)
    plot_epoch_overlay(runs, RUNS, "n_current_baseline_overlay_full", xlim=None, include_yoloinit=True)
    plot_epoch_overlay(runs, RUNS, "n_current_baseline_overlay_early160", xlim=(1, 160), include_yoloinit=True)
    plot_epoch_overlay(runs, RUNS, "n_current_baseline_overlay_no_yoloinit", xlim=None, include_yoloinit=False)
    plot_normalized_overlay(runs)
    plot_losses(runs)
    write_report(summary)
    print(f"Wrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"Wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
