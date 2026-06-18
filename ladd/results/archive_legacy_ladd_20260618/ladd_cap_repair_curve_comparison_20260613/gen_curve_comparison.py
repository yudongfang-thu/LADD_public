from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "ladd/results/ladd_cap_repair_curve_comparison_20260613"
FIG_DIR = REPO / "docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613"
DOC_PATH = REPO / "docs/experiments/LADD_CAP_REPAIR_CURVE_COMPARISON_20260613_CN.md"
SUMMARY_PATH = OUT_ROOT / "ladd_cap_repair_curve_summary_20260613.csv"

BASELINES = {
    "n": {"best": 0.55654, "final": 0.55127, "safe": None},
    "s": {"best": 0.62897, "final": 0.62233, "safe": 0.62697},
    "m": {"best": 0.65580, "final": 0.64903, "safe": 0.65380},
}


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    model: str
    phase: str
    family: str
    path: str
    include_in_main_table: bool = True


def first_existing(pattern: str) -> str:
    matches = sorted(REPO.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return str(matches[0].relative_to(REPO))


CURVES: list[CurveSpec] = [
    CurveSpec(
        "n_mainline_b800",
        "n cap mainline BN-freeze B800",
        "n",
        "B",
        "n_b",
        "ladd/results/ladd_curve_analysis_20260612/source/runs_public_ladd/yolo11n/cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s42_gpu0/results.csv",
    ),
    CurveSpec(
        "n_repair_weakkd025_b200",
        "n repair weakKD0.25 B200",
        "n",
        "B",
        "n_b",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/n_weakkd0p25_b200/unknown/*_b_e200_*/results.csv"
        ),
    ),
    CurveSpec(
        "s_mainline_a2",
        "s cap mainline A2 full50",
        "s",
        "A2",
        "s_a2",
        "ladd/results/ladd_curve_analysis_20260612/source/runs_public_ladd/yolo11s/cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2_a2_e50_b64_s0_gpu1/results.csv",
    ),
    CurveSpec(
        "s_a2_detonly",
        "s A2 det-only full50",
        "s",
        "A2",
        "s_a2",
        "ladd/results/a2_damage_20260612/final_evidence/runs/s_A2_detonly_A2/results.csv",
    ),
    CurveSpec(
        "s_a2_lr3e4_full50",
        "s A2 lr3e-4 full50",
        "s",
        "A2",
        "s_a2",
        "ladd/results/a2_damage_20260612/final_evidence/runs/s_A2_lr3e4_A2/results.csv",
    ),
    CurveSpec(
        "s_a2_lr3e4_short13",
        "s A2 lr3e-4 short13",
        "s",
        "A2",
        "s_a2",
        "ladd/results/a2_selection_20260612/current_evidence/runs/s_A2_lr3e4_short13_B1_A2/results.csv",
    ),
    CurveSpec(
        "s_a2_lr1e4_short15",
        "s A2 lr1e-4 short15",
        "s",
        "A2",
        "s_a2",
        "ladd/results/a2_selection_20260612/current_evidence/runs/s_A2_lr1e4_short15_B1_A2/results.csv",
    ),
    CurveSpec(
        "s_repair_weakkd01_a2",
        "s repair weakKD0.1 A2 short13",
        "s",
        "A2",
        "s_a2",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_weakkd0p1_b120_b32retry2/unknown/*_a2_e13_*/results.csv"
        ),
        include_in_main_table=False,
    ),
    CurveSpec(
        "s_repair_detonlylr1e4_a2",
        "s repair det-only lr1e-4 A2 short13",
        "s",
        "A2",
        "s_a2",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_detonly_lr1e4_b120/unknown/*_a2_e13_*/results.csv"
        ),
        include_in_main_table=False,
    ),
    CurveSpec(
        "s_mainline_b800",
        "s cap mainline BN-freeze B800",
        "s",
        "B",
        "s_b",
        "ladd/results/ladd_curve_analysis_20260612/source/runs_public_ladd/yolo11s/cap2/"
        "ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s0_gpu1/results.csv",
    ),
    CurveSpec(
        "s_alpha05_b400",
        "s alphaKD0.5 B400",
        "s",
        "B",
        "s_b",
        "ladd/results/capacity_kd_20260611/alpha0p5_b400/b_results.csv",
    ),
    CurveSpec(
        "s_alpha025_b400",
        "s alphaKD0.25 B400",
        "s",
        "B",
        "s_b",
        "ladd/results/capacity_kd_20260611/alpha0p25_b400/b_results.csv",
    ),
    CurveSpec(
        "s_bdetonly_r2_b400",
        "s B det-only r2 B400",
        "s",
        "B",
        "s_b",
        "ladd/results/a2_damage_20260612/final_evidence/runs/s_B_detonly_r2_B400/results.csv",
    ),
    CurveSpec(
        "s_select_bdetonly200",
        "s short13 + B det-only200",
        "s",
        "B",
        "s_b",
        "ladd/results/ladd_curve_analysis_20260612/source/runs_public_ladd/yolo11s/cap2/"
        "ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_a2select_s_s0_a2lr3e4_short13_bdetonly200_b_e200_b64_s0_gpu0/results.csv",
    ),
    CurveSpec(
        "s_repair_weakkd01_b120",
        "s repair weakKD0.1 B120",
        "s",
        "B",
        "s_b",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_weakkd0p1_b120_b32retry2/unknown/*_b_e120_*/results.csv"
        ),
    ),
    CurveSpec(
        "s_repair_detonlylr1e4_b120",
        "s repair det-only lr1e-4 B120",
        "s",
        "B",
        "s_b",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_detonly_lr1e4_b120/unknown/*_b_e120_*/results.csv"
        ),
    ),
    CurveSpec(
        "m_a2_probe",
        "m A2 probe full50",
        "m",
        "A2",
        "m_a2",
        "ladd/results/capacity_kd_20260611/m_a2_probe/a2_results.csv",
    ),
    CurveSpec(
        "m_a2_detonly",
        "m A2 det-only full50",
        "m",
        "A2",
        "m_a2",
        "ladd/results/a2_damage_20260612/final_evidence/runs/m_A2_detonly_A2/results.csv",
    ),
    CurveSpec(
        "m_a2_lr3e4_retry2",
        "m A2 lr3e-4 retry2 interrupted",
        "m",
        "A2",
        "m_a2",
        "ladd/results/a2_damage_20260612/final_evidence/runs/m_A2_lr3e4_retry2_A2_incomplete/results.csv",
    ),
    CurveSpec(
        "m_a2_lr3e4_full50_retry3",
        "m A2 lr3e-4 full50 retry3",
        "m",
        "A2",
        "m_a2",
        "ladd/results/a2_selection_20260612/current_evidence/runs/m_A2_lr3e4_full50_retry3_B1_A2/results.csv",
    ),
    CurveSpec(
        "m_a2_short10",
        "m A2 short10",
        "m",
        "A2",
        "m_a2",
        "ladd/results/a2_selection_20260612/current_evidence/runs/m_A2_short10_B1_A2/results.csv",
    ),
    CurveSpec(
        "m_a2_lr3e4_short10",
        "m A2 lr3e-4 short10",
        "m",
        "A2",
        "m_a2",
        "ladd/results/a2_selection_20260612/current_evidence/runs/m_A2_lr3e4_short10_B1_A2/results.csv",
    ),
    CurveSpec(
        "m_repair_lr3e4_short4",
        "m repair lr3e-4 short4",
        "m",
        "A2",
        "m_a2",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr3e4_short4_b1/unknown/*_a2_e4_*/results.csv"
        ),
    ),
    CurveSpec(
        "m_repair_lr1e4_short5",
        "m repair lr1e-4 short5",
        "m",
        "A2",
        "m_a2",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr1e4_short5_b1/unknown/*_a2_e5_*/results.csv"
        ),
    ),
    CurveSpec(
        "m_repair_freezebn_short4_autodl",
        "m repair freezeBN short4 AutoDL",
        "m",
        "A2",
        "m_a2",
        first_existing(
            "ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_m_short4_freezebn/unknown/*_a2_e4_*/results.csv"
        ),
    ),
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_curve(spec: CurveSpec) -> pd.DataFrame:
    path = REPO / spec.path
    if not path.exists():
        raise FileNotFoundError(spec.path)
    df = clean_columns(pd.read_csv(path))
    if "epoch" not in df.columns:
        raise ValueError(f"missing epoch column: {spec.path}")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"])
    df["epoch"] = df["epoch"].astype(int)
    df = df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    for col in df.columns:
        if col != "epoch":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


def metric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def det_total(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"]
    vals = [metric(df, c) for c in cols]
    return vals[0] + vals[1] + vals[2]


def summarize(spec: CurveSpec, df: pd.DataFrame) -> dict:
    y = metric(df, "metrics/mAP50-95(B)")
    best_idx = y.idxmax()
    last_idx = df.index[-1]
    base = BASELINES[spec.model]
    return {
        "key": spec.key,
        "label": spec.label,
        "model": spec.model,
        "phase": spec.phase,
        "family": spec.family,
        "epochs": int(df["epoch"].max()),
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "best_map": float(y.loc[best_idx]),
        "last_epoch": int(df.loc[last_idx, "epoch"]),
        "last_map": float(y.loc[last_idx]),
        "best_final_drop": float(y.loc[best_idx] - y.loc[last_idx]),
        "baseline_best": base["best"],
        "baseline_final": base["final"],
        "safe_threshold": base["safe"],
        "best_minus_baseline_best": float(y.loc[best_idx] - base["best"]),
        "last_minus_baseline_final": float(y.loc[last_idx] - base["final"]),
        "source_path": spec.path,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def add_baselines(ax: plt.Axes, model: str) -> None:
    base = BASELINES[model]
    ax.axhline(base["best"], color="black", linestyle="--", linewidth=0.9, alpha=0.7, label="SAR best")
    ax.axhline(base["final"], color="0.35", linestyle=":", linewidth=0.9, alpha=0.8, label="SAR final")
    if base["safe"] is not None:
        ax.axhline(base["safe"], color="firebrick", linestyle="-.", linewidth=0.9, alpha=0.7, label="safe threshold")


def plot_ap(ax: plt.Axes, curves: Iterable[CurveSpec], data: dict[str, pd.DataFrame], max_epoch: int | None = None) -> None:
    for spec in curves:
        df = data[spec.key]
        if max_epoch is not None:
            df = df[df["epoch"] <= max_epoch]
        ax.plot(df["epoch"], metric(df, "metrics/mAP50-95(B)"), linewidth=1.7, label=spec.label)
    ax.set_xlabel("Epoch in phase")
    ax.set_ylabel("AP50-95")


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def make_figures(data: dict[str, pd.DataFrame]) -> None:
    configure_style()

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    s_b = [spec for spec in CURVES if spec.family == "s_b"]
    plot_ap(ax, s_b, data)
    add_baselines(ax, "s")
    ax.set_xlim(0, 820)
    ax.legend(ncol=2, frameon=False, loc="lower left")
    save(fig, "fig1_s_b_stage_ap_curves")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharex=False)
    loss_specs = [spec for spec in CURVES if spec.key in {
        "s_mainline_b800",
        "s_alpha05_b400",
        "s_bdetonly_r2_b400",
        "s_select_bdetonly200",
        "s_repair_detonlylr1e4_b120",
    }]
    for spec in loss_specs:
        df = data[spec.key]
        axes[0].plot(df["epoch"], det_total(df, "train"), linewidth=1.5, label=spec.label)
        axes[1].plot(df["epoch"], det_total(df, "val"), linewidth=1.5, label=spec.label)
    axes[0].set_xlabel("Epoch in B")
    axes[0].set_ylabel("train det loss total")
    axes[1].set_xlabel("Epoch in B")
    axes[1].set_ylabel("val det loss total")
    axes[0].legend(frameon=False, fontsize=6)
    save(fig, "fig2_s_b_stage_detector_losses")

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2))
    s_a2 = [spec for spec in CURVES if spec.family == "s_a2" and spec.key not in {
        "s_repair_weakkd01_a2",
        "s_repair_detonlylr1e4_a2",
    }]
    plot_ap(axes[0, 0], s_a2, data, max_epoch=55)
    add_baselines(axes[0, 0], "s")
    axes[0, 0].legend(frameon=False, fontsize=6)
    aux_specs = [spec for spec in s_a2 if spec.key in {"s_mainline_a2", "s_a2_lr3e4_full50", "s_a2_lr3e4_short13", "s_a2_lr1e4_short15"}]
    for spec in aux_specs:
        df = data[spec.key]
        axes[0, 1].plot(df["epoch"], metric(df, "train/reach_match_loss"), linewidth=1.5, label=spec.label)
        axes[1, 0].plot(df["epoch"], metric(df, "train/reach_rank_loss"), linewidth=1.5, label=spec.label)
        rec = metric(df, "train/t_rec_loss") + metric(df, "train/s_rec_loss")
        axes[1, 1].plot(df["epoch"], rec, linewidth=1.5, label=spec.label)
    axes[0, 1].set_xlabel("Epoch in A2")
    axes[0, 1].set_ylabel("reach match")
    axes[1, 0].set_xlabel("Epoch in A2")
    axes[1, 0].set_ylabel("reach rank")
    axes[1, 1].set_xlabel("Epoch in A2")
    axes[1, 1].set_ylabel("t_rec + s_rec")
    axes[0, 1].legend(frameon=False, fontsize=6)
    save(fig, "fig3_s_a2_ap_and_aux_losses")

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    n_b = [spec for spec in CURVES if spec.family == "n_b"]
    plot_ap(ax, n_b, data)
    add_baselines(ax, "n")
    ax.set_xlim(0, 820)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig4_n_b_stage_ap_curves")

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    m_a2 = [spec for spec in CURVES if spec.family == "m_a2"]
    plot_ap(ax, m_a2, data, max_epoch=55)
    add_baselines(ax, "m")
    ax.set_xlim(0, 55)
    ax.legend(ncol=2, frameon=False, fontsize=6, loc="lower left")
    save(fig, "fig5_m_a2_ap_curves")

    rows = pd.read_csv(SUMMARY_PATH)
    bar_rows = rows[
        rows["key"].isin(
            [
                "s_mainline_b800",
                "s_alpha05_b400",
                "s_alpha025_b400",
                "s_bdetonly_r2_b400",
                "s_select_bdetonly200",
                "s_repair_weakkd01_b120",
                "s_repair_detonlylr1e4_b120",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = ["#4C78A8" if v >= 0 else "#E45756" for v in bar_rows["last_minus_baseline_final"]]
    labels = [x.replace("s ", "").replace(" B", "\nB") for x in bar_rows["label"]]
    ax.bar(range(len(bar_rows)), bar_rows["last_minus_baseline_final"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(bar_rows)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("final - SAR final")
    save(fig, "fig6_s_b_final_delta_bar")


def markdown_table(rows: pd.DataFrame) -> str:
    cols = ["label", "phase", "epochs", "best_epoch", "best_map", "last_map", "best_final_drop", "best_minus_baseline_best", "last_minus_baseline_final"]
    out = rows[cols].copy()
    for col in ["best_map", "last_map", "best_final_drop", "best_minus_baseline_best", "last_minus_baseline_final"]:
        out[col] = out[col].map(lambda x: f"{x:.5f}")
    return out.to_markdown(index=False)


def write_report(summary: pd.DataFrame) -> None:
    rel_fig = "https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613"
    local_fig = "./figures/ladd_cap_repair_curve_comparison_20260613"
    s_b = summary[summary["family"] == "s_b"]
    s_a2 = summary[summary["family"] == "s_a2"]
    n_b = summary[summary["family"] == "n_b"]
    m_a2 = summary[summary["family"] == "m_a2"]
    text = f"""# LADD cap 主线与后续修改曲线对比

日期：2026-06-13

这份文档把主线 cap 版本 LADD 与后续几轮修改实验放在同一组曲线里比较。重点不是只看最高点，而是同时看 peak、final、best-final gap，以及 detector loss / reach / rec 这些辅助信号是否解释性能漂移。

## 1. 数据与图件

数据汇总：

```text
ladd/results/ladd_cap_repair_curve_comparison_20260613/ladd_cap_repair_curve_summary_20260613.csv
```

绘图脚本：

```text
ladd/results/ladd_cap_repair_curve_comparison_20260613/gen_curve_comparison.py
```

图件：

![s B-stage AP curves]({rel_fig}/fig1_s_b_stage_ap_curves.png)

![s B-stage detector losses]({rel_fig}/fig2_s_b_stage_detector_losses.png)

![s A2 AP and auxiliary losses]({rel_fig}/fig3_s_a2_ap_and_aux_losses.png)

![n B-stage AP curves]({rel_fig}/fig4_n_b_stage_ap_curves.png)

![m A2 AP curves]({rel_fig}/fig5_m_a2_ap_curves.png)

![s B final delta]({rel_fig}/fig6_s_b_final_delta_bar.png)

如果当前网络环境阻止 CDN 图片加载，可直接打开仓库内本地图片：

- [fig1_s_b_stage_ap_curves.png]({local_fig}/fig1_s_b_stage_ap_curves.png)
- [fig2_s_b_stage_detector_losses.png]({local_fig}/fig2_s_b_stage_detector_losses.png)
- [fig3_s_a2_ap_and_aux_losses.png]({local_fig}/fig3_s_a2_ap_and_aux_losses.png)
- [fig4_n_b_stage_ap_curves.png]({local_fig}/fig4_n_b_stage_ap_curves.png)
- [fig5_m_a2_ap_curves.png]({local_fig}/fig5_m_a2_ap_curves.png)
- [fig6_s_b_final_delta_bar.png]({local_fig}/fig6_s_b_final_delta_bar.png)

## 2. YOLO11s：B 阶段修改没有解决 late regression

主线 cap BN-freeze B800 的 best 为 `0.63388@263`，final 为 `0.61759`，best-final gap 达到 `0.01629`。这说明 s 的问题不是“学不到”，而是中后期把已经学到的性能退掉了。

后续几次修改给出了更清楚的拆分：

- `alphaKD0.5/0.25 B400` 的 best 仍在 `0.630` 左右，但 final 分别落到 `0.61802/0.61719`，降低 KD 强度没有解决 final 退化。
- `B det-only r2 B400` best 为 `0.63025@226`，final 为 `0.61923`，说明即使 B 阶段关掉 LADD/KD 辅助项，也仍存在 late regression。
- `repair det-only lr1e-4 B120` best 为 `0.63125@13`，final 为 `0.62556`，短期比 SAR final 高，但仍低于 safe threshold `0.62697`。
- `repair weakKD0.1 B120` best 为 `0.62827@10`，final 为 `0.62267`，没有保住 A2 的高点。

Detector loss 曲线进一步说明：不少 run 的 train detector loss 持续下降，但 AP 在后期下降。这更像 generalization / validation drift，而不是训练 loss 爆炸。

## 3. YOLO11s：A2 的问题可以被低 LR/短 A2 缓解，但 B 仍会再损伤

A2 full50 主线 best/final 为 `0.62664/0.62349`；`s A2 lr3e-4 short13` 能把 A2 final 锁到 `0.63057`。这说明 A2 阶段本身可以通过“低 LR + 短训练”获得更干净的起点。

但后续 B 阶段没有稳定继承这个起点：`short13 + B det-only200` 从 A2 `0.63057` 进入 B 后只到 `0.62436` best、`0.61880` final；repair 的 `det-only lr1e-4 B120` 有短期恢复，但 final 仍回落。结论是：A2 selection 是必要修复，但不是完整修复。

reach/rec 曲线没有出现明显爆炸。`reach_match` 快速下降，`reach_rank` 和 `t_rec+s_rec` 更像平稳收敛或缓慢变化；它们不能单独解释 s 的 B-stage AP 退化。

## 4. YOLO11n：主线 cap 是正向且稳定，repair 不是更强替代

n cap mainline BN-freeze B800 best/final 为 `0.57615/0.57295`，都高于 n SAR baseline `0.55654/0.55127`。`n repair weakKD0.25 B200` 为 `0.56476/0.56419`，方向是正的，但弱于已知 mainline。

这支持一个清晰口径：n 上主线 cap 版本有稳定正证据；repair 变体只是辅助诊断，不应替代 n 主线。

## 5. YOLO11m：短 A2、低 LR、A2 freeze BN 都没有救回来

m 的 A2 曲线没有任何一条达到 safe threshold `0.65380`。少数 run 的 early best 略高于 m SAR final `0.64903`，例如 `m A2 probe` 的 `0.65026` 和 `m A2 lr3e-4 short10` 的 `0.64929`，但 final 都回落到 SAR final 以下。目前最高的 repair 类结果是 `m repair lr1e-4 short5`，best/final 为 `0.64735/0.64416`，仍没过 m SAR final。

这说明 m 的问题不是简单的“训练太长”或“A2 LR 太大”。短 A2、低 LR、det-only、A2 freeze BN 都不能把 m 拉回安全区间，因此 m 不适合直接进入 full B。

## 6. 数值摘要

### s B-stage

{markdown_table(s_b)}

### s A2

{markdown_table(s_a2[s_a2["include_in_main_table"] == True])}

### n B-stage

{markdown_table(n_b)}

### m A2

{markdown_table(m_a2)}

## 7. 汇报建议

1. 先讲现象：s 主线 B800 在中期达到高点，但 final 明显退化；这张图比单表格更能说明 late regression。
2. 再讲第一轮修复：降低 alphaKD、B det-only、B 低 LR 都能改变局部形态，但没有根治 final drift。
3. 然后讲 A2 selection：低 LR + 短 A2 可以得到更好的 A2 起点，说明 A2 不是完全坏掉，但 B 的继承机制仍不稳定。
4. 最后讲容量差异：n 主线稳定，s 有后期退化，m 在 A2 阶段就已经低于安全线。这个容量差异是下一轮方法设计最关键的约束。
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {spec.key: load_curve(spec) for spec in CURVES}
    summary = pd.DataFrame([summarize(spec, data[spec.key]) for spec in CURVES])
    summary["include_in_main_table"] = [spec.include_in_main_table for spec in CURVES]
    summary.to_csv(SUMMARY_PATH, index=False)
    make_figures(data)
    write_report(summary)
    print(f"wrote {SUMMARY_PATH.relative_to(REPO)}")
    print(f"wrote {DOC_PATH.relative_to(REPO)}")
    print(f"wrote {FIG_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
