#!/usr/bin/env python3
"""Generate LADD n/s phase curves for diagnosis notes.

The script reads lightweight Ultralytics results.csv files already mirrored
under this repository. It writes publication-style PNG/PDF figures plus a
compact summary table and Markdown note.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "experiments" / "figures" / "ladd_curve_analysis_20260612"
SUMMARY_DIR = ROOT / "ladd" / "results" / "ladd_curve_analysis_20260612"
SOURCE_DIR = SUMMARY_DIR / "source" / "runs_public_ladd"

MAP = "metrics/mAP50-95(B)"
MAP50 = "metrics/mAP50(B)"
TRAIN_DET = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
VAL_DET = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
REACH = ["train/reach_match_loss", "train/reach_rank_loss"]
REC_TASK = ["train/t_rec_loss", "train/task_loss", "train/kd_loss"]

BASELINES = {
    "n": {"seed": 42, "sar_best": 0.55875, "sar_final": 0.55049},
    "s": {"seed": 0, "sar_best": 0.62897, "sar_final": 0.62233},
}


@dataclass(frozen=True)
class PhaseFile:
    key: str
    model: str
    phase: str
    path: Path
    label: str


MAINLINE_TAGS = {
    "n": "formal_nomosaic_yolo11n_cap2_s42_bnfreeze1e3_public4090dual_final_v2",
    "s": "formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2",
}

S_DIAG_TAGS = {
    "s A2 lr3e-4 short13": {
        "a2": "formal_nomosaic_yolo11s_cap2_s0_diag_a2select_s_s0_a2lr3e4_short13_b1_a2",
        "b": "formal_nomosaic_yolo11s_cap2_s0_diag_a2select_s_s0_a2lr3e4_short13_b1_b",
    },
    "s A2 lr1e-4 short15": {
        "a2": "formal_nomosaic_yolo11s_cap2_s0_diag_a2select_s_s0_a2lr1e4_short15_b1_a2",
        "b": "formal_nomosaic_yolo11s_cap2_s0_diag_a2select_s_s0_a2lr1e4_short15_b1_b",
    },
    "s short13 + B det-only200": {
        "a2": "formal_nomosaic_yolo11s_cap2_s0_diag_a2select_s_s0_a2lr3e4_short13_bdetonly200_a2",
        "b": "formal_nomosaic_yolo11s_cap2_s0_diag_a2select_s_s0_a2lr3e4_short13_bdetonly200_b",
    },
    "s B alphaKD0.5 B400": {
        "b_local": "ladd/results/capacity_kd_20260611/alpha0p5_b400/b_results.csv",
    },
    "s B alphaKD0.25 B400": {
        "b_local": "ladd/results/capacity_kd_20260611/alpha0p25_b400/b_results.csv",
    },
}


def find_result_by_substring(substring: str, *, exact_parent: str | None = None) -> Path:
    matches = sorted(SOURCE_DIR.rglob(f"*{substring}*/results.csv"))
    if exact_parent is not None:
        exact = [p for p in matches if p.parent.name == exact_parent]
        if exact:
            matches = exact
    if not matches:
        raise FileNotFoundError(f"No results.csv matching {substring}")
    return matches[-1]


def read_result(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["epoch"]).copy()


def add_loss_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if all(c in out for c in TRAIN_DET):
        out["train/det_total"] = out[TRAIN_DET].sum(axis=1)
    if all(c in out for c in VAL_DET):
        out["val/det_total"] = out[VAL_DET].sum(axis=1)
    return out


def phase_files_for_mainline(model: str) -> list[PhaseFile]:
    tag = MAINLINE_TAGS[model]
    batch = "64"
    seed = BASELINES[model]["seed"]
    gpu = "0" if model == "n" else "1"
    actual_size_in_run_name = "n" if model == "s" else model
    files = []
    for phase in ["a1", "a2", "b"]:
        epochs = {"a1": 10, "a2": 50, "b": 800}[phase]
        exact_parent = (
            f"ladd_hbb_ogsod11{actual_size_in_run_name}_{tag}_"
            f"{phase}_e{epochs}_b{batch}_s{seed}_gpu{gpu}"
        )
        path = find_result_by_substring(f"{tag}_{phase}", exact_parent=exact_parent)
        files.append(PhaseFile(tag, model, phase, path, f"{model.upper()} {phase.upper()}"))
    return files


def phase_df(files: Iterable[PhaseFile]) -> pd.DataFrame:
    chunks = []
    offset = 0
    for pf in files:
        df = add_loss_totals(read_result(pf.path))
        df["phase"] = pf.phase.upper()
        df["model"] = pf.model
        df["label"] = pf.label
        df["source_path"] = str(pf.path.relative_to(ROOT))
        df["phase_epoch"] = df["epoch"].astype(int)
        df["global_epoch"] = df["phase_epoch"] + offset
        offset += int(df["phase_epoch"].max()) if len(df) else 0
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def summarize_run(name: str, model: str, phase: str, path: Path) -> dict[str, object]:
    df = add_loss_totals(read_result(path))
    best_idx = df[MAP].idxmax()
    best = df.loc[best_idx]
    last = df.iloc[-1]
    return {
        "run": name,
        "model": model,
        "phase": phase,
        "epochs": int(len(df)),
        "best_epoch": int(best["epoch"]),
        "best_map": float(best[MAP]),
        "last_epoch": int(last["epoch"]),
        "last_map": float(last[MAP]),
        "best_final_drop": float(best[MAP] - last[MAP]),
        "last_train_det_total": float(last.get("train/det_total", np.nan)),
        "last_val_det_total": float(last.get("val/det_total", np.nan)),
        "path": str(path.relative_to(ROOT)),
    }


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "lines.linewidth": 1.7,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png")
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def mark_phases(ax: plt.Axes, df: pd.DataFrame) -> None:
    boundaries = []
    for phase in ["A1", "A2"]:
        mx = df.loc[df["phase"] == phase, "global_epoch"].max()
        if np.isfinite(mx):
            boundaries.append(mx + 0.5)
    for x in boundaries:
        ax.axvline(x, color="0.55", linestyle="--", linewidth=1)
    phase_centers = df.groupby("phase")["global_epoch"].mean().to_dict()
    for phase, cx in phase_centers.items():
        ax.text(cx, 0.96, phase, ha="center", va="top", color="0.25", fontsize=8, transform=ax.get_xaxis_transform())


def plot_mainline_map(main: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.7), sharex=False)
    for ax, model in zip(axes, ["n", "s"]):
        df = main[model]
        ax.plot(df["global_epoch"], df[MAP], color="#1f77b4", label="LADD AP50-95")
        ax.axhline(BASELINES[model]["sar_best"], color="#2ca02c", linestyle="-.", label="SAR baseline best")
        ax.axhline(BASELINES[model]["sar_final"], color="#9467bd", linestyle=":", label="SAR baseline final")
        best_idx = df[MAP].idxmax()
        best = df.loc[best_idx]
        last = df.iloc[-1]
        ax.scatter([best["global_epoch"], last["global_epoch"]], [best[MAP], last[MAP]], color=["#d62728", "#111111"], zorder=4)
        best_yoff = -18 if model == "n" else 10
        last_yoff = -20 if model == "n" else 10
        ax.annotate(f"best {best[MAP]:.4f}", (best["global_epoch"], best[MAP]), xytext=(6, best_yoff), textcoords="offset points", fontsize=9)
        ax.annotate(f"last {last[MAP]:.4f}", (last["global_epoch"], last[MAP]), xytext=(-82, last_yoff), textcoords="offset points", fontsize=9)
        ax.set_title(f"YOLO11{model} LADD phase AP curve", pad=10)
        ax.set_ylabel("AP50-95")
        ymin = min(df[MAP].min(), BASELINES[model]["sar_final"]) - 0.003
        ymax = max(df[MAP].max(), BASELINES[model]["sar_best"]) + 0.004
        ax.set_ylim(ymin, ymax)
        mark_phases(ax, df)
        ax.legend(loc="lower right", frameon=False)
    axes[-1].set_xlabel("Cumulative phase epoch (A1 + A2 + B)")
    save(fig, "fig1_ladd_ns_phase_map")


def plot_detector_losses(main: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), sharex="row")
    for row, model in enumerate(["n", "s"]):
        df = main[model]
        axes[row, 0].plot(df["global_epoch"], df["train/det_total"], color="#ff7f0e", label="train box+cls+dfl")
        axes[row, 1].plot(df["global_epoch"], df["val/det_total"], color="#1f77b4", label="val box+cls+dfl")
        for col in range(2):
            axes[row, col].set_title(f"YOLO11{model} {'train' if col == 0 else 'val'} detector loss")
            axes[row, col].set_ylabel("loss")
            mark_phases(axes[row, col], df)
            axes[row, col].legend(frameon=False)
    axes[1, 0].set_xlabel("Cumulative phase epoch")
    axes[1, 1].set_xlabel("Cumulative phase epoch")
    save(fig, "fig2_ladd_ns_detector_loss")


def plot_aux_a2(main: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8), sharex=False)
    for row, model in enumerate(["n", "s"]):
        df = main[model]
        a2 = df[df["phase"] == "A2"].copy()
        for col in REACH:
            if col in a2:
                y = a2[col].replace(0, np.nan)
                axes[row, 0].plot(a2["phase_epoch"], y, label=col.replace("train/", ""))
        for col in REC_TASK:
            if col in a2:
                y = a2[col].replace(0, np.nan)
                axes[row, 1].plot(a2["phase_epoch"], y, label=col.replace("train/", ""))
        axes[row, 0].set_title(f"YOLO11{model} A2 reach losses")
        axes[row, 1].set_title(f"YOLO11{model} A2 rec/task/KD losses")
        axes[row, 0].set_ylabel("loss")
        axes[row, 1].set_ylabel("loss")
        axes[row, 0].set_yscale("log")
        axes[row, 1].set_yscale("log")
        axes[row, 0].legend(frameon=False)
        axes[row, 1].legend(frameon=False)
    axes[1, 0].set_xlabel("A2 epoch")
    axes[1, 1].set_xlabel("A2 epoch")
    save(fig, "fig3_ladd_ns_a2_aux_losses")


def plot_s_diagnostics() -> list[dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=False)
    rows: list[dict[str, object]] = []
    colors = plt.cm.tab10.colors
    for idx, (label, spec) in enumerate(S_DIAG_TAGS.items()):
        color = colors[idx % len(colors)]
        if "a2" in spec:
            a2_path = find_result_by_substring(spec["a2"])
            a2 = read_result(a2_path)
            axes[0].plot(a2["epoch"], a2[MAP], color=color, linestyle="-", label=f"{label} A2")
            rows.append(summarize_run(label, "s", "A2", a2_path))
        if "b" in spec:
            b_path = find_result_by_substring(spec["b"])
            b = read_result(b_path)
            axes[1].plot(b["epoch"], b[MAP], color=color, linestyle="-", label=f"{label} B")
            rows.append(summarize_run(label, "s", "B", b_path))
        if "b_local" in spec:
            b_path = ROOT / spec["b_local"]
            b = read_result(b_path)
            axes[1].plot(b["epoch"], b[MAP], color=color, linestyle="--", label=label)
            rows.append(summarize_run(label, "s", "B", b_path))

    for ax in axes:
        ax.axhline(BASELINES["s"]["sar_best"], color="#2ca02c", linestyle="-.", label="s SAR baseline best")
        ax.axhline(BASELINES["s"]["sar_final"], color="#9467bd", linestyle=":", label="s SAR baseline final")
        ax.set_ylabel("AP50-95")
        ax.legend(frameon=False, ncol=2)
    axes[0].set_title("YOLO11s A2 selection curves")
    axes[1].set_title("YOLO11s B-stage degradation/repair curves")
    axes[1].set_xlabel("Phase epoch")
    axes[0].set_xlabel("A2 epoch")
    save(fig, "fig4_ladd_s_diagnostic_map")
    return rows


def write_markdown(summary: pd.DataFrame) -> None:
    md = ROOT / "docs" / "experiments" / "LADD_CURVE_ANALYSIS_20260612_CN.md"
    lines = [
        "# LADD n/s 曲线诊断图 2026-06-12",
        "",
        "本页汇总 YOLO11n / YOLO11s LADD 分阶段曲线，用于解释 A2 损伤、B 阶段 late regression，以及 reach/rec 等辅助损失是否本身收敛。",
        "",
        "## 图件",
        "",
        "![n/s phase AP](figures/ladd_curve_analysis_20260612/fig1_ladd_ns_phase_map.png)",
        "",
        "![n/s detector loss](figures/ladd_curve_analysis_20260612/fig2_ladd_ns_detector_loss.png)",
        "",
        "![n/s A2 aux losses](figures/ladd_curve_analysis_20260612/fig3_ladd_ns_a2_aux_losses.png)",
        "",
        "![s diagnostics](figures/ladd_curve_analysis_20260612/fig4_ladd_s_diagnostic_map.png)",
        "",
        "## 读图要点",
        "",
        "1. YOLO11n 的 stabilized BN-freeze 主线在 B 阶段 best 与 final 都保持在 SAR baseline 上方，说明 n 容量下方法主线有正向证据。",
        "2. YOLO11s 的 B 阶段出现明显 best-final gap：best 高于 SAR baseline，但 final 掉到 SAR baseline final 下方，属于 late regression 而不是单纯没有学到。",
        "3. YOLO11s A2 full50 在早期达到较好点后回落；short13/lr3e-4 能把 A2 final 锁在峰值附近，是当前更干净的 A2 起点。",
        "4. A2 reach_match 通常快速降到很小，reach_rank 维持在约 0.15 附近；rec/task/KD 曲线并没有爆炸，说明当前主要异常更像 detector performance drift / B-stage regression，而不是 reach/rec 不收敛。",
        "5. s 的 B det-only/alphaKD 曲线显示：进入 B 后即使 detector-only 或弱 KD，仍可能低于 A2 best；因此汇报时需要把 A2 peak 与 B best/final 分开讲。",
        "",
        "## 数值摘要",
        "",
        summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## 数据来源",
        "",
        "- `ladd/results/mainline_stability_20260609/`：4090D n/s stabilized mainline lightweight results。",
        "- `ladd/results/ladd_curve_analysis_20260612/source/runs_public_ladd/`：从服务器同步的轻量 `results.csv` / diagnostics / args / manifest。",
        "- `ladd/results/capacity_kd_20260611/`：s alphaKD B400 对比曲线。",
        "",
        "未包含 checkpoint、TensorBoard event、wandb 或完整大日志。",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    mainline = {model: phase_df(phase_files_for_mainline(model)) for model in ["n", "s"]}
    plot_mainline_map(mainline)
    plot_detector_losses(mainline)
    plot_aux_a2(mainline)

    rows: list[dict[str, object]] = []
    for model in ["n", "s"]:
        for pf in phase_files_for_mainline(model):
            rows.append(summarize_run(f"mainline YOLO11{model}", model, pf.phase.upper(), pf.path))
    rows.extend(plot_s_diagnostics())

    summary = pd.DataFrame(rows)
    summary_path = SUMMARY_DIR / "ladd_curve_analysis_summary_20260612.csv"
    summary.to_csv(summary_path, index=False)
    write_markdown(summary)

    print(f"Wrote figures to {OUT_DIR}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
