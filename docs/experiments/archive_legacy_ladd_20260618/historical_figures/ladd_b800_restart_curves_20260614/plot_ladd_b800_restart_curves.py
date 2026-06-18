#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = Path(__file__).resolve().parent
REPORT = ROOT / "docs/experiments/LADD_B800_RESTART_CURVE_ANALYSIS_20260614_CN.md"
SUMMARY_CSV = ROOT / "docs/experiments/ladd_b800_restart_curve_summary_20260614.csv"
RESULT_SUMMARY_DIR = ROOT / "ladd/results/b800_restart_20260614/summary"
RESULT_SUMMARY_CSV = RESULT_SUMMARY_DIR / "ladd_b800_restart_curve_summary_20260614.csv"
ANOMALY_DIR = RESULT_SUMMARY_DIR / "log_extracts"

MAP_COL = "metrics/mAP50-95(B)"
BASELINE_N_BEST = 0.55654
BASELINE_N_FINAL = 0.55127


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    group: str
    source: Path
    kind: str
    note: str = ""


CURRENT_SPECS = [
    RunSpec(
        "N0_yoloinit_detonly_B800sched",
        "N0 YOLO-init det-only B800sched",
        "current",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N0_yoloinit_detonly_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "current_control",
        "YOLO initial detector; no LADD loss.",
    ),
    RunSpec(
        "N1_basebest_continue_B800sched",
        "N1 SAR-best det-only B800sched",
        "current",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N1_basebest_continue_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "current_control",
        "SAR baseline best checkpoint continued with detection-only B.",
    ),
    RunSpec(
        "N1_baselast_continue_B800sched",
        "N1 SAR-last det-only B800sched",
        "current",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N1_baselast_continue_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "current_control",
        "SAR baseline last checkpoint continued with detection-only B.",
    ),
    RunSpec(
        "N2_a2best_continue_B800sched",
        "N2 A2-best full LADD B800sched",
        "current",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2best_continue_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "current_ladd",
        "Continues from A2 best checkpoint; crashed after NaN recovery failed.",
    ),
    RunSpec(
        "N2_a2last_continue_B800sched",
        "N2 A2-last full LADD B800sched",
        "current",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2last_continue_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "current_ladd",
        "Continues from A2 last checkpoint; crashed after NaN recovery failed.",
    ),
    RunSpec(
        "N3_yoloinit_a2last_decomp_B800sched",
        "N3 YOLO-init + A2 decomp B800sched",
        "current",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N3_yoloinit_a2last_decomp_20260614_cfgfix_retry2_b_e800_b64_s0_gpu1/results.csv",
        "current_ladd",
        "YOLO initial detector plus A2 decomposition split-load.",
    ),
    RunSpec(
        "N4_yoloinit_a2last_decomp_kdwarmup_B800sched",
        "N4 YOLO-init + decomp KD-warmup B800sched",
        "current",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N4_yoloinit_a2last_decomp_kdwarmup_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "current_ladd",
        "YOLO initial detector plus A2 decomposition and KD-only warmup.",
    ),
]

HISTORICAL_SPECS = [
    RunSpec(
        "old_n_nomosaic_cap2_s0_no_bnfreeze",
        "old 90 n no-mosaic cap2 s0 no-BN-freeze",
        "historical",
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu6/results.csv",
        "old_nomosaic",
        "Healthy 90 no-mosaic LADD mainline; best late.",
    ),
    RunSpec(
        "old_n_nomosaic_cap2_s0_bnfreeze",
        "old 90 n no-mosaic cap2 s0 BN-freeze",
        "historical",
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_bnfreeze1e3_90_gpu7_b_e800_b64_s0_gpu7/results.csv",
        "old_nomosaic",
        "Healthy 90 no-mosaic LADD with BN freeze.",
    ),
    RunSpec(
        "old_n_nomosaic_s123_late_regression",
        "old 90 n no-mosaic seed123 late regression",
        "historical",
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_bstable1e3_b_e800_b64_s123_gpu2/results.csv",
        "old_late_regression",
        "Historical late-regression case.",
    ),
    RunSpec(
        "old_n_nomosaic_s123_crash",
        "old 90 n no-mosaic seed123 crash",
        "historical",
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/n_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s123_a2mu1e3_b_e800_b64_s123_gpu5/results.csv",
        "old_crash",
        "Historical crash/collapse case.",
    ),
    RunSpec(
        "old_n_mosaic100_cap2_s0",
        "old 90 n mosaic100 cap2 s0",
        "historical",
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/mosaic90/ladd_b_runs/ladd_hbb_ogsod11n_ladd800r2_cap2_s0_b_e800_b64_s0_gpu4/results.csv",
        "old_mosaic",
        "Mosaic-open historical mainline; starts low then climbs strongly.",
    ),
    RunSpec(
        "old_s_nomosaic_cap2_s0_no_bnfreeze",
        "old 90 s no-mosaic cap2 s0 no-BN-freeze",
        "historical_s",
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_90/s_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_a2mu1e3_b_e800_b64_s0_gpu5/results.csv",
        "old_nomosaic_s",
        "Historical s no-mosaic LADD mainline.",
    ),
    RunSpec(
        "old_s_nomosaic_cap2_s0_bnfreeze",
        "old 4090 s no-mosaic cap2 s0 BN-freeze",
        "historical_s",
        ROOT
        / "ladd/results/converged_mainline_ladd_20260613/source/no_mosaic_4090dual/s_cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s0_gpu1/results.csv",
        "old_nomosaic_s",
        "Historical s BN-freeze LADD mainline.",
    ),
]

B100_SPECS = [
    RunSpec(
        "prev_N1_B100",
        "prev N1 SAR-base continue B100",
        "previous_b_entrance",
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N1_basecontinue_b100/run_files/results.csv",
        "previous_short",
    ),
    RunSpec(
        "prev_N2_B100",
        "prev N2 A2-best continue B100",
        "previous_b_entrance",
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N2_a2best_continue_b100/run_files/results.csv",
        "previous_short",
    ),
    RunSpec(
        "prev_N3_B100",
        "prev N3 SAR-base + decomp B100",
        "previous_b_entrance",
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N3_base_a2last_decomp_b100/run_files/results.csv",
        "previous_short",
    ),
    RunSpec(
        "prev_N4_B120",
        "prev N4 SAR-base + decomp KD-ramp B120",
        "previous_b_entrance",
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N4_base_a2last_kdramp_b120/run_files/results.csv",
        "previous_short",
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
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.5,
        }
    )


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if c != "epoch":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    return df


def value_at_last_finite(df: pd.DataFrame, col: str) -> tuple[float, int]:
    if col not in df:
        return math.nan, -1
    s = df[["epoch", col]].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s[np.isfinite(s[col])]
    if s.empty:
        return math.nan, -1
    row = s.iloc[-1]
    return float(row[col]), int(row["epoch"])


def best_value(df: pd.DataFrame, col: str) -> tuple[float, int]:
    if col not in df:
        return math.nan, -1
    s = df[["epoch", col]].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s[np.isfinite(s[col])]
    if s.empty:
        return math.nan, -1
    idx = s[col].idxmax()
    return float(s.loc[idx, col]), int(s.loc[idx, "epoch"])


def first_nonfinite_epoch(df: pd.DataFrame, cols: list[str]) -> int | None:
    check_cols = [c for c in cols if c in df.columns]
    if not check_cols:
        return None
    vals = df[check_cols].apply(pd.to_numeric, errors="coerce")
    mask = ~np.isfinite(vals.to_numpy(dtype=float))
    if not mask.any():
        return None
    first_idx = np.where(mask.any(axis=1))[0][0]
    return int(df.iloc[first_idx]["epoch"])


def load_runs(specs: list[RunSpec]) -> dict[str, pd.DataFrame]:
    runs: dict[str, pd.DataFrame] = {}
    missing = []
    for spec in specs:
        if spec.source.exists():
            runs[spec.key] = read_results(spec.source)
        else:
            missing.append(str(spec.source.relative_to(ROOT)))
    if missing:
        print("Missing result files:")
        for p in missing:
            print("  ", p)
    return runs


def summarize(specs: list[RunSpec], runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    loss_cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss", MAP_COL]
    for spec in specs:
        df = runs.get(spec.key)
        if df is None or df.empty:
            rows.append({"key": spec.key, "label": spec.label, "status": "missing"})
            continue
        first_ap = float(pd.to_numeric(df[MAP_COL], errors="coerce").dropna().iloc[0]) if MAP_COL in df and df[MAP_COL].notna().any() else math.nan
        best_ap, best_epoch = best_value(df, MAP_COL)
        last_ap, last_ap_epoch = value_at_last_finite(df, MAP_COL)
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "group": spec.group,
                "kind": spec.kind,
                "epochs_recorded": len(df),
                "last_raw_epoch": int(df["epoch"].dropna().iloc[-1]),
                "first_ap": first_ap,
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "last_finite_ap": last_ap,
                "last_finite_ap_epoch": last_ap_epoch,
                "best_final_drop": best_ap - last_ap if np.isfinite(best_ap) and np.isfinite(last_ap) else math.nan,
                "first_nonfinite_epoch": first_nonfinite_epoch(df, loss_cols),
                "last_train_box": value_at_last_finite(df, "train/box_loss")[0],
                "last_train_cls": value_at_last_finite(df, "train/cls_loss")[0],
                "last_train_dfl": value_at_last_finite(df, "train/dfl_loss")[0],
                "last_lr_pg0": value_at_last_finite(df, "lr/pg0")[0],
                "note": spec.note,
                "source": str(spec.source.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_ap_curves(
    name: str,
    specs: list[RunSpec],
    runs: dict[str, pd.DataFrame],
    *,
    xlim: tuple[int, int] | None = None,
    baseline: bool = True,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = plt.cm.tab10.colors
    for i, spec in enumerate(specs):
        df = runs.get(spec.key)
        if df is None or MAP_COL not in df:
            continue
        y = pd.to_numeric(df[MAP_COL], errors="coerce")
        ax.plot(df["epoch"], y, label=spec.label, color=colors[i % len(colors)], alpha=0.95)
        bad_ep = first_nonfinite_epoch(df, ["train/box_loss", "train/cls_loss", "train/dfl_loss", MAP_COL])
        if bad_ep is not None and (xlim is None or xlim[0] <= bad_ep <= xlim[1]):
            ax.axvline(bad_ep, color=colors[i % len(colors)], linestyle=":", alpha=0.55)
            ax.text(bad_ep, ax.get_ylim()[0], f"NaN {bad_ep}", rotation=90, va="bottom", ha="right", color=colors[i % len(colors)], fontsize=7)
    if baseline:
        ax.axhline(BASELINE_N_BEST, color="black", linestyle="--", linewidth=1.1, label="SAR n baseline best 0.55654")
        ax.axhline(BASELINE_N_FINAL, color="gray", linestyle="--", linewidth=1.0, label="SAR n baseline final 0.55127")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("B-stage epoch")
    ax.set_ylabel("AP50-95")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    save(fig, name)


def plot_detector_losses(runs: dict[str, pd.DataFrame]) -> None:
    keys = [
        "N1_basebest_continue_B800sched",
        "N2_a2best_continue_B800sched",
        "N2_a2last_continue_B800sched",
        "N3_yoloinit_a2last_decomp_B800sched",
        "N4_yoloinit_a2last_decomp_kdwarmup_B800sched",
    ]
    labels = {s.key: s.label for s in CURRENT_SPECS}
    cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True)
    colors = plt.cm.tab10.colors
    for ax, col in zip(axes, cols):
        for i, key in enumerate(keys):
            df = runs.get(key)
            if df is None or col not in df:
                continue
            ax.plot(df["epoch"], pd.to_numeric(df[col], errors="coerce"), label=labels[key], color=colors[i])
            bad_ep = first_nonfinite_epoch(df, cols)
            if bad_ep is not None:
                ax.axvline(bad_ep, color=colors[i], linestyle=":", alpha=0.5)
        ax.set_ylabel(col.replace("train/", ""))
    axes[-1].set_xlabel("B-stage epoch")
    axes[0].legend(frameon=False, ncol=2)
    save(fig, "current_n_b800_detector_losses")


def plot_n2_zoom(runs: dict[str, pd.DataFrame]) -> None:
    keys = ["N2_a2best_continue_B800sched", "N2_a2last_continue_B800sched"]
    labels = {s.key: s.label for s in CURRENT_SPECS}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=True)
    colors = ["#d62728", "#9467bd"]
    for i, key in enumerate(keys):
        df = runs.get(key)
        if df is None:
            continue
        axes[0].plot(df["epoch"], pd.to_numeric(df[MAP_COL], errors="coerce"), label=labels[key], color=colors[i])
        for col, ls in [("train/box_loss", "-"), ("train/cls_loss", "--"), ("train/dfl_loss", "-.")]:
            axes[1].plot(df["epoch"], pd.to_numeric(df[col], errors="coerce"), linestyle=ls, color=colors[i], alpha=0.85, label=f"{key.split('_B')[0]} {col.split('/')[-1]}")
        bad_ep = first_nonfinite_epoch(df, ["train/box_loss", "train/cls_loss", "train/dfl_loss", MAP_COL])
        best_ap, best_ep = best_value(df, MAP_COL)
        axes[0].axvline(best_ep, color=colors[i], linestyle="--", alpha=0.45)
        axes[0].axvline(bad_ep or -1, color=colors[i], linestyle=":", alpha=0.7)
        axes[1].axvline(bad_ep or -1, color=colors[i], linestyle=":", alpha=0.7)
        axes[0].text(best_ep, best_ap, f"best {best_ap:.5f}@{best_ep}", color=colors[i], fontsize=7)
    axes[0].axhline(BASELINE_N_BEST, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("AP50-95")
    axes[1].set_ylabel("train loss")
    axes[1].set_xlabel("B-stage epoch")
    axes[0].set_xlim(150, 330)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, ncol=2, fontsize=7)
    save(fig, "n2_abnormal_zoom")


def plot_ladd_losses(runs: dict[str, pd.DataFrame]) -> None:
    keys = [
        "N2_a2best_continue_B800sched",
        "N2_a2last_continue_B800sched",
        "N3_yoloinit_a2last_decomp_B800sched",
        "N4_yoloinit_a2last_decomp_kdwarmup_B800sched",
    ]
    labels = {s.key: s.label for s in CURRENT_SPECS}
    cols = ["train/kd_loss", "train/reach_match_loss", "train/s_rec_loss", "train/r_aux_loss"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True)
    colors = plt.cm.tab10.colors
    for ax, col in zip(axes.ravel(), cols):
        for i, key in enumerate(keys):
            df = runs.get(key)
            if df is None or col not in df:
                continue
            ax.plot(df["epoch"], pd.to_numeric(df[col], errors="coerce"), label=labels[key], color=colors[i])
        ax.set_ylabel(col.replace("train/", ""))
    for ax in axes[-1]:
        ax.set_xlabel("B-stage epoch")
    axes[0, 0].legend(frameon=False, fontsize=7)
    save(fig, "current_n_b800_ladd_losses")


def write_anomaly_extracts() -> dict[str, Path]:
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    log_roots = [
        (
            "N2_a2best_continue_B800sched",
            ROOT
            / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2best_continue_20260614_cfgfix_gpu1/b/master.log",
        ),
        (
            "N2_a2last_continue_B800sched",
            ROOT
            / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2last_continue_20260614_cfgfix_gpu1/b/master.log",
        ),
    ]
    pat = re.compile(r"(NaN|Inf|RuntimeError|Traceback|Loss NaN|deepcopy|recovering from last\\.pt)", re.I)
    for key, path in log_roots:
        blocks: list[tuple[int, list[str]]] = []
        if path.exists():
            raw = path.read_text(errors="replace").splitlines()
            for i, line in enumerate(raw):
                if pat.search(line):
                    lo = max(0, i - 3)
                    hi = min(len(raw), i + 6)
                    blocks.append((i + 1, raw[lo:hi]))
        selected = blocks[:6] + blocks[-8:] if len(blocks) > 14 else blocks
        lines = [f"# {key}", f"# matched_blocks={len(blocks)} selected_blocks={len(selected)}"]
        for lineno, block in selected:
            lines.append(f"--- {path.name}:{lineno} ---")
            lines.extend(block)
        out = ANOMALY_DIR / f"{key}_anomaly_extract.txt"
        out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        outputs[key] = out
    return outputs


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if not np.isfinite(x) else f"{x:.5f}")
    return view.to_markdown(index=False)


def write_report(summary: pd.DataFrame, anomaly_extracts: dict[str, Path]) -> None:
    current = summary[summary["group"].eq("current")].copy()
    historical = summary[summary["group"].isin(["historical", "historical_s"])].copy()
    text = f"""# LADD B800 重启批次曲线分析（2026-06-14）

本报告基于 2026-06-14 同步的轻量证据生成，只使用 `results.csv`、`ladd_diagnostics.csv`、`args.yaml`、`manifest.txt` 和 log extract；未复制或提交 checkpoint 权重、TensorBoard event、wandb 或完整 run 目录。

## 当前批次概览

{markdown_table(current, ["key", "epochs_recorded", "best_ap", "best_epoch", "last_finite_ap", "last_finite_ap_epoch", "first_nonfinite_epoch", "note"])}

## 异常定位

- `N2_a2best_continue_B800sched` 在 B epoch 229 记录到非有限训练 loss，best 为 `0.55681@214`，随后触发 NaN recovery。
- `N2_a2last_continue_B800sched` 在 B epoch 319 记录到非有限训练 loss，best 为 `0.56073@271`，随后触发 NaN recovery。
- 两条异常都不是 OOM。log 中的直接退出点是 NaN recovery 尝试从 `last.pt` 恢复时，在 Ultralytics recovery 路径里遇到 `Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol`。这说明“恢复机制”也有一个实现层面的失败点，但根因信号仍然是 B 阶段 loss 先变成 NaN/Inf。
- N1 det-only、N3/N4 在当前同步 epoch 内没有同类 NaN。这个对照使异常更像是“继承 A2 detector/full checkpoint 后进入 full LADD B 的数值稳定性问题”，而不是单纯 B800 schedule、BN freeze 或 YOLO-init split-load 必然导致。

异常 log 摘要：

- `{anomaly_extracts["N2_a2best_continue_B800sched"].relative_to(ROOT)}`
- `{anomaly_extracts["N2_a2last_continue_B800sched"].relative_to(ROOT)}`

## 图 1：当前 B800 AP 曲线

![current_n_b800_ap](figures/ladd_b800_restart_curves_20260614/current_n_b800_ap.png)

当前批次中，N1 baseline continuation 明显最强；N2 在 200-300 epoch 区间达到接近/略高于 SAR n baseline best 的点，但随后 NaN；N3/N4 从 YOLO 初始 detector 出发，曲线持续上升但截至同步点仍低于 N1/N2。

## 图 2：当前 detector loss 曲线

![current_n_b800_detector_losses](figures/ladd_b800_restart_curves_20260614/current_n_b800_detector_losses.png)

N2 的异常不是 AP 自然平台化，而是训练 loss 在中期进入非有限值。N1 det-only loss 更平稳，N3/N4 也没有在相同阶段爆掉。

## 图 3：N2 异常区间放大

![n2_abnormal_zoom](figures/ladd_b800_restart_curves_20260614/n2_abnormal_zoom.png)

放大后可以看到：N2 并不是从一开始崩，它先有一段正常上升并到达局部 best，之后才发生数值异常。这和“刚进 B 阶段被冲坏”不是同一种现象，更像中后段 full LADD objective 与继承 checkpoint 的组合出现不稳定。

## 图 4：当前 B800 与历史 n 主线 B800

![current_vs_old_n_b800_ap](figures/ladd_b800_restart_curves_20260614/current_vs_old_n_b800_ap.png)

历史 90 no-mosaic n 主线在 B800 后期仍能继续改善，best 通常出现在 700+ epoch；当前 N1 也显示 B800 schedule 不是短程 B100 能替代的。当前 N3/N4 的问题是起点和中期平台明显偏低，不能用 100 epoch 的表现直接代表 800 epoch 结论。

## 图 5：早期 B 阶段对比

![early_b_current_vs_old_n](figures/ladd_b800_restart_curves_20260614/early_b_current_vs_old_n.png)

最近实验的 B 起点较高，主要因为 N1/N2 使用的是已收敛 SAR/A2 detector checkpoint；mosaic100 历史曲线 B 起点低，是因为当时协议与进入 B 的状态不同，曲线呈现“先低后强爬升”。因此不能只凭 B 前 100 epoch 的平台程度判断 B800 的最终潜力。

## 图 6：当前 LADD loss 分量

![current_n_b800_ladd_losses](figures/ladd_b800_restart_curves_20260614/current_n_b800_ladd_losses.png)

N3/N4 的 decomposition/KD 分支没有导致明显 NaN；N2 的 NaN 更集中在继承 A2 full checkpoint 后的 full B 训练稳定性上。由于当前批次 `LADD_DIAG_LOG_GRAD=0`，还不能直接判断是否存在梯度尖峰；如果要继续定位，建议补一个短程 N2 复现实验打开 grad log。

## 图 7：当前 B800 前缀 vs 之前 B100/B120

![current_vs_previous_b_entrance_early_n](figures/ladd_b800_restart_curves_20260614/current_vs_previous_b_entrance_early_n.png)

这张图单独对齐前 140 epoch。之前 B100/B120 可以作为入口 smoke/短程趋势参考，但当前 B800 的学习率仍处于长程 schedule 的早期，不能把 B100 的末尾直接当成 B800 的稳定收敛结论。

## 图 8：历史 n/s LADD 主线

![historical_ladd_n_s_mainlines](figures/ladd_b800_restart_curves_20260614/historical_ladd_n_s_mainlines.png)

历史 no-mosaic n/s 都存在健康主线；这支持“当前异常不是 LADD 必然不收敛”，而是当前入口、checkpoint 组合、目标开启方式或数值防护需要继续定位。

## 历史参照表

{markdown_table(historical, ["key", "epochs_recorded", "best_ap", "best_epoch", "last_finite_ap", "last_finite_ap_epoch", "best_final_drop", "note"])}

## 结论草案

1. 当前异常批次最关键的问题是 N2 在 B 中期出现 NaN，恢复逻辑失败只是第二层问题。
2. N1 det-only 在 B800 schedule 下持续向上，说明 B800 的学习率调度长度本身有价值，不能用 B100 平台直接否定长程训练。
3. N3/N4 从 YOLO-init 出发还在缓慢上升，但中期远低于 N1；这说明“只加载 A2 decomposition、detector 从 YOLO 初始化”目前不是高优先级主线候选，除非后期曲线发生很强反转。
4. 历史 no-mosaic n/s 主线证明 LADD 在 formal no-mosaic 下曾经可以健康收敛；当前需要重点排查 N2 的 full B 数值稳定性和 A2 checkpoint 入口差异。
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    all_specs = CURRENT_SPECS + HISTORICAL_SPECS + B100_SPECS
    runs = load_runs(all_specs)
    summary = summarize(all_specs, runs)
    summary.to_csv(SUMMARY_CSV, index=False)
    summary.to_csv(RESULT_SUMMARY_CSV, index=False)

    plot_ap_curves("current_n_b800_ap", CURRENT_SPECS, runs, title="Current N-model B800-schedule diagnostics")
    plot_detector_losses(runs)
    plot_n2_zoom(runs)
    plot_ap_curves(
        "current_vs_old_n_b800_ap",
        CURRENT_SPECS[1:] + HISTORICAL_SPECS[:5],
        runs,
        title="Current N B800 vs historical N LADD mainlines",
    )
    plot_ap_curves(
        "early_b_current_vs_old_n",
        CURRENT_SPECS[1:] + HISTORICAL_SPECS[:5],
        runs,
        xlim=(1, 330),
        title="Early B-stage: current high-start curves vs old mainlines",
    )
    plot_ladd_losses(runs)
    plot_ap_curves(
        "current_vs_previous_b_entrance_early_n",
        CURRENT_SPECS[1:] + B100_SPECS,
        runs,
        xlim=(1, 140),
        title="Current B800 prefix vs previous B100/B120 entrance diagnostics",
    )
    plot_ap_curves(
        "historical_ladd_n_s_mainlines",
        HISTORICAL_SPECS,
        runs,
        baseline=False,
        title="Historical LADD mainlines: n/s, no-mosaic and mosaic references",
    )

    anomaly_extracts = write_anomaly_extracts()
    write_report(summary, anomaly_extracts)
    print(f"Wrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    for png in sorted(FIG_DIR.glob("*.png")):
        print(f"Wrote {png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
