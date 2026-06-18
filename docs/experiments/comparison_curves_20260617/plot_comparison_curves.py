from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
CMDISTILL_PENDING_SOURCE = (
    REPO
    / "docs/experiments/archive_pending_cmdistill_20260618/cmdistill_mainline_progress_20260616/source"
)
LEGACY_LADD_RESULTS = REPO / "ladd/results/archive_legacy_ladd_20260618"


@dataclass(frozen=True)
class RunSpec:
    model: str
    name: str
    path: Path
    protocol: str
    style: str = "-"
    color: str | None = None
    linewidth: float = 1.8


RUNS = [
    RunSpec(
        "n",
        "SAR baseline",
        CMDISTILL_PENDING_SOURCE / "sar_yolo11n_baseline_results.csv",
        "SAR-only detector, formal no-mosaic, b64, seed0",
        "--",
        "#7f7f7f",
        1.5,
    ),
    RunSpec(
        "n",
        "RGB teacher",
        CMDISTILL_PENDING_SOURCE / "rgb_yolo11n_baseline_results.csv",
        "RGB-only detector reference, formal no-mosaic, b64, seed0",
        ":",
        "#8c564b",
        1.5,
    ),
    RunSpec(
        "n",
        "FGD",
        LEGACY_LADD_RESULTS / "ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/yolo11n/fgd/transfer_fgd_hbb_ogsod11n_from_yolo_formal_nomosaic_yolo11n_fgd_v2_20260612_fgd_original_low_from_yolo_s0_b_e800_b64_s0_gpu0/results.csv",
        "from YOLO pretrain, frozen RGB teacher, b64, seed0",
        "-",
        "#1f77b4",
    ),
    RunSpec(
        "n",
        "LD",
        LEGACY_LADD_RESULTS / "ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/yolo11n/ld/transfer_ld_hbb_ogsod11n_from_yolo_formal_nomosaic_yolo11n_ld_v2_20260612_ld_from_yolo_clean_from_yolo_s0_b_e800_b64_s0_gpu1/results.csv",
        "from YOLO pretrain, frozen RGB teacher, b64, seed0",
        "-",
        "#ff7f0e",
    ),
    RunSpec(
        "n",
        "CMDistill",
        REPO / "comparison/cmdistill/results_autodl_sync_20260616/from_yolo_b64_800ep/yolo11n/transfer_cmdistill_hbb_ogsod11n_from_yolo_formal_nomosaic_yolo11n_cmdistill_v3_smoke_ready_20260615_from_yolo_s0_formal800_b_e800_b64_s0_gpu0/results.csv",
        "from YOLO pretrain, frozen RGB teacher, b64, seed0",
        "-",
        "#2ca02c",
        2.2,
    ),
    RunSpec(
        "n",
        "HalluciDet",
        REPO / "comparison/hallucidet/results_autodl_sync_20260616/official_b64_800ep/yolo11n/hallucidet_yolo11n_official_unet_b64_800ep_s0_20260615/results.csv",
        "standalone hallucination protocol, frozen RGB detector, b64, seed0",
        "-",
        "#d62728",
    ),
    RunSpec(
        "n",
        "LADD ref",
        CMDISTILL_PENDING_SOURCE / "prev_ladd_cap2_yolo11n_b800_s42_results.csv",
        "LADD cap2 reference, formal no-mosaic, b64, seed42",
        "-",
        "#9467bd",
        2.2,
    ),
    RunSpec(
        "s",
        "SAR baseline",
        CMDISTILL_PENDING_SOURCE / "sar_yolo11s_baseline_results.csv",
        "SAR-only detector, formal no-mosaic, b64, seed0",
        "--",
        "#7f7f7f",
        1.5,
    ),
    RunSpec(
        "s",
        "RGB teacher",
        CMDISTILL_PENDING_SOURCE / "rgb_yolo11s_baseline_results.csv",
        "RGB-only detector reference, formal no-mosaic, b64, seed0",
        ":",
        "#8c564b",
        1.5,
    ),
    RunSpec(
        "s",
        "FGD",
        LEGACY_LADD_RESULTS / "ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/yolo11s/fgd/transfer_fgd_hbb_ogsod11s_from_yolo_formal_nomosaic_yolo11s_fgd_v2_20260612_fgd_original_low_from_yolo_from_yolo_s0_b_e800_b64_s0_gpu1/results.csv",
        "from YOLO pretrain, frozen RGB teacher, b64, seed0",
        "-",
        "#1f77b4",
    ),
    RunSpec(
        "s",
        "LD",
        LEGACY_LADD_RESULTS / "ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/yolo11s/ld/transfer_ld_hbb_ogsod11s_from_yolo_formal_nomosaic_yolo11s_ld_v2_20260612_ld_from_yolo_clean_from_yolo_s0_b_e800_b64_s0_gpu0/results.csv",
        "from YOLO pretrain, frozen RGB teacher, b64, seed0",
        "-",
        "#ff7f0e",
    ),
    RunSpec(
        "s",
        "CMDistill",
        REPO / "comparison/cmdistill/results_autodl_sync_20260616/from_yolo_b64_800ep/yolo11s/transfer_cmdistill_hbb_ogsod11s_from_yolo_formal_nomosaic_yolo11s_cmdistill_v3_smoke_ready_20260615_from_yolo_s0_formal800_b_e800_b64_s0_gpu0/results.csv",
        "from YOLO pretrain, frozen RGB teacher, b64, seed0",
        "-",
        "#2ca02c",
        2.2,
    ),
    RunSpec(
        "s",
        "HalluciDet",
        REPO / "comparison/hallucidet/results_autodl_sync_20260616/official_b64_800ep/yolo11s/hallucidet_yolo11s_official_unet_b64_800ep_s0_20260615/results.csv",
        "standalone hallucination protocol, frozen RGB detector, b64, seed0",
        "-",
        "#d62728",
    ),
    RunSpec(
        "s",
        "LADD ref",
        CMDISTILL_PENDING_SOURCE / "prev_ladd_cap2_yolo11s_b800_s0_results.csv",
        "LADD cap2 reference, formal no-mosaic, b64, seed0",
        "-",
        "#9467bd",
        2.2,
    ),
]


def read_run(spec: RunSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    df = pd.read_csv(spec.path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" not in df:
        df.insert(0, "epoch", range(1, len(df) + 1))
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    # Ultralytics/HalluciDet variants differ: some log 0..799, some 1..800.
    if df["epoch"].min(skipna=True) == 0:
        df["epoch_plot"] = df["epoch"] + 1
    else:
        df["epoch_plot"] = df["epoch"]
    for col in df.columns:
        if col != "epoch_plot":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
    for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
        if col not in df:
            df[col] = pd.NA
    df["train/det_loss"] = df[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].sum(axis=1, min_count=1)
    df["model"] = spec.model
    df["run"] = spec.name
    return df


def metric_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df:
            return col
    return None


def smooth(series: pd.Series, window: int = 11) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=1, center=True).mean()


def summarize(spec: RunSpec, df: pd.DataFrame) -> dict[str, object]:
    ap_col = metric_col(df, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95"])
    ap50_col = metric_col(df, ["metrics/mAP50(B)", "metrics/mAP50", "map50"])
    if ap_col is None:
        raise RuntimeError(f"No mAP50-95 column in {spec.path}")
    best_idx = pd.to_numeric(df[ap_col], errors="coerce").idxmax()
    last = df.iloc[-1]
    best = df.loc[best_idx]
    return {
        "model": spec.model,
        "run": spec.name,
        "rows": int(len(df)),
        "last_epoch": float(last["epoch_plot"]),
        "best_ap": float(best[ap_col]),
        "best_ap_epoch": float(best["epoch_plot"]),
        "last_ap": float(last[ap_col]),
        "best_ap50": float(best[ap50_col]) if ap50_col else "",
        "last_ap50": float(last[ap50_col]) if ap50_col else "",
        "last_det_loss": float(last["train/det_loss"]) if pd.notna(last["train/det_loss"]) else "",
        "last_kd_loss": float(last["train/kd_loss"]) if "train/kd_loss" in df and pd.notna(last["train/kd_loss"]) else "",
        "protocol": spec.protocol,
        "path": str(spec.path.relative_to(REPO)),
    }


def plot_model(model: str, specs: list[RunSpec], frames: dict[str, pd.DataFrame], suffix: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    panel_specs = [
        (axes[0, 0], ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95"], "mAP50-95(B)", None),
        (axes[0, 1], ["metrics/mAP50(B)", "metrics/mAP50", "map50"], "mAP50(B)", None),
        (axes[1, 0], ["train/det_loss"], "train box+cls+dfl loss", None),
        (axes[1, 1], ["train/kd_loss"], "train KD loss", "skip_missing"),
    ]
    for ax, candidates, ylabel, missing_policy in panel_specs:
        for spec in specs:
            df = frames[spec.name]
            col = metric_col(df, candidates)
            if col is None:
                if missing_policy == "skip_missing":
                    continue
                raise RuntimeError(f"No column from {candidates} in {spec.path}")
            y = smooth(df[col])
            if y.notna().sum() == 0:
                continue
            ax.plot(
                df["epoch_plot"],
                y,
                label=spec.name,
                color=spec.color,
                linestyle=spec.style,
                linewidth=spec.linewidth,
                alpha=0.95,
            )
            if "mAP" in ylabel:
                raw = pd.to_numeric(df[col], errors="coerce")
                if raw.notna().any():
                    idx = raw.idxmax()
                    ax.scatter(
                        [df.loc[idx, "epoch_plot"]],
                        [raw.loc[idx]],
                        color=spec.color,
                        s=18,
                        zorder=3,
                    )
        if "mAP" in ylabel:
            for ref_name, ref_color, ref_style, ref_label, y_offset in [
                ("SAR baseline", "#555555", "--", "SAR baseline best", 0.004),
                ("RGB teacher", "#8c564b", ":", "RGB teacher best", 0.004),
            ]:
                ref_df = frames.get(ref_name)
                if ref_df is None:
                    continue
                ref_col = metric_col(ref_df, candidates)
                if ref_col is None:
                    continue
                ref_values = pd.to_numeric(ref_df[ref_col], errors="coerce")
                if ref_values.notna().sum() == 0:
                    continue
                ref_best = float(ref_values.max())
                ax.axhline(
                    ref_best,
                    color=ref_color,
                    linestyle=ref_style,
                    linewidth=1.2,
                    alpha=0.7,
                    zorder=1,
                )
                ax.text(
                    796,
                    ref_best + y_offset,
                    ref_label,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=ref_color,
                    alpha=0.9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
                )
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 800)
    axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_xlabel("epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"comparison_curves_yolo11{model}_{suffix}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict[str, object]]) -> None:
    summary_path = OUT_DIR / "comparison_curve_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = OUT_DIR / "README.md"
    lines = [
        "# Comparison Curves 2026-06-17",
        "",
        "Curves compare LD, FGD, CMDistill, HalluciDet, LADD reference, SAR baseline, and RGB teacher.",
        "All plotted comparison methods use formal no-mosaic, 800 epoch, YOLO11n/s, batch 64 where applicable.",
        "",
        "No valid local `results.csv` was found for a same-protocol bimodal/fusion baseline; it is therefore not plotted.",
        "Add its `results.csv` to `RUNS` in `plot_comparison_curves.py` and rerun this script when available.",
        "",
        "## Outputs",
        "",
        "- `comparison_curves_yolo11n_main.png/pdf`",
        "- `comparison_curves_yolo11s_main.png/pdf`",
        "- `comparison_curve_summary.csv`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_rows: list[dict[str, object]] = []
    for model in ["n", "s"]:
        specs = [s for s in RUNS if s.model == model]
        frames = {}
        for spec in specs:
            df = read_run(spec)
            frames[spec.name] = df
            all_rows.append(summarize(spec, df))
        plot_model(model, specs, frames, "main")
    write_summary(all_rows)
    print(f"Wrote figures and summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
