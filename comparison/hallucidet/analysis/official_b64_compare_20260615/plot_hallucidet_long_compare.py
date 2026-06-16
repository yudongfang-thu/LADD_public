#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
RAW_DIR = OUT_DIR / "raw"


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p if p.is_absolute() else REPO / p)
    if "epoch" not in df.columns:
        raise RuntimeError(f"{p} has no epoch column")
    df = df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    return df


def ap(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["metrics/mAP50-95(B)"], errors="coerce")


def ap50(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["metrics/mAP50(B)"], errors="coerce")


def load_main_runs() -> dict[str, list[dict]]:
    return {
        "n": [
            {
                "label": "SAR baseline n b64 s0",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv"),
                "style": dict(color="#4C78A8", linewidth=2.0, linestyle="-"),
            },
            {
                "label": "RGB teacher n b64 s0",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/rgb_yolo11n_s0_b64.csv"),
                "style": dict(color="#72B7B2", linewidth=2.0, linestyle="-"),
            },
            {
                "label": "Old custom standalone n b16",
                "df": read_csv(
                    "comparison/results_shutdown_sync_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/hallucidet_standalone/yolo11n/hallucidet_yolo11n_s0_800ep_img256_b16_autodl_20260613/results.csv"
                ),
                "style": dict(color="#F58518", linewidth=2.0, linestyle="--"),
            },
            {
                "label": "Official-style n b16 stopped@25",
                "df": read_csv(RAW_DIR / "hallucidet_official_unet_n_b16_stopped25_results.csv"),
                "style": dict(color="#E45756", linewidth=2.0, linestyle="-."),
            },
            {
                "label": "Official-style n b64 running",
                "df": read_csv(RAW_DIR / "hallucidet_official_unet_n_b64_running_results.csv"),
                "style": dict(color="#B279A2", linewidth=2.2, linestyle="-"),
            },
        ],
        "s": [
            {
                "label": "SAR baseline s b64 s0",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s0_b64.csv"),
                "style": dict(color="#4C78A8", linewidth=2.0, linestyle="-"),
            },
            {
                "label": "RGB teacher s b64 s0",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/rgb_yolo11s_s0_b64.csv"),
                "style": dict(color="#72B7B2", linewidth=2.0, linestyle="-"),
            },
            {
                "label": "Old custom standalone s b16",
                "df": read_csv(
                    "comparison/results_shutdown_sync_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/hallucidet_standalone/yolo11s/hallucidet_yolo11s_s0_800ep_img256_b16_dual4090_20260613/results.csv"
                ),
                "style": dict(color="#F58518", linewidth=2.0, linestyle="--"),
            },
            {
                "label": "Official-style s b64 running",
                "df": read_csv(RAW_DIR / "hallucidet_official_unet_s_b64_running_results.csv"),
                "style": dict(color="#B279A2", linewidth=2.2, linestyle="-"),
            },
        ],
    }


def load_deprecated_style_runs() -> dict[str, list[dict]]:
    return {
        "n": [
            {
                "label": "Deprecated hallucidet_style n transfer",
                "df": read_csv(
                    "ladd/results/ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11n/hallucidet_style/transfer_hallucidet_style_hbb_ogsod11n_formal_nomosaic_yolo11n_hallucidet_style_v2_20260610_transfer_s0_b_e800_b64_s0_gpu1/results.csv"
                ),
                "style": dict(color="#54A24B", linewidth=1.9, linestyle=":"),
            },
            {
                "label": "Deprecated hallucidet_style n from-yolo",
                "df": read_csv(
                    "ladd/results/ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/yolo11n/hallucidet_style/transfer_hallucidet_style_hbb_ogsod11n_from_yolo_formal_nomosaic_yolo11n_hallucidet_style_v2_20260612_hallucidet_from_yolo_clean_from_yolo_s0_b_e800_b64_s0_gpu1/results.csv"
                ),
                "style": dict(color="#EECA3B", linewidth=1.9, linestyle=":"),
            },
        ],
        "s": [
            {
                "label": "Deprecated hallucidet_style s transfer",
                "df": read_csv(
                    "ladd/results/ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11s/hallucidet_style/transfer_hallucidet_style_hbb_ogsod11s_formal_nomosaic_yolo11s_hallucidet_style_v2_20260610_transfer_s0_b_e800_b64_s0_gpu0/results.csv"
                ),
                "style": dict(color="#54A24B", linewidth=1.9, linestyle=":"),
            },
            {
                "label": "Deprecated hallucidet_style s from-yolo",
                "df": read_csv(
                    "ladd/results/ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/yolo11s/hallucidet_style/transfer_hallucidet_style_hbb_ogsod11s_from_yolo_formal_nomosaic_yolo11s_hallucidet_style_v2_20260612_hallucidet_from_yolo_clean_from_yolo_s0_b_e800_b64_s0_gpu0/results.csv"
                ),
                "style": dict(color="#EECA3B", linewidth=1.9, linestyle=":"),
            },
        ],
    }


def style_axes(ax):
    ax.grid(True, linewidth=0.7, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 0.72)


def plot_runs(runs: dict[str, list[dict]], filename: str, include_title: str):
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), sharey=True)
    for ax, model in zip(axes, ("n", "s")):
        for entry in runs[model]:
            df = entry["df"]
            ax.plot(df["epoch"], ap(df), label=entry["label"], **entry["style"])
            best_i = ap(df).idxmax()
            ax.scatter([df.loc[best_i, "epoch"]], [ap(df).loc[best_i]], s=18, color=entry["style"]["color"], zorder=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP50-95")
        ax.set_title(f"YOLO11{model}")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        style_axes(ax)
    fig.suptitle(include_title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{filename}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def summarize(runs: dict[str, list[dict]], tag: str) -> pd.DataFrame:
    rows = []
    for model, entries in runs.items():
        baseline = next((e for e in entries if e["label"].startswith("SAR baseline")), None)
        if baseline is None:
            baseline = {
                "n": {
                    "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv")
                },
                "s": {
                    "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s0_b64.csv")
                },
            }[model]
        sar_best = float(ap(baseline["df"]).max())
        sar_last = float(ap(baseline["df"]).iloc[-1])
        for e in entries:
            df = e["df"]
            y = ap(df)
            y50 = ap50(df)
            rows.append(
                {
                    "group": tag,
                    "model": model,
                    "label": e["label"],
                    "epochs_recorded": len(df),
                    "last_epoch": int(df["epoch"].iloc[-1]),
                    "best_epoch": int(df["epoch"].iloc[y.idxmax()]),
                    "best_map50_95": float(y.max()),
                    "last_map50_95": float(y.iloc[-1]),
                    "best_map50": float(y50.max()),
                    "last_map50": float(y50.iloc[-1]),
                    "best_minus_sar_best": float(y.max() - sar_best),
                    "last_minus_sar_last": float(y.iloc[-1] - sar_last),
                }
            )
    return pd.DataFrame(rows)


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )
    main_runs = load_main_runs()
    plot_runs(
        main_runs,
        "hallucidet_long_standalone_vs_baseline_map5095",
        "Standalone HalluciDet-YOLO Curves vs Formal Baselines",
    )

    with_deprecated = {
        m: main_runs[m] + load_deprecated_style_runs()[m]
        for m in ("n", "s")
    }
    plot_runs(
        with_deprecated,
        "hallucidet_long_with_deprecated_style_map5095",
        "HalluciDet-Related Historical Curves (deprecated style shown separately)",
    )
    summary = pd.concat(
        [
            summarize(main_runs, "standalone_main"),
            summarize(load_deprecated_style_runs(), "deprecated_style_only"),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT_DIR / "hallucidet_long_compare_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
