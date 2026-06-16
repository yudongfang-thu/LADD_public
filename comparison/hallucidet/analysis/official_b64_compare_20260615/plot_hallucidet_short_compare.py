#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
RAW_DIR = OUT_DIR / "raw"


def read_csv(path: str | Path, epoch_offset: int = 0) -> pd.DataFrame:
    df = pd.read_csv(REPO / path if not Path(path).is_absolute() else path)
    if "epoch" not in df.columns:
        raise RuntimeError(f"{path} has no epoch column")
    df = df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce") + epoch_offset
    return df


def metric(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    raise RuntimeError(f"missing metric {name}")


def det_train_loss(df: pd.DataFrame) -> pd.Series:
    if "train/loss" in df.columns:
        return pd.to_numeric(df["train/loss"], errors="coerce")
    keys = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    if all(k in df.columns for k in keys):
        return sum(pd.to_numeric(df[k], errors="coerce") for k in keys)
    raise RuntimeError("cannot infer train loss")


def det_val_loss(df: pd.DataFrame) -> pd.Series:
    if "val/loss" in df.columns:
        return pd.to_numeric(df["val/loss"], errors="coerce")
    keys = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
    if all(k in df.columns for k in keys):
        return sum(pd.to_numeric(df[k], errors="coerce") for k in keys)
    return pd.Series([float("nan")] * len(df))


def load_runs() -> dict[str, list[dict]]:
    return {
        "n": [
            {
                "label": "SAR baseline n b64 seed0",
                "kind": "baseline",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv"),
                "style": dict(color="#4C78A8", linestyle="-", linewidth=1.8),
            },
            {
                "label": "RGB teacher n b64 seed0",
                "kind": "baseline",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/rgb_yolo11n_s0_b64.csv"),
                "style": dict(color="#72B7B2", linestyle="-", linewidth=1.8),
            },
            {
                "label": "Old custom standalone n b16",
                "kind": "hallucidet",
                "df": read_csv(
                    "comparison/results_shutdown_sync_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/hallucidet_standalone/yolo11n/hallucidet_yolo11n_s0_800ep_img256_b16_autodl_20260613/results.csv"
                ),
                "style": dict(color="#F58518", linestyle="--", linewidth=2.0),
            },
            {
                "label": "Official-style n b16 stopped@25",
                "kind": "hallucidet",
                "df": read_csv(RAW_DIR / "hallucidet_official_unet_n_b16_stopped25_results.csv"),
                "style": dict(color="#E45756", linestyle="-.", linewidth=2.0),
            },
            {
                "label": "Official-style n b64 running",
                "kind": "hallucidet",
                "df": read_csv(RAW_DIR / "hallucidet_official_unet_n_b64_running_results.csv"),
                "style": dict(color="#B279A2", linestyle="-", linewidth=2.2),
            },
        ],
        "s": [
            {
                "label": "SAR baseline s b64 seed0",
                "kind": "baseline",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11s_s0_b64.csv"),
                "style": dict(color="#4C78A8", linestyle="-", linewidth=1.8),
            },
            {
                "label": "RGB teacher s b64 seed0",
                "kind": "baseline",
                "df": read_csv("ladd/results/ladd90_formal_baselines_20260612/results/rgb_yolo11s_s0_b64.csv"),
                "style": dict(color="#72B7B2", linestyle="-", linewidth=1.8),
            },
            {
                "label": "Old custom standalone s b16",
                "kind": "hallucidet",
                "df": read_csv(
                    "comparison/results_shutdown_sync_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/hallucidet_standalone/yolo11s/hallucidet_yolo11s_s0_800ep_img256_b16_dual4090_20260613/results.csv"
                ),
                "style": dict(color="#F58518", linestyle="--", linewidth=2.0),
            },
            {
                "label": "Official-style s b64 running",
                "kind": "hallucidet",
                "df": read_csv(RAW_DIR / "hallucidet_official_unet_s_b64_running_results.csv"),
                "style": dict(color="#B279A2", linestyle="-", linewidth=2.2),
            },
        ],
    }


def summarize(runs: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for model, entries in runs.items():
        for e in entries:
            df = e["df"].copy()
            ap = metric(df, "metrics/mAP50-95(B)")
            ap50 = metric(df, "metrics/mAP50(B)")
            rows.append(
                {
                    "model": model,
                    "label": e["label"],
                    "epochs_recorded": int(len(df)),
                    "last_epoch": int(df["epoch"].iloc[-1]),
                    "last_map50_95": float(ap.iloc[-1]),
                    "best_epoch": int(df["epoch"].iloc[ap.idxmax()]),
                    "best_map50_95": float(ap.max()),
                    "last_map50": float(ap50.iloc[-1]),
                    "best_map50": float(ap50.max()),
                    "last_train_loss": float(det_train_loss(df).iloc[-1]),
                    "last_val_loss": float(det_val_loss(df).iloc[-1]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "hallucidet_short_compare_summary.csv", index=False)
    return out


def style_axes(ax):
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_model(model: str, entries: list[dict], max_epoch: int = 60):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    metric_specs = [
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/mAP50(B)", "mAP50"),
        ("train_loss", "train loss"),
        ("val_loss", "val loss"),
    ]
    for ax, (m, ylabel) in zip(axes.ravel(), metric_specs):
        for e in entries:
            df = e["df"].copy()
            short = df[df["epoch"] <= max_epoch]
            if short.empty:
                continue
            y = det_train_loss(short) if m == "train_loss" else det_val_loss(short) if m == "val_loss" else metric(short, m)
            label = e["label"]
            ax.plot(short["epoch"], y, label=label, **e["style"])

            if e["kind"] == "baseline" and m.startswith("metrics/"):
                full_y = metric(df, m)
                best = full_y.max()
                ax.axhline(best, color=e["style"]["color"], linestyle=":", linewidth=1.2, alpha=0.8)

        ax.set_ylabel(ylabel)
        style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"HalluciDet Standalone Short-Curve Comparison: YOLO11{model}", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"hallucidet_yolo11{model}_short_compare.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ap_only(runs: dict[str, list[dict]], max_epoch: int = 60):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    for ax, model in zip(axes, ("n", "s")):
        for e in runs[model]:
            df = e["df"]
            short = df[df["epoch"] <= max_epoch]
            if short.empty:
                continue
            ax.plot(short["epoch"], metric(short, "metrics/mAP50-95(B)"), label=e["label"], **e["style"])
            if e["kind"] == "baseline":
                ax.axhline(metric(df, "metrics/mAP50-95(B)").max(), color=e["style"]["color"], linestyle=":", linewidth=1.2, alpha=0.8)
        ax.set_title(f"YOLO11{model}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("mAP50-95")
        ax.set_ylim(bottom=0)
        style_axes(ax)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"hallucidet_short_compare_map5095.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )
    runs = load_runs()
    summary = summarize(runs)
    for model, entries in runs.items():
        plot_model(model, entries, max_epoch=60)
    plot_ap_only(runs, max_epoch=60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
