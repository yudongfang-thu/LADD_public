#!/usr/bin/env python3
"""Plot observed YOLOv5x CCLKD component curves and heuristic ATKD projection.

The projection is intentionally diagnostic only. It extends unfinished runs to
YOLOv5 epoch 399 by extrapolating the same-epoch AP delta against the completed
det-only baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
ARCHIVE = (
    REPO
    / "cclkd_reproduction"
    / "yolov5_sanity"
    / "results"
    / "scalingfix_paper_components_400ep_20260613"
)
RUNS = ARCHIVE / "runs"
FIGURES = ARCHIVE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

RUN_INFO = {
    "det_only": ("Det-only", "det_only_same_trainer", "#4c4c4c"),
    "atkd": ("ATKD-only", "paper_atkd_only", "#1f77b4"),
    "ccl": ("CCL-only", "paper_ccl_only", "#ff7f0e"),
    "full": ("Full CCLKD", "paper_full", "#2ca02c"),
}


def load_results(run_dir: str) -> pd.DataFrame:
    path = RUNS / run_dir / "results.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(
        columns={
            "metrics/mAP_0.5": "ap50",
            "metrics/mAP_0.5:0.95": "ap",
            "metrics/precision": "precision",
            "metrics/recall": "recall",
        }
    )[["epoch", "ap50", "ap", "precision", "recall"]].copy()


def fit_delta_projection(
    method: pd.DataFrame,
    det: pd.DataFrame,
    metric: str,
    start_epoch: int,
    target_epoch: int = 399,
) -> pd.DataFrame:
    """Extrapolate method = det + linear(method-det) from start_epoch."""
    joined = method[["epoch", metric]].merge(
        det[["epoch", metric]].rename(columns={metric: f"det_{metric}"}),
        on="epoch",
        how="inner",
    )
    joined["delta"] = joined[metric] - joined[f"det_{metric}"]
    window = joined[joined["epoch"] >= start_epoch]
    if len(window) == 1:
        coef = np.array([0.0, float(window["delta"].iloc[0])])
    else:
        coef = np.polyfit(window["epoch"].to_numpy(), window["delta"].to_numpy(), 1)

    last_epoch = int(method["epoch"].max())
    future_epochs = np.arange(last_epoch + 1, target_epoch + 1)
    det_future = det.set_index("epoch").loc[future_epochs, metric]
    projected_delta = np.polyval(coef, future_epochs)
    out = pd.DataFrame(
        {
            "epoch": future_epochs,
            metric: det_future.to_numpy() + projected_delta,
            f"delta_{metric}_vs_det": projected_delta,
            "projection_start_epoch": start_epoch,
            "projection_slope_per_epoch": coef[0],
            "projection_intercept": coef[1],
        }
    )
    return out


def make_delta_frame(method: pd.DataFrame, det: pd.DataFrame, key: str) -> pd.DataFrame:
    out = method[["epoch", "ap50", "ap"]].merge(
        det[["epoch", "ap50", "ap"]].rename(columns={"ap50": "det_ap50", "ap": "det_ap"}),
        on="epoch",
        how="inner",
    )
    out["run_key"] = key
    out["delta_ap50_vs_det"] = out["ap50"] - out["det_ap50"]
    out["delta_ap_vs_det"] = out["ap"] - out["det_ap"]
    return out


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    data = {
        key: load_results(run_dir)
        for key, (_, run_dir, _) in RUN_INFO.items()
    }
    det = data["det_only"]
    target_epoch = int(det["epoch"].max())

    # Figure 1: strictly observed curves, no extrapolation.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for metric, ax, ylabel in [
        ("ap", axes[0], "mAP@0.5:0.95"),
        ("ap50", axes[1], "mAP@0.5"),
    ]:
        for key, (label, _, color) in RUN_INFO.items():
            df = data[key]
            ax.plot(df["epoch"], df[metric], label=f"{label} (to {int(df['epoch'].max())})", color=color, lw=1.8)
            ax.scatter(df["epoch"].iloc[-1], df[metric].iloc[-1], color=color, s=18, zorder=4)
        ax.set_xlim(0, target_epoch)
        ax.set_xlabel("YOLOv5 CSV epoch")
        ax.set_ylabel(ylabel)
        style_axes(ax)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURES / "yolov5x_cclkd_observed_actual_to400.png", bbox_inches="tight")
    fig.savefig(FIGURES / "yolov5x_cclkd_observed_actual_to400.pdf", bbox_inches="tight")
    plt.close(fig)

    # Projection policy:
    # - ATKD: primary projection uses the stable post-200 delta trend.
    # - ATKD band: flat latest delta to recent-50 delta trend.
    # - Full: short extension from epoch 392 to 399 using post-300 delta trend.
    atkd_ap_proj = fit_delta_projection(data["atkd"], det, "ap", start_epoch=200, target_epoch=target_epoch)
    atkd_ap50_proj = fit_delta_projection(data["atkd"], det, "ap50", start_epoch=200, target_epoch=target_epoch)
    atkd_flat_ap = fit_delta_projection(data["atkd"], det, "ap", start_epoch=287, target_epoch=target_epoch)
    atkd_fast_ap = fit_delta_projection(data["atkd"], det, "ap", start_epoch=237, target_epoch=target_epoch)
    full_ap_proj = fit_delta_projection(data["full"], det, "ap", start_epoch=300, target_epoch=target_epoch)
    full_ap50_proj = fit_delta_projection(data["full"], det, "ap50", start_epoch=300, target_epoch=target_epoch)

    atkd_proj = atkd_ap_proj[["epoch", "ap", "delta_ap_vs_det"]].merge(
        atkd_ap50_proj[["epoch", "ap50", "delta_ap50_vs_det"]], on="epoch"
    )
    full_proj = full_ap_proj[["epoch", "ap", "delta_ap_vs_det"]].merge(
        full_ap50_proj[["epoch", "ap50", "delta_ap50_vs_det"]], on="epoch"
    )
    atkd_proj.to_csv(ARCHIVE / "atkd_projection_to_epoch399.csv", index=False)
    full_proj.to_csv(ARCHIVE / "full_projection_to_epoch399.csv", index=False)

    all_delta = pd.concat(
        [make_delta_frame(data[k], det, k) for k in ["atkd", "ccl", "full"]],
        ignore_index=True,
    )
    all_delta.to_csv(ARCHIVE / "observed_component_delta_curves.csv", index=False)

    # Figure 2: observed curves plus heuristic dashed projection.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True, constrained_layout=True)
    ax = axes[0]
    for key, (label, _, color) in RUN_INFO.items():
        df = data[key]
        ax.plot(df["epoch"], df["ap"], label=f"{label} observed", color=color, lw=1.8)
    ax.plot(atkd_proj["epoch"], atkd_proj["ap"], color=RUN_INFO["atkd"][2], lw=2.0, ls="--", label="ATKD projected")
    ax.plot(full_proj["epoch"], full_proj["ap"], color=RUN_INFO["full"][2], lw=2.0, ls="--", label="Full projected")

    band_epochs = atkd_ap_proj["epoch"].to_numpy()
    det_future_ap = det.set_index("epoch").loc[band_epochs, "ap"].to_numpy()
    atkd_flat_y = det_future_ap + atkd_flat_ap["delta_ap_vs_det"].to_numpy()
    atkd_fast_y = det_future_ap + atkd_fast_ap["delta_ap_vs_det"].to_numpy()
    lo = np.minimum(atkd_flat_y, atkd_fast_y)
    hi = np.maximum(atkd_flat_y, atkd_fast_y)
    ax.fill_between(band_epochs, lo, hi, color=RUN_INFO["atkd"][2], alpha=0.12, label="ATKD projection band")
    ax.set_xlim(0, target_epoch)
    ax.set_ylabel("mAP@0.5:0.95")
    style_axes(ax)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    for key in ["atkd", "ccl", "full"]:
        label, _, color = RUN_INFO[key]
        df = all_delta[all_delta["run_key"] == key]
        ax.plot(df["epoch"], df["delta_ap_vs_det"], label=f"{label} observed", color=color, lw=1.8)
    ax.plot(atkd_proj["epoch"], atkd_proj["delta_ap_vs_det"], color=RUN_INFO["atkd"][2], lw=2.0, ls="--", label="ATKD projected")
    ax.plot(full_proj["epoch"], full_proj["delta_ap_vs_det"], color=RUN_INFO["full"][2], lw=2.0, ls="--", label="Full projected")
    ax.axhline(0, color="black", lw=0.8, alpha=0.7)
    ax.set_xlabel("YOLOv5 CSV epoch")
    ax.set_ylabel("Delta AP vs det-only")
    style_axes(ax)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(FIGURES / "yolov5x_cclkd_projection_to400.png", bbox_inches="tight")
    fig.savefig(FIGURES / "yolov5x_cclkd_projection_to400.pdf", bbox_inches="tight")
    plt.close(fig)

    # Milestone table with observed and projected values.
    milestone_epochs = [287, 300, 350, 399]
    det_i = det.set_index("epoch")
    ccl_i = data["ccl"].set_index("epoch")
    full_i = data["full"].set_index("epoch")
    atkd_i = data["atkd"].set_index("epoch")
    atkd_proj_i = atkd_proj.set_index("epoch")
    full_proj_i = full_proj.set_index("epoch")
    rows = []
    for ep in milestone_epochs:
        row = {
            "epoch": ep,
            "det_only_ap": float(det_i.loc[ep, "ap"]),
            "ccl_ap": float(ccl_i.loc[ep, "ap"]) if ep in ccl_i.index else np.nan,
            "ccl_delta_ap": float(ccl_i.loc[ep, "ap"] - det_i.loc[ep, "ap"]) if ep in ccl_i.index else np.nan,
        }
        if ep in atkd_i.index:
            row["atkd_ap"] = float(atkd_i.loc[ep, "ap"])
            row["atkd_delta_ap"] = float(atkd_i.loc[ep, "ap"] - det_i.loc[ep, "ap"])
            row["atkd_status"] = "observed"
        else:
            row["atkd_ap"] = float(atkd_proj_i.loc[ep, "ap"])
            row["atkd_delta_ap"] = float(atkd_proj_i.loc[ep, "delta_ap_vs_det"])
            row["atkd_status"] = "projected_delta_from_200_287"
        if ep in full_i.index:
            row["full_ap"] = float(full_i.loc[ep, "ap"])
            row["full_delta_ap"] = float(full_i.loc[ep, "ap"] - det_i.loc[ep, "ap"])
            row["full_status"] = "observed"
        else:
            row["full_ap"] = float(full_proj_i.loc[ep, "ap"])
            row["full_delta_ap"] = float(full_proj_i.loc[ep, "delta_ap_vs_det"])
            row["full_status"] = "projected_delta_from_300_392"
        row["best_ap"] = max(row["det_only_ap"], row["ccl_ap"], row["atkd_ap"], row["full_ap"])
        row["best_method"] = max(
            {
                "det_only": row["det_only_ap"],
                "ccl": row["ccl_ap"],
                "atkd": row["atkd_ap"],
                "full": row["full_ap"],
            },
            key=lambda k: {
                "det_only": row["det_only_ap"],
                "ccl": row["ccl_ap"],
                "atkd": row["atkd_ap"],
                "full": row["full_ap"],
            }[k],
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    for col in summary.select_dtypes(include=[float]).columns:
        summary[col] = summary[col].round(6)
    summary.to_csv(ARCHIVE / "projection_to400_milestone_summary.csv", index=False)

    final_rows = [
        {
            "estimate": "atkd_observed_epoch287",
            "epoch": 287,
            "ap": float(atkd_i.loc[287, "ap"]),
            "delta_ap_vs_det": float(atkd_i.loc[287, "ap"] - det_i.loc[287, "ap"]),
        },
        {
            "estimate": "atkd_epoch399_flat_latest_delta_conservative",
            "epoch": 399,
            "ap": float(det_i.loc[399, "ap"] + atkd_flat_ap.set_index("epoch").loc[399, "delta_ap_vs_det"]),
            "delta_ap_vs_det": float(atkd_flat_ap.set_index("epoch").loc[399, "delta_ap_vs_det"]),
        },
        {
            "estimate": "atkd_epoch399_linear_delta_from_200_287_main",
            "epoch": 399,
            "ap": float(atkd_proj_i.loc[399, "ap"]),
            "delta_ap_vs_det": float(atkd_proj_i.loc[399, "delta_ap_vs_det"]),
        },
        {
            "estimate": "atkd_epoch399_recent_delta_from_237_287_aggressive",
            "epoch": 399,
            "ap": float(det_i.loc[399, "ap"] + atkd_fast_ap.set_index("epoch").loc[399, "delta_ap_vs_det"]),
            "delta_ap_vs_det": float(atkd_fast_ap.set_index("epoch").loc[399, "delta_ap_vs_det"]),
        },
        {
            "estimate": "full_epoch399_projected_delta_from_300_392",
            "epoch": 399,
            "ap": float(full_proj_i.loc[399, "ap"]),
            "delta_ap_vs_det": float(full_proj_i.loc[399, "delta_ap_vs_det"]),
        },
        {
            "estimate": "ccl_observed_epoch399",
            "epoch": 399,
            "ap": float(ccl_i.loc[399, "ap"]),
            "delta_ap_vs_det": float(ccl_i.loc[399, "ap"] - det_i.loc[399, "ap"]),
        },
        {
            "estimate": "det_only_observed_epoch399",
            "epoch": 399,
            "ap": float(det_i.loc[399, "ap"]),
            "delta_ap_vs_det": 0.0,
        },
    ]
    final_summary = pd.DataFrame(final_rows)
    for col in ["ap", "delta_ap_vs_det"]:
        final_summary[col] = final_summary[col].round(6)
    final_summary.to_csv(ARCHIVE / "projection_to400_final_estimates.csv", index=False)

    md_path = ARCHIVE / "projection_to400_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# YOLOv5x CCLKD projection-to-400 diagnostic\n\n")
        f.write("YOLOv5 records 400 training epochs as CSV epochs `0..399`.\n\n")
        f.write("This is a diagnostic extrapolation, not a completed experimental result. ")
        f.write("Observed curves remain the authoritative evidence.\n\n")
        f.write("## Final estimates\n\n")
        f.write(final_summary.to_markdown(index=False))
        f.write("\n\n## Milestones\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Projection rule\n\n")
        f.write("- ATKD main projection: completed det-only curve plus linear extrapolation of ATKD same-epoch AP delta fitted on epochs 200..287.\n")
        f.write("- ATKD conservative bound: keep epoch-287 delta constant to epoch 399.\n")
        f.write("- ATKD aggressive bound: extrapolate delta trend from epochs 237..287.\n")
        f.write("- Full projection only fills the small 392..399 gap from its post-300 delta trend.\n")

    print("observed_last_epochs", {k: int(v["epoch"].max()) for k, v in data.items()})
    print("wrote", FIGURES / "yolov5x_cclkd_observed_actual_to400.png")
    print("wrote", FIGURES / "yolov5x_cclkd_projection_to400.png")
    print("wrote", ARCHIVE / "projection_to400_summary.md")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
