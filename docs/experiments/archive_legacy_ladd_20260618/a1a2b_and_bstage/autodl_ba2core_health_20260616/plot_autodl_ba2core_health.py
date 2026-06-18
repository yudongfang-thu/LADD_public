#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/experiments/autodl_ba2core_health_20260616"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def add_curve(ax, path: str | Path, label: str, *, color=None, lw=1.8, ls="-", max_epoch=360):
    path = Path(path)
    if not path.exists():
        return None
    df = read_results(path)
    if "epoch" not in df or "metrics/mAP50-95(B)" not in df:
        return None
    x = pd.to_numeric(df["epoch"], errors="coerce")
    y = pd.to_numeric(df["metrics/mAP50-95(B)"], errors="coerce")
    mask = x.notna() & y.notna() & (x <= max_epoch)
    if not mask.any():
        return None
    line = ax.plot(x[mask], y[mask], label=label, color=color, lw=lw, ls=ls, marker="o" if mask.sum() <= 3 else None, ms=4)
    return {
        "label": label,
        "epochs": int(mask.sum()),
        "last_epoch": int(x[mask].iloc[-1]),
        "first_ap": float(y[mask].iloc[0]),
        "last_ap": float(y[mask].iloc[-1]),
        "best_ap": float(y[mask].max()),
        "best_epoch": int(x[mask].iloc[y[mask].argmax()]),
    }


def main() -> None:
    b800 = pd.read_csv(ROOT / "docs/experiments/ladd_b800_restart_curve_summary_20260614.csv")
    b800 = {row["key"]: row["source"] for _, row in b800.iterrows()}
    current = OUT / "raw/current/b/results.csv"

    mosaic_root = ROOT / "docs/experiments/ladd_mosaic_protocol_compare_20260615/raw/mnt/dataY/ydf/projects"
    old_mosaic_s0 = (
        mosaic_root
        / "LADD_og/legacy_results_archive/pre_formal_nomosaic_20260528/runs_public/ogsod/hbb/ladd_converged_20260524/"
        / "ladd_hbb_ogsod11n_ladd800r2_cap2_s0_b_e800_b64_s0_gpu4/results.csv"
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    summaries: list[dict] = []
    specs = [
        (current, "new: YOLOinit A1->B(A2-core), AutoDL", "#d62728", 2.8, "-"),
        (b800["N0_yoloinit_detonly_B800sched"], "N0 YOLO-init det-only B800sched", "#7f7f7f", 1.6, "--"),
        (b800["N1_basebest_continue_B800sched"], "N1 SAR baseline-best continue", "#1f77b4", 1.8, "-"),
        (b800["N2_a2best_continue_B800sched"], "N2 A2-best continue", "#2ca02c", 1.8, "-"),
        (b800["N2_a2last_continue_B800sched"], "N2 A2-last continue", "#17becf", 1.8, "-"),
        (b800["N3_yoloinit_a2last_decomp_B800sched"], "N3 YOLO-init + A2 decomp", "#9467bd", 1.5, ":"),
        (old_mosaic_s0, "old mosaic100 LADD s0", "#ff7f0e", 1.8, "-."),
    ]
    for path, label, color, lw, ls in specs:
        row = add_curve(ax, ROOT / path if isinstance(path, str) else path, label, color=color, lw=lw, ls=ls)
        if row:
            summaries.append(row)

    ax.axhline(0.54091, color="black", lw=1.0, ls=":", alpha=0.8, label="n SAR baseline best 0.54091")
    ax.set_xlabel("B epoch")
    ax.set_ylabel("AP50-95")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 0.6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_autodl_ba2core_ap_early_compare.png", dpi=220)
    fig.savefig(FIG / "fig1_autodl_ba2core_ap_early_compare.pdf")

    cur = read_results(current)
    x = pd.to_numeric(cur["epoch"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for col, label in [
        ("train/box_loss", "box"),
        ("train/cls_loss", "cls"),
        ("train/dfl_loss", "dfl"),
    ]:
        axes[0].plot(x, pd.to_numeric(cur[col], errors="coerce"), marker="o", label=label)
    axes[0].set_xlabel("B epoch")
    axes[0].set_ylabel("detector train loss")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    for col, label in [
        ("train/t_rec_loss", "t_rec"),
        ("train/reach_match_loss", "reach_match"),
        ("train/reach_rank_loss", "reach_rank"),
        ("train/task_loss", "task"),
        ("train/kd_loss", "kd"),
        ("train/s_rec_loss", "s_rec"),
        ("train/r_aux_loss", "r_aux"),
    ]:
        if col in cur.columns:
            axes[1].plot(x, pd.to_numeric(cur[col], errors="coerce"), marker="o", label=label)
    axes[1].set_xlabel("B epoch")
    axes[1].set_ylabel("LADD core / KD losses")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_autodl_ba2core_loss_health.png", dpi=220)
    fig.savefig(FIG / "fig2_autodl_ba2core_loss_health.pdf")

    pd.DataFrame(summaries).to_csv(OUT / "autodl_ba2core_health_curve_summary_20260616.csv", index=False)
    print(FIG / "fig1_autodl_ba2core_ap_early_compare.png")
    print(FIG / "fig2_autodl_ba2core_loss_health.png")
    print(OUT / "autodl_ba2core_health_curve_summary_20260616.csv")


if __name__ == "__main__":
    main()
