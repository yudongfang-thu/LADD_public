#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


BASE = ROOT / "ladd/results/b800_restart_20260614/evidence_raw"


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    path: Path
    color: str
    linestyle: object = "-"
    linewidth: float = 1.8


RUNS = [
    Run(
        "n0_yoloinit_detonly",
        "N0 YOLO-init det-only",
        BASE
        / "autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N0_yoloinit_detonly_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "#d62728",
        linestyle=(0, (4, 2)),
        linewidth=1.4,
    ),
    Run(
        "n1_sarlast_detonly",
        "N1 SAR-last det-only",
        BASE
        / "autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N1_baselast_continue_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "#1f77b4",
        linestyle=(0, (4, 2)),
        linewidth=1.4,
    ),
    Run(
        "n2_a2best_full",
        "N2 A2-best full LADD",
        BASE
        / "ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2best_continue_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "#9467bd",
    ),
    Run(
        "n2_a2last_full",
        "N2 A2-last full LADD",
        BASE
        / "ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N2_a2last_continue_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "#8c564b",
    ),
    Run(
        "n3_yoloinit_decomp",
        "N3 YOLO-init + A2 decomp",
        BASE
        / "ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N3_yoloinit_a2last_decomp_20260614_cfgfix_retry2_b_e800_b64_s0_gpu1/results.csv",
        "#2ca02c",
    ),
    Run(
        "n4_yoloinit_decomp_kdwarm",
        "N4 YOLO-init + A2 decomp KD-warm",
        BASE
        / "ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N4_yoloinit_a2last_decomp_kdwarmup_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "#ff7f0e",
    ),
]


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def smooth(s: pd.Series, window: int = 9) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).mean()


def det_loss(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["train/box_loss", "train/cls_loss", "train/dfl_loss"] if c in df.columns]
    return df[cols].sum(axis=1) if cols else pd.Series(index=df.index, dtype=float)


def best_ap(df: pd.DataFrame) -> tuple[int, float]:
    col = "metrics/mAP50-95(B)"
    idx = df[col].idxmax()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, col])


def plot_kd_focus(data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6), sharex=True)
    ax_ap, ax_kd, ax_kd_zoom, ax_ratio = axes.flatten()

    for run in RUNS:
        df = data[run.key]
        x = df["epoch"]
        ap = df["metrics/mAP50-95(B)"]
        kd = df["train/kd_loss"] if "train/kd_loss" in df.columns else pd.Series(0.0, index=df.index)
        det = det_loss(df)
        ratio = kd / det.replace(0, np.nan)

        ax_ap.plot(x, ap, label=run.label, color=run.color, linestyle=run.linestyle, linewidth=run.linewidth)
        ep, val = best_ap(df)
        ax_ap.scatter([ep], [val], marker="*", color=run.color, edgecolor="black", linewidth=0.35, s=55, zorder=4)

        ax_kd.plot(x, smooth(kd), label=run.label, color=run.color, linestyle=run.linestyle, linewidth=run.linewidth)
        ax_kd_zoom.plot(x, smooth(kd), label=run.label, color=run.color, linestyle=run.linestyle, linewidth=run.linewidth)
        ax_ratio.plot(x, smooth(ratio), label=run.label, color=run.color, linestyle=run.linestyle, linewidth=run.linewidth)

    ax_kd_zoom.set_ylim(-0.02, 1.15)
    panels = [
        (ax_ap, "AP (mAP50-95)", "AP"),
        (ax_kd, "train KD loss", "KD loss"),
        (ax_kd_zoom, "train KD loss, zoomed", "KD zoom"),
        (ax_ratio, "train KD / detector loss", "KD/det"),
    ]
    for ax, ylabel, tag in panels:
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.text(0.01, 0.96, tag, transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.set_xlim(0, 560)
    for ax in axes[-1]:
        ax.set_xlabel("B-stage epoch")
    ax_ap.legend(frameon=False, loc="best")
    ax_kd.legend(frameon=False, loc="best")
    fig.text(
        0.01,
        0.01,
        "YOLO11n no-mosaic B800 restart diagnostics. N0/N1 are det-only controls, so KD loss is exactly zero; N2/N3/N4 show actual KD pressure.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"n_nomosaic_b800_kdloss_focus.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(out)
    plt.close(fig)


def write_table(data: dict[str, pd.DataFrame]) -> None:
    rows = []
    for run in RUNS:
        df = data[run.key]
        kd = df["train/kd_loss"] if "train/kd_loss" in df.columns else pd.Series(0.0, index=df.index)
        det = det_loss(df)
        ratio = kd / det.replace(0, np.nan)
        ep, ap = best_ap(df)
        rows.append(
            {
                "run": run.label,
                "epochs_recorded": int(df["epoch"].iloc[-1]),
                "best_ap": ap,
                "best_ap_epoch": ep,
                "last_ap": float(df["metrics/mAP50-95(B)"].iloc[-1]),
                "kd_first": float(kd.iloc[0]),
                "kd_last": float(kd.iloc[-1]),
                "kd_min": float(kd.min()),
                "kd_max": float(kd.max()),
                "kd_nonzero_epochs": int((kd.fillna(0).abs() > 1e-12).sum()),
                "kd_det_ratio_last": float(ratio.iloc[-1]) if pd.notna(ratio.iloc[-1]) else "",
            }
        )

    csv_path = OUT_DIR / "n_nomosaic_b800_kdloss_focus_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)

    md_path = OUT_DIR / "n_nomosaic_b800_kdloss_focus_table.md"
    lines = [
        "# YOLO11n no-mosaic B800 KD-loss focus table",
        "",
        "| Run | Epochs | Best AP @ epoch | Last AP | KD first | KD last | KD max | nonzero KD epochs | Last KD/det |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        last_ratio = r["kd_det_ratio_last"]
        ratio_text = f"{last_ratio:.5f}" if isinstance(last_ratio, float) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    r["run"],
                    str(r["epochs_recorded"]),
                    f"{r['best_ap']:.5f} @{r['best_ap_epoch']}",
                    f"{r['last_ap']:.5f}",
                    f"{r['kd_first']:.5f}",
                    f"{r['kd_last']:.5f}",
                    f"{r['kd_max']:.5f}",
                    str(r["kd_nonzero_epochs"]),
                    ratio_text,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "N0/N1 are det-only controls and therefore have zero KD loss by construction. They are useful initialization controls, not KD-behavior runs.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    data = {run.key: read_results(run.path) for run in RUNS}
    plot_kd_focus(data)
    write_table(data)


if __name__ == "__main__":
    main()
