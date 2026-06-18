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
        "legend.fontsize": 7.5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    init_source: str
    schedule: str
    path: Path
    color: str
    linestyle: object


RUNS = [
    Run(
        "sarbase_decomp_b100",
        "SAR-base + A2 decomp B100",
        "SAR baseline detector",
        "B100 compressed",
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N3_base_a2last_decomp_b100/run_files/results.csv",
        "#1f77b4",
        "-",
    ),
    Run(
        "sarbase_decomp_kdramp_b120",
        "SAR-base + A2 decomp KD-ramp B120",
        "SAR baseline detector",
        "B120 compressed",
        ROOT / "ladd/results/b_entrance_20260613/evidence/ladd4090/N4_base_a2last_kdramp_b120/run_files/results.csv",
        "#1f77b4",
        (0, (4, 2)),
    ),
    Run(
        "yoloinit_decomp_b800",
        "YOLO-init + A2 decomp B800",
        "YOLO init detector",
        "B800 snapshot",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N3_yoloinit_a2last_decomp_20260614_cfgfix_retry2_b_e800_b64_s0_gpu1/results.csv",
        "#d62728",
        "-",
    ),
    Run(
        "yoloinit_decomp_kdwarm_b800",
        "YOLO-init + A2 decomp KD-warm B800",
        "YOLO init detector",
        "B800 snapshot",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/ladd4090/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N4_yoloinit_a2last_decomp_kdwarmup_20260614_cfgfix_b_e800_b64_s0_gpu1/results.csv",
        "#d62728",
        (0, (4, 2)),
    ),
]

BASELINES = {
    "SAR": ROOT / "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv",
    "RGB": ROOT / "ladd/results/ladd90_formal_baselines_20260612/results/rgb_yolo11n_s0_b64.csv",
}
BASELINE_STYLE = {
    "SAR": {"color": "#6e6e6e", "linestyle": (0, (5, 2))},
    "RGB": {"color": "#8c564b", "linestyle": (0, (2, 2))},
}


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


def best(df: pd.DataFrame, col: str) -> tuple[int, float]:
    idx = df[col].idxmax()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, col])


def baseline_bests() -> dict[str, dict[str, float | int]]:
    out = {}
    for name, path in BASELINES.items():
        df = read_results(path)
        ap_ep, ap = best(df, "metrics/mAP50-95(B)")
        ap50_ep, ap50 = best(df, "metrics/mAP50(B)")
        out[name] = {"ap": ap, "ap_epoch": ap_ep, "ap50": ap50, "ap50_epoch": ap50_ep}
    return out


def add_baseline_lines(
    ax,
    bests: dict[str, dict[str, float | int]],
    metric: str,
    *,
    label_text: bool = True,
    text_x: float = 555,
) -> None:
    metric_key = "ap" if metric == "AP" else "ap50"
    epoch_key = "ap_epoch" if metric == "AP" else "ap50_epoch"
    for name, values in bests.items():
        style = BASELINE_STYLE[name]
        value = float(values[metric_key])
        epoch = int(values[epoch_key])
        ax.axhline(value, color=style["color"], linestyle=style["linestyle"], linewidth=1.0, alpha=0.85)
        if label_text:
            ax.text(
                text_x,
                value,
                f"{name} best {value:.3f}@{epoch}",
                color=style["color"],
                fontsize=7,
                va="center",
                ha="left",
            )


def plot_curves(data: dict[str, pd.DataFrame]) -> None:
    bests = baseline_bests()
    fig, axes = plt.subplots(3, 2, figsize=(11.3, 8.0), sharex=False)
    ax_ap, ax_ap_zoom, ax_kd, ax_ratio, ax_det, ax_srec = axes.flatten()

    for run in RUNS:
        df = data[run.key]
        x = df["epoch"]
        ap = df["metrics/mAP50-95(B)"]
        kd = df["train/kd_loss"]
        detector = det_loss(df)
        ratio = kd / detector.replace(0, np.nan)
        s_rec = df["train/s_rec_loss"] if "train/s_rec_loss" in df.columns else pd.Series(np.nan, index=df.index)

        for ax in [ax_ap, ax_ap_zoom]:
            ax.plot(x, ap, label=run.label, color=run.color, linestyle=run.linestyle, linewidth=1.9)
            ep, val = best(df, "metrics/mAP50-95(B)")
            ax.scatter([ep], [val], marker="*", s=52, color=run.color, edgecolor="black", linewidth=0.35, zorder=4)
        ax_kd.plot(x, smooth(kd), label=run.label, color=run.color, linestyle=run.linestyle, linewidth=1.8)
        ax_ratio.plot(x, smooth(ratio), label=run.label, color=run.color, linestyle=run.linestyle, linewidth=1.8)
        ax_det.plot(x, smooth(detector), label=run.label, color=run.color, linestyle=run.linestyle, linewidth=1.8)
        ax_srec.plot(x, smooth(s_rec), label=run.label, color=run.color, linestyle=run.linestyle, linewidth=1.8)

    add_baseline_lines(ax_ap, bests, "AP", label_text=True, text_x=555)
    add_baseline_lines(ax_ap_zoom, bests, "AP", label_text=False)
    ax_ap.set_xlim(0, 560)
    ax_ap_zoom.set_xlim(0, 140)
    ax_kd.set_xlim(0, 560)
    ax_ratio.set_xlim(0, 560)
    ax_det.set_xlim(0, 560)
    ax_srec.set_xlim(0, 560)

    panels = [
        (ax_ap, "AP (mAP50-95)", "AP raw epoch"),
        (ax_ap_zoom, "AP (mAP50-95)", "AP first 140 epochs"),
        (ax_kd, "train KD loss", "KD loss"),
        (ax_ratio, "train KD / detector loss", "KD/det"),
        (ax_det, "train box+cls+dfl", "det loss"),
        (ax_srec, "train s_rec loss", "student rec"),
    ]
    for ax, ylabel, tag in panels:
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.text(0.01, 0.96, tag, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("B-stage epoch")
    ax_ap.legend(frameon=False, loc="lower right")
    ax_kd.legend(frameon=False, loc="upper right")
    fig.text(
        0.01,
        0.01,
        "YOLO11n no-mosaic LADD-like split-load comparison. Blue = SAR-baseline detector init; red = YOLO-init detector. Solid = direct KD, dashed = KD warm/ramp. Note B100/B120 compressed schedules are not final B800-equivalent.",
        fontsize=8,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"n_nomosaic_laddlike_init_source_curves.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(out)
    plt.close(fig)


def write_table(data: dict[str, pd.DataFrame]) -> None:
    bests = baseline_bests()
    sar_ap = float(bests["SAR"]["ap"])
    rows = []
    for run in RUNS:
        df = data[run.key]
        ap_ep, ap = best(df, "metrics/mAP50-95(B)")
        kd = df["train/kd_loss"]
        detector = det_loss(df)
        ratio = kd / detector.replace(0, np.nan)
        rows.append(
            {
                "run": run.label,
                "init_source": run.init_source,
                "schedule": run.schedule,
                "epochs_recorded": int(df["epoch"].iloc[-1]),
                "first_ap": float(df["metrics/mAP50-95(B)"].iloc[0]),
                "best_ap": ap,
                "best_ap_epoch": ap_ep,
                "last_ap": float(df["metrics/mAP50-95(B)"].iloc[-1]),
                "gap_best_ap_vs_sar_baseline": ap - sar_ap,
                "kd_first": float(kd.iloc[0]),
                "kd_last": float(kd.iloc[-1]),
                "kd_max": float(kd.max()),
                "kd_det_ratio_last": float(ratio.iloc[-1]) if pd.notna(ratio.iloc[-1]) else "",
                "s_rec_first": float(df["train/s_rec_loss"].iloc[0]),
                "s_rec_last": float(df["train/s_rec_loss"].iloc[-1]),
                "source": str(run.path.relative_to(ROOT)),
            }
        )

    csv_path = OUT_DIR / "n_nomosaic_laddlike_init_source_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)

    md_path = OUT_DIR / "n_nomosaic_laddlike_init_source_table.md"
    lines = [
        "# YOLO11n no-mosaic LADD-like init-source comparison",
        "",
        f"SAR baseline best AP reference: `{sar_ap:.5f}`.",
        "",
        "| Run | Init source | Schedule | Epochs | First AP | Best AP @ epoch | Gap vs SAR best | Last AP | KD first | KD last | KD max | Last KD/det | s_rec first -> last |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio = r["kd_det_ratio_last"]
        ratio_text = f"{ratio:.5f}" if isinstance(ratio, float) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    r["run"],
                    r["init_source"],
                    r["schedule"],
                    str(r["epochs_recorded"]),
                    f"{r['first_ap']:.5f}",
                    f"{r['best_ap']:.5f} @{r['best_ap_epoch']}",
                    f"{r['gap_best_ap_vs_sar_baseline']:+.5f}",
                    f"{r['last_ap']:.5f}",
                    f"{r['kd_first']:.5f}",
                    f"{r['kd_last']:.5f}",
                    f"{r['kd_max']:.5f}",
                    ratio_text,
                    f"{r['s_rec_first']:.5f} -> {r['s_rec_last']:.5f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Caveat: the SAR-baseline-init split-load runs are B100/B120 compressed entrance diagnostics, while YOLO-init split-load runs are B800 running snapshots. They compare entrance behavior and loss scale, not final converged B800 capacity.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    data = {run.key: read_results(run.path) for run in RUNS}
    plot_curves(data)
    write_table(data)


if __name__ == "__main__":
    main()
