#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class Run:
    key: str
    label: str
    path: Path
    color: str
    linestyle: object = "-"
    linewidth: float = 1.8
    alpha: float = 1.0
    kind: str = "control"


RUNS = [
    Run(
        "yolo_init_detonly",
        "YOLO-init det-only",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N0_yoloinit_detonly_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "#d62728",
    ),
    Run(
        "sar_last_detonly",
        "SAR-last det-only",
        ROOT
        / "ladd/results/b800_restart_20260614/evidence_raw/autodl/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_b800sched_N1_baselast_continue_20260614_autodl_b_e800_b64_s0_gpu0/results.csv",
        "#1f77b4",
    ),
    Run(
        "sar_baseline",
        "SAR baseline",
        ROOT / "ladd/results/ladd90_formal_baselines_20260612/results/sar_yolo11n_s0_b64.csv",
        "#6e6e6e",
        linestyle=(0, (5, 2)),
        linewidth=1.3,
        alpha=0.78,
        kind="baseline",
    ),
    Run(
        "rgb_baseline",
        "RGB baseline",
        ROOT / "ladd/results/ladd90_formal_baselines_20260612/results/rgb_yolo11n_s0_b64.csv",
        "#8c564b",
        linestyle=(0, (2, 2)),
        linewidth=1.3,
        alpha=0.78,
        kind="baseline",
    ),
]


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def smooth(s: pd.Series, window: int = 15) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).mean()


def det_loss(df: pd.DataFrame, prefix: str = "train") -> pd.Series:
    cols = [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(index=df.index, dtype=float)
    return df[existing].sum(axis=1)


def best(df: pd.DataFrame, metric: str) -> tuple[int, float]:
    idx = df[metric].idxmax()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, metric])


def value_at_epoch(df: pd.DataFrame, metric: str, epoch: int) -> float | None:
    match = df.loc[df["epoch"].astype(int) == int(epoch), metric]
    if match.empty:
        return None
    return float(match.iloc[-1])


def plot_performance(data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharex=True)
    panels = [
        (axes[0], "metrics/mAP50-95(B)", "AP (mAP50-95)"),
        (axes[1], "metrics/mAP50(B)", "AP50"),
    ]
    for ax, metric, ylabel in panels:
        for run in RUNS:
            df = data[run.key]
            ax.plot(
                df["epoch"],
                df[metric],
                label=run.label,
                color=run.color,
                linestyle=run.linestyle,
                linewidth=run.linewidth,
                alpha=run.alpha,
            )
            best_epoch, best_value = best(df, metric)
            ax.scatter(
                [best_epoch],
                [best_value],
                s=28 if run.kind == "baseline" else 42,
                marker="o" if run.kind == "baseline" else "*",
                color=run.color,
                edgecolor="black",
                linewidth=0.35,
                zorder=4,
            )
            if run.kind == "baseline":
                ax.axhline(best_value, color=run.color, linestyle=run.linestyle, linewidth=0.9, alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("epoch")
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.set_xlim(0, 820)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    fig.text(
        0.01,
        0.01,
        "YOLO11n formal no-mosaic. Solid red/blue are the paired B800 init-source controls; gray/brown dashed curves are same-protocol SAR/RGB baselines.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"n_nomosaic_init_controls_performance.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(out)
    plt.close(fig)


def plot_losses(data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), sharex=True)
    panels = [
        (axes[0, 0], lambda df: det_loss(df, "train"), "train box+cls+dfl"),
        (axes[0, 1], lambda df: det_loss(df, "val"), "val box+cls+dfl"),
        (axes[1, 0], lambda df: df["train/box_loss"], "train box loss"),
        (axes[1, 1], lambda df: df["train/cls_loss"], "train cls loss"),
    ]
    for ax, series_fn, ylabel in panels:
        for run in RUNS:
            df = data[run.key]
            y = series_fn(df)
            if y.isna().all():
                continue
            ax.plot(
                df["epoch"],
                smooth(y),
                label=run.label,
                color=run.color,
                linestyle=run.linestyle,
                linewidth=run.linewidth,
                alpha=run.alpha,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.set_xlim(0, 820)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].legend(frameon=False, loc="best")
    fig.text(
        0.01,
        0.01,
        "Loss curves use centered rolling mean (window=15). The two B800 control runs share the same B-stage schedule but differ in detector initialization.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ["png", "pdf"]:
        out = FIG_DIR / f"n_nomosaic_init_controls_losses.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(out)
    plt.close(fig)


def write_tables(data: dict[str, pd.DataFrame]) -> None:
    sar_df = data["sar_baseline"]
    sar_ap_epoch, sar_ap_best = best(sar_df, "metrics/mAP50-95(B)")
    sar_ap50_epoch, sar_ap50_best = best(sar_df, "metrics/mAP50(B)")

    rows = []
    for run in RUNS:
        df = data[run.key]
        ap_epoch, ap_best = best(df, "metrics/mAP50-95(B)")
        ap50_epoch, ap50_best = best(df, "metrics/mAP50(B)")
        last_epoch = int(df["epoch"].iloc[-1])
        last_ap = float(df["metrics/mAP50-95(B)"].iloc[-1])
        last_ap50 = float(df["metrics/mAP50(B)"].iloc[-1])
        sar_same_epoch_ap = value_at_epoch(sar_df, "metrics/mAP50-95(B)", last_epoch)
        rows.append(
            {
                "key": run.key,
                "run": run.label,
                "role": "baseline reference" if run.kind == "baseline" else "init-source control",
                "epochs_recorded": last_epoch,
                "first_ap": float(df["metrics/mAP50-95(B)"].iloc[0]),
                "best_ap": ap_best,
                "best_ap_epoch": ap_epoch,
                "last_ap": last_ap,
                "best_ap_gap_vs_sar_best": ap_best - sar_ap_best,
                "last_ap_gap_vs_sar_best": last_ap - sar_ap_best,
                "last_ap_gap_vs_sar_same_epoch": (last_ap - sar_same_epoch_ap) if sar_same_epoch_ap is not None else "",
                "best_ap50": ap50_best,
                "best_ap50_epoch": ap50_epoch,
                "last_ap50": last_ap50,
                "best_ap50_gap_vs_sar_best": ap50_best - sar_ap50_best,
                "source": str(run.path.relative_to(ROOT)),
            }
        )

    csv_path = OUT_DIR / "n_nomosaic_init_control_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)

    md_path = OUT_DIR / "n_nomosaic_init_control_table.md"
    lines = [
        "# YOLO11n no-mosaic init-source control table",
        "",
        f"SAR baseline best AP is `{sar_ap_best:.5f}@{sar_ap_epoch}`; SAR baseline best AP50 is `{sar_ap50_best:.5f}@{sar_ap50_epoch}`.",
        "",
        "| Run | Role | Epochs | First AP | Best AP @ epoch | Gap AP vs SAR best | Last AP | Last AP gap vs SAR best | Best AP50 @ epoch | Gap AP50 vs SAR best |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["run"],
                    r["role"],
                    str(r["epochs_recorded"]),
                    f"{r['first_ap']:.5f}",
                    f"{r['best_ap']:.5f} @{r['best_ap_epoch']}",
                    f"{r['best_ap_gap_vs_sar_best']:+.5f}",
                    f"{r['last_ap']:.5f}",
                    f"{r['last_ap_gap_vs_sar_best']:+.5f}",
                    f"{r['best_ap50']:.5f} @{r['best_ap50_epoch']}",
                    f"{r['best_ap50_gap_vs_sar_best']:+.5f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: `YOLO-init det-only` and `SAR-last det-only` are the cleanest paired controls here because both use the B800 schedule and turn off LADD losses; their main difference is detector initialization.",
            "The SAR/RGB baseline curves are reference curves from the same formal no-mosaic protocol, not B-stage continuation runs.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(md_path)


def main() -> None:
    data = {run.key: read_results(run.path) for run in RUNS}
    plot_performance(data)
    plot_losses(data)
    write_tables(data)


if __name__ == "__main__":
    main()
