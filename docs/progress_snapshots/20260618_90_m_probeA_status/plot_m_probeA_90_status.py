#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


RUNS = {
    "mosaic100": {
        "ladd": RAW / "m_probeA_mosaic100_b64_results.csv",
        "sar": RAW / "m_sar_mosaic100_b64_baseline.csv",
        "rgb": RAW / "m_rgb_mosaic100_b64_baseline.csv",
        "ladd_label": "Probe-A b64",
        "sar_label": "SAR baseline b64",
        "rgb_label": "RGB baseline b64",
        "note": "mosaic100/close700, b64 references",
    },
    "no-mosaic": {
        "ladd": RAW / "m_probeA_nomosaic_b64_results.csv",
        "sar": RAW / "m_sar_nomosaic_b32_baseline_ref.csv",
        "rgb": RAW / "m_rgb_nomosaic_b32_baseline_ref.csv",
        "ladd_label": "Probe-A b64",
        "sar_label": "SAR baseline b32 ref",
        "rgb_label": "RGB baseline b32 ref",
        "note": "Probe-A b64; no completed b64 SAR/RGB baseline found, so b32 formal baselines are shown as references",
    },
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def metric_col(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    candidates = [c for c in df.columns if target.replace(" ", "") in c.replace(" ", "")]
    if candidates:
        return candidates[0]
    raise KeyError(f"Missing column {target!r} in {list(df.columns)}")


def ap_col(df: pd.DataFrame) -> str:
    return metric_col(df, "metrics/mAP50-95(B)")


def ap50_col(df: pd.DataFrame) -> str:
    return metric_col(df, "metrics/mAP50(B)")


def det_loss(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ("train/box_loss", "train/cls_loss", "train/dfl_loss") if c in df.columns]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    return df[cols].sum(axis=1)


def latest_and_best(df: pd.DataFrame) -> dict[str, float]:
    ap = ap_col(df)
    ap50 = ap50_col(df)
    latest = df.iloc[-1]
    best_ap_i = df[ap].astype(float).idxmax()
    best_ap = df.loc[best_ap_i]
    best_ap50_i = df[ap50].astype(float).idxmax()
    best_ap50 = df.loc[best_ap50_i]
    return {
        "rows": int(len(df)),
        "latest_epoch": int(latest["epoch"]),
        "current_ap50": float(latest[ap50]),
        "current_ap": float(latest[ap]),
        "best_ap_epoch": int(best_ap["epoch"]),
        "best_ap": float(best_ap[ap]),
        "ap50_at_best_ap": float(best_ap[ap50]),
        "best_ap50_epoch": int(best_ap50["epoch"]),
        "best_ap50": float(best_ap50[ap50]),
    }


def row_at_or_before(df: pd.DataFrame, epoch: int) -> pd.Series:
    sub = df[df["epoch"] <= epoch]
    if sub.empty:
        return df.iloc[0]
    return sub.iloc[-1]


def summarize() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, cfg in RUNS.items():
        ladd = load_csv(cfg["ladd"])
        sar = load_csv(cfg["sar"])
        rgb = load_csv(cfg["rgb"])
        ls = latest_and_best(ladd)
        ss = latest_and_best(sar)
        rs = latest_and_best(rgb)
        sar_same = row_at_or_before(sar, ls["latest_epoch"])
        rgb_same = row_at_or_before(rgb, ls["latest_epoch"])
        ap, ap50 = ap_col(ladd), ap50_col(ladd)
        sar_ap, sar_ap50 = ap_col(sar), ap50_col(sar)
        rgb_ap, rgb_ap50 = ap_col(rgb), ap50_col(rgb)
        gap_ap = float(rs["best_ap"] - ss["best_ap"])
        gap_ap50 = float(rs["best_ap50"] - ss["best_ap50"])
        delta_ap = ls["current_ap"] - float(sar_same[sar_ap])
        delta_ap50 = ls["current_ap50"] - float(sar_same[sar_ap50])
        rows.append(
            {
                "protocol": protocol,
                "ladd_rows": ls["rows"],
                "latest_epoch": ls["latest_epoch"],
                "current_ap50": ls["current_ap50"],
                "current_ap": ls["current_ap"],
                "best_ap_epoch": ls["best_ap_epoch"],
                "best_ap": ls["best_ap"],
                "ap50_at_best_ap": ls["ap50_at_best_ap"],
                "best_ap50_epoch": ls["best_ap50_epoch"],
                "best_ap50": ls["best_ap50"],
                "sar_ref_rows": ss["rows"],
                "sar_same_epoch": int(sar_same["epoch"]),
                "sar_same_ap50": float(sar_same[sar_ap50]),
                "sar_same_ap": float(sar_same[sar_ap]),
                "delta_ap50_vs_sar_same_epoch": delta_ap50,
                "delta_ap_vs_sar_same_epoch": delta_ap,
                "sar_best_ap50": ss["best_ap50"],
                "sar_best_ap": ss["best_ap"],
                "sar_best_ap_epoch": ss["best_ap_epoch"],
                "sar_best_ap50_epoch": ss["best_ap50_epoch"],
                "rgb_ref_rows": rs["rows"],
                "rgb_same_epoch": int(rgb_same["epoch"]),
                "rgb_same_ap50": float(rgb_same[rgb_ap50]),
                "rgb_same_ap": float(rgb_same[rgb_ap]),
                "rgb_best_ap50": rs["best_ap50"],
                "rgb_best_ap": rs["best_ap"],
                "rgb_best_ap_epoch": rs["best_ap_epoch"],
                "rgb_best_ap50_epoch": rs["best_ap50_epoch"],
                "ap_gap_closure_current_pct": 100.0 * delta_ap / gap_ap if gap_ap else float("nan"),
                "ap50_gap_closure_current_pct": 100.0 * delta_ap50 / gap_ap50 if gap_ap50 else float("nan"),
                "note": cfg["note"],
            }
        )
    return pd.DataFrame(rows)


def save_summary(summary: pd.DataFrame) -> None:
    summary.to_csv(ROOT / "summary.csv", index=False)
    display = summary.copy()
    for col in display.columns:
        if display[col].dtype.kind == "f":
            display[col] = display[col].map(lambda x: f"{x:.5f}")
    (ROOT / "summary.md").write_text(display.to_markdown(index=False) + "\n")


def plot() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "savefig.dpi": 240,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.2), sharex="col")
    colors = {"ladd": "#2ca02c", "sar": "#7f7f7f", "rgb": "#8c564b"}
    styles = {"ladd": "-", "sar": "--", "rgb": ":"}

    for col, (protocol, cfg) in enumerate(RUNS.items()):
        data = {k: load_csv(cfg[k]) for k in ("ladd", "sar", "rgb")}
        labels = {"ladd": cfg["ladd_label"], "sar": cfg["sar_label"], "rgb": cfg["rgb_label"]}
        for key, df in data.items():
            epoch = df["epoch"]
            axes[0, col].plot(epoch, df[ap_col(df)], styles[key], color=colors[key], lw=2.0, label=labels[key])
            axes[1, col].plot(epoch, df[ap50_col(df)], styles[key], color=colors[key], lw=2.0, label=labels[key])
            loss = det_loss(df)
            if not loss.empty:
                axes[2, col].plot(epoch, loss, styles[key], color=colors[key], lw=1.8, label=labels[key])

        ladd = data["ladd"]
        last = ladd.iloc[-1]
        best_ap = ladd.loc[ladd[ap_col(ladd)].idxmax()]
        best_ap50 = ladd.loc[ladd[ap50_col(ladd)].idxmax()]
        axes[0, col].scatter([last["epoch"]], [last[ap_col(ladd)]], color=colors["ladd"], s=48, zorder=4)
        axes[0, col].scatter([best_ap["epoch"]], [best_ap[ap_col(ladd)]], marker="*", color="#ff7f0e", s=120, zorder=5)
        axes[1, col].scatter([last["epoch"]], [last[ap50_col(ladd)]], color=colors["ladd"], s=48, zorder=4)
        axes[1, col].scatter([best_ap50["epoch"]], [best_ap50[ap50_col(ladd)]], marker="*", color="#ff7f0e", s=120, zorder=5)
        axes[0, col].text(
            0.02,
            0.04,
            f"{protocol}\nlatest e{int(last['epoch'])}: AP={last[ap_col(ladd)]:.3f}\nbest e{int(best_ap['epoch'])}: AP={best_ap[ap_col(ladd)]:.3f}",
            transform=axes[0, col].transAxes,
            va="bottom",
            fontsize=9,
        )

    axes[0, 0].set_ylabel("AP (mAP50-95)")
    axes[1, 0].set_ylabel("AP50")
    axes[2, 0].set_ylabel("train box+cls+dfl")
    axes[2, 0].set_xlabel("epoch")
    axes[2, 1].set_xlabel("epoch")
    for ax in axes.flat:
        ax.set_xlim(0, 820)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "m_probeA_90_status.png")
    fig.savefig(ROOT / "m_probeA_90_status.pdf")


def main() -> None:
    summary = summarize()
    save_summary(summary)
    plot()
    print(summary.to_string(index=False))
    print(ROOT / "m_probeA_90_status.png")


if __name__ == "__main__":
    main()
