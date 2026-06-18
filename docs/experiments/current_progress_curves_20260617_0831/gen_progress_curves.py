from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


SERIES = [
    {
        "model": "n",
        "label": "SAR baseline s0",
        "path": RAW / "autodl" / "n_sar_mosaic_baseline.csv",
        "kind": "baseline",
        "status": "done",
        "color": "#222222",
        "linestyle": "--",
    },
    {
        "model": "n",
        "label": "clean A1B static s0 (AutoDL)",
        "path": RAW / "autodl" / "n_static_clean_a1b_b.csv",
        "kind": "ladd_clean",
        "status": "running",
        "color": "#0072B2",
        "linestyle": "-",
    },
    {
        "model": "n",
        "label": "clean A1B dynamic s0 (AutoDL)",
        "path": RAW / "autodl" / "n_dynamic_clean_a1b_b.csv",
        "kind": "ladd_clean",
        "status": "running",
        "color": "#D55E00",
        "linestyle": "-",
    },
    {
        "model": "n",
        "label": "A-scheme dynprobe s0 (4090)",
        "path": RAW / "4090" / "n_dynprobe_a_scheme_b.csv",
        "kind": "ladd_probe",
        "status": "running",
        "color": "#56B4E9",
        "linestyle": "-",
    },
    {
        "model": "n",
        "label": "legacy A2last s123 (90)",
        "path": RAW / "90" / "n_ladd_a2last_s123_b.csv",
        "kind": "legacy_ladd",
        "status": "running",
        "color": "#009E73",
        "linestyle": "-.",
    },
    {
        "model": "s",
        "label": "SAR baseline s0",
        "path": RAW / "90" / "s_sar_mosaic_baseline.csv",
        "kind": "baseline",
        "status": "done",
        "color": "#222222",
        "linestyle": "--",
    },
    {
        "model": "s",
        "label": "clean A1B static s0 (4090)",
        "path": RAW / "4090" / "s_static_clean_a1b_b.csv",
        "kind": "ladd_clean",
        "status": "running",
        "color": "#0072B2",
        "linestyle": "-",
    },
    {
        "model": "s",
        "label": "clean A1B dynamic s0 (4090)",
        "path": RAW / "4090" / "s_dynamic_clean_a1b_b.csv",
        "kind": "ladd_clean",
        "status": "running",
        "color": "#D55E00",
        "linestyle": "-",
    },
    {
        "model": "s",
        "label": "A-scheme dynprobe s0 (AutoDL)",
        "path": RAW / "autodl" / "s_dynprobe_a_scheme_b.csv",
        "kind": "ladd_probe",
        "status": "running",
        "color": "#56B4E9",
        "linestyle": "-",
    },
    {
        "model": "s",
        "label": "legacy skipA2 s0 (90)",
        "path": RAW / "90" / "s_ladd_skipA2_b.csv",
        "kind": "legacy_ladd",
        "status": "running",
        "color": "#009E73",
        "linestyle": "-.",
    },
    {
        "model": "s",
        "label": "legacy A1A2B s123 (AutoDL)",
        "path": RAW / "autodl" / "s_ladd_a1a2b_s123_b.csv",
        "kind": "legacy_ladd",
        "status": "running",
        "color": "#CC79A7",
        "linestyle": ":",
    },
    {
        "model": "m",
        "label": "SAR baseline s0",
        "path": RAW / "90" / "m_sar_mosaic_baseline_running.csv",
        "kind": "baseline",
        "status": "running",
        "color": "#222222",
        "linestyle": "--",
    },
    {
        "model": "m",
        "label": "RGB teacher baseline s0 (stopped)",
        "path": RAW / "90" / "m_rgb_mosaic_baseline_stopped.csv",
        "kind": "teacher_baseline",
        "status": "early_stopped",
        "color": "#D55E00",
        "linestyle": ":",
    },
]


AP_COL = "metrics/mAP50-95(B)"
AP50_COL = "metrics/mAP50(B)"


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["epoch"] = df["epoch"].astype(float).astype(int)
    for col in [AP_COL, AP50_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def finite_float(x: float | int | None) -> float:
    if x is None:
        return math.nan
    try:
        value = float(x)
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def build_tables() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    baselines: dict[str, pd.DataFrame] = {}
    records: list[dict[str, object]] = []

    for idx, item in enumerate(SERIES):
        if not item["path"].exists():
            records.append(
                {
                    "model": item["model"],
                    "label": item["label"],
                    "status": "missing",
                    "path": str(item["path"].relative_to(ROOT)),
                }
            )
            continue
        df = read_results(item["path"])
        item["frame_key"] = f"{idx}:{item['model']}:{item['label']}"
        frames[item["frame_key"]] = df
        if item["kind"] == "baseline" and item["model"] not in baselines:
            baselines[item["model"]] = df

    for item in SERIES:
        path = item["path"]
        if not path.exists():
            continue
        df = frames[item["frame_key"]]
        if df.empty:
            continue
        last = df.iloc[-1]
        best_idx = df[AP_COL].idxmax()
        best = df.loc[best_idx]
        actual_status = item["status"]
        if actual_status == "running" and int(last["epoch"]) >= 800:
            actual_status = "done"

        baseline_same_epoch = math.nan
        delta_latest = math.nan
        baseline_best_epoch = math.nan
        delta_best = math.nan
        baseline_df = baselines.get(item["model"])
        if baseline_df is not None and item["kind"] != "baseline":
            b_latest = baseline_df[baseline_df["epoch"] == int(last["epoch"])]
            if len(b_latest):
                baseline_same_epoch = finite_float(b_latest.iloc[-1][AP_COL])
                delta_latest = finite_float(last[AP_COL]) - baseline_same_epoch
            b_best = baseline_df[baseline_df["epoch"] == int(best["epoch"])]
            if len(b_best):
                baseline_best_epoch = finite_float(b_best.iloc[-1][AP_COL])
                delta_best = finite_float(best[AP_COL]) - baseline_best_epoch

        records.append(
            {
                "model": item["model"],
                "label": item["label"],
                "kind": item["kind"],
                "status": actual_status,
                "latest_epoch": int(last["epoch"]),
                "latest_ap50": finite_float(last[AP50_COL]),
                "latest_ap": finite_float(last[AP_COL]),
                "best_epoch": int(best["epoch"]),
                "best_ap50_at_best_ap": finite_float(best[AP50_COL]),
                "best_ap": finite_float(best[AP_COL]),
                "baseline_ap_same_latest_epoch": baseline_same_epoch,
                "delta_ap_vs_baseline_same_latest_epoch": delta_latest,
                "baseline_ap_same_best_epoch": baseline_best_epoch,
                "delta_best_ap_vs_baseline_same_epoch": delta_best,
                "path": str(path.relative_to(ROOT)),
            }
        )

    summary = pd.DataFrame.from_records(records)
    return frames, summary


def plot_metric(metric: str, out_name: str, ylabel: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    models = ["n", "s", "m"]
    display = {"n": "YOLO11n", "s": "YOLO11s", "m": "YOLO11m"}

    for ax, model in zip(axes, models):
        for item in SERIES:
            if item["model"] != model or not item["path"].exists():
                continue
            df = read_results(item["path"])
            if df.empty or metric not in df:
                continue
            y = df[metric]
            ax.plot(
                df["epoch"],
                y,
                label=item["label"],
                color=item["color"],
                linestyle=item["linestyle"],
                linewidth=1.7 if item["kind"] == "baseline" else 1.45,
                alpha=0.95,
            )
            last = df.iloc[-1]
            ax.scatter(
                [last["epoch"]],
                [last[metric]],
                color=item["color"],
                s=22,
                marker="x" if item["status"] == "early_stopped" else "o",
                zorder=3,
            )
            if item["status"] == "early_stopped":
                ax.annotate(
                    f"stopped @{int(last['epoch'])}",
                    xy=(last["epoch"], last[metric]),
                    xytext=(-72, 9),
                    textcoords="offset points",
                    fontsize=8,
                    color=item["color"],
                    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": item["color"]},
                )

        ax.axvline(100, color="#999999", linestyle=":", linewidth=0.9, alpha=0.65)
        ax.text(
            0.01,
            0.93,
            display[model],
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 800)
        ax.legend(loc="lower right", fontsize=7.3, frameon=False, ncol=1)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(FIG / f"{out_name}.png", dpi=240)
    fig.savefig(FIG / f"{out_name}.pdf")
    plt.close(fig)


def write_summary(summary: pd.DataFrame) -> None:
    summary_path = ROOT / "progress_summary.csv"
    summary.to_csv(summary_path, index=False)

    md_lines = [
        "# Current Progress Curves Snapshot",
        "",
        "Source: refreshed CSV snapshots under `raw/`.",
        "",
        "Important caveat: LADD B-phase epoch is the local B-stage epoch. The same-epoch baseline columns are a reference on the plotted x-axis, not a claim that both runs have identical pretraining history.",
        "",
        "| model | run | status | latest epoch | latest AP | best epoch | best AP | baseline AP at latest epoch | delta latest |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        def fmt(value: object) -> str:
            value = finite_float(value)  # type: ignore[arg-type]
            return "" if math.isnan(value) else f"{value:.5f}"

        md_lines.append(
            "| {model} | {label} | {status} | {latest_epoch} | {latest_ap} | {best_epoch} | {best_ap} | {base} | {delta} |".format(
                model=row.get("model", ""),
                label=row.get("label", ""),
                status=row.get("status", ""),
                latest_epoch=row.get("latest_epoch", ""),
                latest_ap=fmt(row.get("latest_ap")),
                best_epoch=row.get("best_epoch", ""),
                best_ap=fmt(row.get("best_ap")),
                base=fmt(row.get("baseline_ap_same_latest_epoch")),
                delta=fmt(row.get("delta_ap_vs_baseline_same_latest_epoch")),
            )
        )

    md_lines.extend(
        [
            "",
            "Stopped baseline check:",
            "",
            "- `m / RGB teacher baseline s0` stopped at epoch 680 due Ultralytics EarlyStopping: no improvement in the last 80 epochs. Best epoch reported by the log is 600.",
            "",
            "Generated figures:",
            "",
            "- `figures/progress_ap_by_model.png` / `.pdf`",
            "- `figures/progress_ap50_by_model.png` / `.pdf`",
        ]
    )
    (ROOT / "progress_summary.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    _, summary = build_tables()
    plot_metric(AP_COL, "progress_ap_by_model", "AP")
    plot_metric(AP50_COL, "progress_ap50_by_model", "AP50")
    write_summary(summary)
    print(summary[["model", "label", "status", "latest_epoch", "latest_ap", "best_epoch", "best_ap"]].to_string(index=False))
    print(f"saved {FIG / 'progress_ap_by_model.png'}")
    print(f"saved {FIG / 'progress_ap50_by_model.png'}")


if __name__ == "__main__":
    main()
