#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


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


COLORS = {
    "Static": "#1f77b4",
    "Dynamic": "#ff7f0e",
    "LADD": "#2ca02c",
    "Static-90": "#1f77b4",
}
LINESTYLES = {
    "Static": "-",
    "Dynamic": "-",
    "LADD": "-",
    "Static-90": "--",
}
INCOMPLETE_STATUS = {
    ("n", "LADD"): "stopped",
    ("s", "Dynamic"): "running",
}


@dataclass
class RunSpec:
    model: str
    method: str
    path: Path
    label: str
    curve_note: str = ""


RUNS = [
    RunSpec("n", "Static", DATA / "n_static_results.csv", "Static"),
    RunSpec("n", "Dynamic", DATA / "n_dynamic_results.csv", "Dynamic"),
    RunSpec("n", "LADD", DATA / "n_ladd_results.csv", "LADD"),
    RunSpec("s", "Static", DATA / "s_static_4090_results.csv", "Static"),
    RunSpec("s", "Dynamic", DATA / "s_dynamic_4090_results.csv", "Dynamic"),
    RunSpec("s", "LADD", DATA / "s_ladd_results.csv", "LADD"),
]

BASELINES = {
    "n": {
        "SAR": DATA / "n_sar_baseline_results.csv",
        "RGB": DATA / "n_rgb_baseline_results.csv",
    },
    "s": {
        "SAR": DATA / "s_sar_baseline_results.csv",
        "RGB": DATA / "s_rgb_baseline_results.csv",
    },
}

BASELINE_STYLE = {
    "SAR": {"color": "#6e6e6e", "linestyle": (0, (5, 2)), "alpha": 0.72},
    "RGB": {"color": "#8c564b", "linestyle": (0, (2, 2)), "alpha": 0.72},
}


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    else:
        df["epoch"] = range(len(df))
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def metric_col(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def smooth(s: pd.Series, window: int = 15) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).mean()


def best_metric(df: pd.DataFrame, metric: str) -> tuple[int, float]:
    idx = df[metric].idxmax()
    return int(df.loc[idx, "epoch"]), float(df.loc[idx, metric])


def det_loss(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["train/box_loss", "train/cls_loss", "train/dfl_loss"] if c in df.columns]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    return df[cols].sum(axis=1)


def reach_loss(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["train/reach_match_loss", "train/reach_rank_loss"] if c in df.columns]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    return df[cols].sum(axis=1)


def task_rec_loss(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["train/task_loss", "train/s_rec_loss", "train/t_rec_loss"] if c in df.columns]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    return df[cols].sum(axis=1)


def load_markers() -> list[dict[str, str]]:
    path = DATA / "manual_snapshot_markers.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def marker_base_method(method: str) -> str:
    if method == "Static-final":
        return "Static"
    return method


def performance_extension_points(model: str, method: str) -> list[tuple[float, float, float, str]]:
    points = []
    for marker in load_markers():
        if marker["model"] != model:
            continue
        if marker_base_method(marker["method"]) != method:
            continue
        points.append(
            (
                float(marker["epoch"]),
                float(marker["ap"]),
                float(marker["ap50"]),
                marker["method"],
            )
        )
    return sorted(points, key=lambda x: x[0])


def curve_epoch_max(model: str, method: str) -> float | None:
    for spec in RUNS:
        if spec.model != model or spec.method != method or not spec.path.exists():
            continue
        df = read_csv(spec.path)
        if "epoch" not in df.columns or df.empty:
            continue
        return float(df["epoch"].max())
    return None


def marker_is_covered_by_curve(marker: dict[str, str]) -> bool:
    max_epoch = curve_epoch_max(marker["model"], marker_base_method(marker["method"]))
    if max_epoch is None:
        return False
    return float(marker["epoch"]) <= max_epoch


def endpoint_status(model: str, method: str, last_epoch: float) -> str | None:
    if last_epoch >= 799:
        return None
    state = INCOMPLETE_STATUS.get((model, method), "incomplete")
    return f"{state} @{int(last_epoch)}"


def baseline_bests() -> dict[str, dict[str, dict[str, float | int]]]:
    bests: dict[str, dict[str, dict[str, float | int]]] = {}
    for model, baselines in BASELINES.items():
        bests[model] = {}
        for name, path in baselines.items():
            if not path.exists():
                continue
            df = read_csv(path)
            ap = metric_col(df, "metrics/mAP50-95(B)", "metrics/mAP50-95")
            ap50 = metric_col(df, "metrics/mAP50(B)", "metrics/mAP50")
            if not ap or not ap50:
                continue
            ap_ep, ap_best = best_metric(df, ap)
            ap50_ep, ap50_best = best_metric(df, ap50)
            bests[model][name] = {
                "best_ap": ap_best,
                "best_ap_epoch": ap_ep,
                "best_ap50": ap50_best,
                "best_ap50_epoch": ap50_ep,
            }
    return bests


def add_baseline_deltas(row: dict, bests: dict[str, dict[str, dict[str, float | int]]]) -> dict:
    model = row["model"]
    sar = bests.get(model, {}).get("SAR")
    rgb = bests.get(model, {}).get("RGB")
    best_ap = float(row["best_ap"])
    best_ap50 = float(row["best_ap50"])
    row["sar_baseline_best_ap"] = sar["best_ap"] if sar else ""
    row["sar_baseline_best_ap50"] = sar["best_ap50"] if sar else ""
    row["delta_best_ap_vs_sar_baseline"] = best_ap - float(sar["best_ap"]) if sar else ""
    row["delta_best_ap50_vs_sar_baseline"] = best_ap50 - float(sar["best_ap50"]) if sar else ""
    row["rgb_baseline_best_ap"] = rgb["best_ap"] if rgb else ""
    row["rgb_baseline_best_ap50"] = rgb["best_ap50"] if rgb else ""
    row["gap_best_ap_to_rgb_baseline"] = best_ap - float(rgb["best_ap"]) if rgb else ""
    row["gap_best_ap50_to_rgb_baseline"] = best_ap50 - float(rgb["best_ap50"]) if rgb else ""
    return row


def fmt_float(v, ndigits: int = 5) -> str:
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return ""


def write_markdown_table(rows: list[dict]):
    path = ROOT / "performance_table.md"
    keep = [
        r
        for r in rows
        if not str(r["method"]).endswith("baseline")
        and r["method"] not in {"Static-final snapshot"}
    ]
    keep = sorted(keep, key=lambda r: (r["model"], str(r["method"])))
    lines = [
        "# Mosaic100 LADD Mainline Performance Table",
        "",
        "Delta columns are computed against the same-capacity SAR baseline best value. "
        "`gap_to_RGB` is best AP minus same-capacity RGB baseline best AP.",
        "",
        "| Model | Method | Epochs | Last AP | Best AP @ epoch | Delta AP vs SAR | Gap AP to RGB | Last AP50 | Best AP50 @ epoch | Delta AP50 vs SAR | Gap AP50 to RGB | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in keep:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"YOLO11{r['model']}",
                    str(r["method"]),
                    str(r["epochs_recorded"]),
                    fmt_float(r["last_ap"]),
                    f"{fmt_float(r['best_ap'])} @{r['best_ap_epoch']}",
                    fmt_float(r["delta_best_ap_vs_sar_baseline"]),
                    fmt_float(r["gap_best_ap_to_rgb_baseline"]),
                    fmt_float(r["last_ap50"]),
                    f"{fmt_float(r['best_ap50'])} @{r['best_ap50_epoch']}",
                    fmt_float(r["delta_best_ap50_vs_sar_baseline"]),
                    fmt_float(r["gap_best_ap50_to_rgb_baseline"]),
                    str(r.get("note", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(path)


def plot_baseline_lines(ax, model: str, metric: str):
    for name, path in BASELINES[model].items():
        if not path.exists():
            continue
        df = read_csv(path)
        col = metric_col(df, metric)
        if col is None:
            continue
        ep, best = best_metric(df, col)
        style = BASELINE_STYLE[name]
        color = style["color"]
        ls = style["linestyle"]
        ax.axhline(best, color=color, linestyle=ls, linewidth=1.1, alpha=0.9)
        ax.text(
            875,
            best,
            f"{name} best {best:.3f}@{ep}",
            va="center",
            ha="left",
            fontsize=7,
            color=color,
        )


def plot_baseline_curves(ax_ap, ax_ap50, ax_det, model: str):
    for name, path in BASELINES[model].items():
        if not path.exists():
            continue
        df = read_csv(path)
        x = df["epoch"]
        style = BASELINE_STYLE[name]
        label = f"{name} baseline"
        ap_col = metric_col(df, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        ap50_col = metric_col(df, "metrics/mAP50(B)", "metrics/mAP50")
        if ap_col:
            ax_ap.plot(
                x,
                df[ap_col],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                linewidth=1.25,
            )
        if ap50_col:
            ax_ap50.plot(
                x,
                df[ap50_col],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                linewidth=1.25,
            )
        baseline_det = det_loss(df)
        if baseline_det.notna().any():
            ax_det.plot(
                x,
                smooth(baseline_det),
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                linewidth=1.2,
            )


def plot_model(model: str):
    specs = [r for r in RUNS if r.model == model and r.path.exists()]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.2), sharex=True)
    ax_ap, ax_ap50, ax_det, ax_kd, ax_reach, ax_aux = axes.flatten()
    plot_baseline_curves(ax_ap, ax_ap50, ax_det, model)

    for spec in specs:
        df = read_csv(spec.path)
        x = df["epoch"]
        color = COLORS.get(spec.method, "#333333")
        ls = LINESTYLES.get(spec.method, "-")
        status_label = endpoint_status(model, spec.method, float(x.max()))
        ap_col = metric_col(df, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        ap50_col = metric_col(df, "metrics/mAP50(B)", "metrics/mAP50")
        if ap_col:
            ax_ap.plot(x, df[ap_col], label=spec.label, color=color, linestyle=ls, linewidth=1.8)
            if status_label:
                ax_ap.scatter(
                    [float(x.iloc[-1])],
                    [float(df[ap_col].iloc[-1])],
                    marker="o",
                    s=38,
                    color=color,
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=6,
                )
                ax_ap.annotate(
                    status_label,
                    (float(x.iloc[-1]), float(df[ap_col].iloc[-1])),
                    xytext=(7, 8),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    va="bottom",
                )
            ext = [(ep, ap, ap50, name) for ep, ap, ap50, name in performance_extension_points(model, spec.method) if ep > float(x.max())]
            if ext:
                ext_x = [float(x.iloc[-1])] + [p[0] for p in ext]
                ext_y = [float(df[ap_col].iloc[-1])] + [p[1] for p in ext]
                ax_ap.plot(ext_x, ext_y, color=color, linestyle="-", linewidth=2.4, alpha=0.95)
        if ap50_col:
            ax_ap50.plot(x, df[ap50_col], label=spec.label, color=color, linestyle=ls, linewidth=1.8)
            if status_label:
                ax_ap50.scatter(
                    [float(x.iloc[-1])],
                    [float(df[ap50_col].iloc[-1])],
                    marker="o",
                    s=38,
                    color=color,
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=6,
                )
                ax_ap50.annotate(
                    status_label,
                    (float(x.iloc[-1]), float(df[ap50_col].iloc[-1])),
                    xytext=(7, -12),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    va="top",
                )
            ext = [(ep, ap, ap50, name) for ep, ap, ap50, name in performance_extension_points(model, spec.method) if ep > float(x.max())]
            if ext:
                ext_x = [float(x.iloc[-1])] + [p[0] for p in ext]
                ext_y = [float(df[ap50_col].iloc[-1])] + [p[2] for p in ext]
                ax_ap50.plot(ext_x, ext_y, color=color, linestyle="-", linewidth=2.4, alpha=0.95)
        ax_det.plot(x, smooth(det_loss(df)), label=spec.label, color=color, linestyle=ls, linewidth=1.5)
        if "train/kd_loss" in df.columns:
            ax_kd.plot(x, smooth(df["train/kd_loss"]), label=spec.label, color=color, linestyle=ls, linewidth=1.5)
        rl = reach_loss(df)
        if rl.notna().any():
            ax_reach.plot(x, smooth(rl), label=spec.label, color=color, linestyle=ls, linewidth=1.5)
        al = task_rec_loss(df)
        if al.notna().any():
            ax_aux.plot(x, smooth(al), label=spec.label, color=color, linestyle=ls, linewidth=1.5)

    plot_baseline_lines(ax_ap, model, "metrics/mAP50-95(B)")
    plot_baseline_lines(ax_ap50, model, "metrics/mAP50(B)")

    for m in load_markers():
        if m["model"] != model:
            continue
        if marker_is_covered_by_curve(m):
            continue
        method = m["method"]
        if method.endswith("-final"):
            marker_label = "Static final"
        else:
            marker_label = method
        color = COLORS.get(marker_base_method(method), "#333333")
        epoch = float(m["epoch"])
        ap = float(m["ap"])
        ap50 = float(m["ap50"])
        ax_ap.scatter([epoch], [ap], marker="*", s=95, color=color, edgecolor="black", linewidth=0.4, zorder=5)
        ax_ap50.scatter([epoch], [ap50], marker="*", s=95, color=color, edgecolor="black", linewidth=0.4, zorder=5)
        ax_ap.annotate(
            f"{marker_label} {ap:.3f}",
            (epoch, ap),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color=color,
            va="top",
        )
        ax_ap50.annotate(
            f"{marker_label} {ap50:.3f}",
            (epoch, ap50),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color=color,
            va="top",
        )

    panels = [
        (ax_ap, "AP (mAP50-95)", "AP"),
        (ax_ap50, "AP50", "AP50"),
        (ax_det, "train box+cls+dfl", "det loss"),
        (ax_kd, "train KD", "KD loss"),
        (ax_reach, "train reach match+rank", "reach loss"),
        (ax_aux, "train task+s_rec+t_rec", "task/rec loss"),
    ]
    for ax, ylabel, label in panels:
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.text(0.01, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("B-stage epoch")
    for ax in [ax_ap, ax_ap50]:
        ax.set_xlim(0, 900)
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend([h for h, _ in uniq], [l for _, l in uniq], loc="best", frameon=False)

    missing = {
        "n": "LADD is the 4090 run interrupted at epoch 347; the green curve is short because the experiment has not been resumed.",
        "s": "Static and LADD are completed. Dynamic was resumed on 4090 and is shown up to the latest synced epoch.",
    }[model]
    fig.text(
        0.01,
        0.01,
        f"YOLO11{model} mosaic100 protocol. Thin dashed/dotted curves: same-capacity SAR/RGB baselines; horizontal lines: their best AP. {missing}",
        fontsize=8,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ["png", "pdf"]:
        out = OUT / f"ladd_mosaic100_yolo11{model}_mainline_curves.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(out)
    plt.close(fig)


def write_summary():
    rows = []
    bests = baseline_bests()
    for spec in RUNS:
        if not spec.path.exists():
            continue
        df = read_csv(spec.path)
        ap = metric_col(df, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        ap50 = metric_col(df, "metrics/mAP50(B)", "metrics/mAP50")
        if not ap:
            continue
        best_ep, best_ap = best_metric(df, ap)
        best50_ep, best_ap50 = best_metric(df, ap50) if ap50 else (None, None)
        last = df.iloc[-1]
        rows.append(
            add_baseline_deltas(
                {
                "model": spec.model,
                "method": spec.method,
                "source": spec.path.name,
                "epochs_recorded": int(last["epoch"]),
                "last_ap": float(last[ap]),
                "last_ap50": float(last[ap50]) if ap50 else "",
                "best_ap": best_ap,
                "best_ap_epoch": best_ep,
                "best_ap50": best_ap50,
                "best_ap50_epoch": best50_ep,
                "note": spec.curve_note,
                },
                bests,
            )
        )
    for model, bs in BASELINES.items():
        for name, path in bs.items():
            if not path.exists():
                continue
            df = read_csv(path)
            ap = metric_col(df, "metrics/mAP50-95(B)", "metrics/mAP50-95")
            ap50 = metric_col(df, "metrics/mAP50(B)", "metrics/mAP50")
            best_ep, best_ap = best_metric(df, ap)
            best50_ep, best_ap50 = best_metric(df, ap50)
            last = df.iloc[-1]
            rows.append(
                add_baseline_deltas(
                    {
                    "model": model,
                    "method": f"{name} baseline",
                    "source": path.name,
                    "epochs_recorded": int(last["epoch"]),
                    "last_ap": float(last[ap]),
                    "last_ap50": float(last[ap50]),
                    "best_ap": best_ap,
                    "best_ap_epoch": best_ep,
                    "best_ap50": best_ap50,
                    "best_ap50_epoch": best50_ep,
                    "note": "baseline reference line",
                    },
                    bests,
                )
            )
    for marker in load_markers():
        if marker_is_covered_by_curve(marker):
            continue
        rows.append(
            add_baseline_deltas(
                {
                "model": marker["model"],
                "method": f"{marker['method']} snapshot",
                "source": "manual_snapshot_markers.csv",
                "epochs_recorded": marker["epoch"],
                "last_ap": marker["ap"],
                "last_ap50": marker["ap50"],
                "best_ap": marker["ap"],
                "best_ap_epoch": marker["epoch"],
                "best_ap50": marker["ap50"],
                "best_ap50_epoch": marker["epoch"],
                "note": marker["note"],
                },
                bests,
            )
        )
    out = ROOT / "summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)
    write_markdown_table(rows)


def main():
    write_summary()
    plot_model("n")
    plot_model("s")


if __name__ == "__main__":
    main()
