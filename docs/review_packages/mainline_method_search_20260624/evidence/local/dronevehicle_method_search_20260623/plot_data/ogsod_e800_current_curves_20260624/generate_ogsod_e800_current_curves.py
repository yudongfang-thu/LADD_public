from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG_DIR = ROOT.parents[1] / "figures"

SAR_BASELINE_BEST = 0.55654
RGB_BASELINE_DOC_BEST = 0.63018

RUNS = {
    "detonly": {
        "path": DATA / "ogsod_e800_detonly_results.csv",
        "label": "YOLO-init det-only",
        "color": "#D55E00",
    },
    "probeA": {
        "path": DATA / "ogsod_e800_probeA_results.csv",
        "label": "YOLO-init ProbeA",
        "color": "#0072B2",
    },
    "dynamic": {
        "path": DATA / "ogsod_e800_dynamic_results.csv",
        "label": "YOLO-init dynamic",
        "color": "#009E73",
    },
    "rgb": {
        "path": DATA / "ogsod_rgb_baseline_results.csv",
        "label": "RGB baseline curve",
        "color": "#7A3E9D",
    },
    "sar": {
        "path": DATA / "ogsod_sar_baseline_results.csv",
        "label": "SAR baseline curve",
        "color": "#111111",
    },
}


def load_curve(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "epoch": int(float(row["epoch"])),
                        "ap50": float(row["metrics/mAP50(B)"]),
                        "ap": float(row["metrics/mAP50-95(B)"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def best(rows: list[dict[str, float]]) -> dict[str, float]:
    return max(rows, key=lambda r: r["ap"])


def common_delta(
    rows: list[dict[str, float]], control: list[dict[str, float]]
) -> tuple[list[int], list[float], list[float]]:
    by_epoch = {r["epoch"]: r for r in control}
    epochs: list[int] = []
    ap_deltas: list[float] = []
    ap50_deltas: list[float] = []
    for row in rows:
        ref = by_epoch.get(row["epoch"])
        if ref is None:
            continue
        epochs.append(int(row["epoch"]))
        ap_deltas.append(row["ap"] - ref["ap"])
        ap50_deltas.append(row["ap50"] - ref["ap50"])
    return epochs, ap_deltas, ap50_deltas


def main() -> None:
    curves = {k: load_curve(v["path"]) for k, v in RUNS.items()}

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.dpi": 240,
        }
    )

    fig, (ax, dax) = plt.subplots(
        2,
        1,
        figsize=(10.6, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    for key in ["detonly", "probeA", "dynamic"]:
        spec = RUNS[key]
        rows = curves[key]
        ax.plot(
            [r["epoch"] for r in rows],
            [r["ap"] for r in rows],
            label=spec["label"],
            color=spec["color"],
            linewidth=2.0,
        )
        last = rows[-1]
        ax.scatter([last["epoch"]], [last["ap"]], color=spec["color"], s=22, zorder=4)
        ax.text(
            last["epoch"] + 5,
            last["ap"],
            f"{last['ap']:.3f}",
            color=spec["color"],
            fontsize=9,
            va="center",
        )

    for key, alpha in [("sar", 0.95), ("rgb", 0.58)]:
        base_rows = curves[key]
        ax.plot(
            [r["epoch"] for r in base_rows],
            [r["ap"] for r in base_rows],
            label=RUNS[key]["label"],
            color=RUNS[key]["color"],
            linewidth=2.35 if key == "sar" else 1.45,
            alpha=alpha,
            zorder=3 if key == "sar" else 1,
        )
    rgb_rows = curves["rgb"]
    sar_rows = curves["sar"]
    rgb_best = best(rgb_rows)
    sar_best = best(sar_rows)
    ax.axhline(
        sar_best["ap"],
        color=RUNS["sar"]["color"],
        linestyle="--",
        linewidth=1.2,
        label=f"SAR baseline best {sar_best['ap']:.5f}",
    )
    ax.axhline(
        rgb_best["ap"],
        color=RUNS["rgb"]["color"],
        linestyle=":",
        linewidth=1.2,
        label=f"RGB baseline best {rgb_best['ap']:.5f}",
    )
    ax.set_ylabel("AP50-95")
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 0.665)
    ax.legend(loc="lower right", ncol=2)

    control = curves["detonly"]
    summary_lines = [
        "run,rows,latest_epoch,latest_ap50,latest_ap50_95,best_epoch,best_ap50,best_ap50_95,delta_vs_det_latest_ap50,delta_vs_det_latest_ap50_95,late5_delta,late10_delta,late20_delta"
    ]
    for key in ["detonly", "probeA", "dynamic", "sar", "rgb"]:
        rows = curves[key]
        b = best(rows)
        last = rows[-1]
        if key in {"probeA", "dynamic"}:
            epochs, deltas, d50 = common_delta(rows, control)
            label = RUNS[key]["label"] + " - det-only"
            dax.plot(
                epochs,
                deltas,
                label=label,
                color=RUNS[key]["color"],
                linewidth=1.9,
            )
            dax.fill_between(
                epochs,
                deltas,
                0.0,
                where=[d >= 0 for d in deltas],
                color=RUNS[key]["color"],
                alpha=0.12,
                interpolate=True,
            )
            summary_lines.append(
                ",".join(
                    [
                        key,
                        str(len(rows)),
                        str(last["epoch"]),
                        f"{last['ap50']:.6f}",
                        f"{last['ap']:.6f}",
                        str(b["epoch"]),
                        f"{b['ap50']:.6f}",
                        f"{b['ap']:.6f}",
                        f"{d50[-1]:.6f}",
                        f"{deltas[-1]:.6f}",
                        f"{mean(deltas[-5:]):.6f}",
                        f"{mean(deltas[-10:]):.6f}",
                        f"{mean(deltas[-20:]):.6f}",
                    ]
                )
            )
        else:
            summary_lines.append(
                ",".join(
                    [
                        key,
                        str(len(rows)),
                        str(last["epoch"]),
                        f"{last['ap50']:.6f}",
                        f"{last['ap']:.6f}",
                        str(b["epoch"]),
                        f"{b['ap50']:.6f}",
                        f"{b['ap']:.6f}",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            )

    dax.axhline(0.0, color="#333333", linewidth=0.9)
    dax.set_ylabel("Delta AP50-95")
    dax.set_xlabel("Epoch")
    dax.set_ylim(-0.02, 0.03)
    dax.legend(loc="upper right")

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "ogsod_e800_yoloinit_current_with_sarcurve_v2_20260624.png"
    out_pdf = FIG_DIR / "ogsod_e800_yoloinit_current_with_sarcurve_v2_20260624.pdf"
    out_csv = ROOT / "ogsod_e800_yoloinit_current_summary_20260624.csv"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    out_csv.write_text("\n".join(summary_lines) + "\n")
    print(out_png)
    print(out_pdf)
    print(out_csv)
    print(f"sar_curve_best={sar_best['ap']:.6f}@{sar_best['epoch']}")
    print(f"rgb_curve_best={rgb_best['ap']:.6f}@{rgb_best['epoch']}")

    zoom_fig, (zax, zdax) = plt.subplots(
        2,
        1,
        figsize=(10.6, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for key in ["sar", "detonly", "probeA", "dynamic", "rgb"]:
        spec = RUNS[key]
        rows = [r for r in curves[key] if r["epoch"] <= 200]
        if not rows:
            continue
        linewidth = 2.0 if key in {"detonly", "probeA", "dynamic"} else (2.35 if key == "sar" else 1.35)
        alpha = 1.0 if key in {"detonly", "probeA", "dynamic", "sar"} else 0.62
        zax.plot(
            [r["epoch"] for r in rows],
            [r["ap"] for r in rows],
            label=spec["label"],
            color=spec["color"],
            linewidth=linewidth,
            alpha=alpha,
            zorder=3 if key == "sar" else 2,
        )
        if key in {"detonly", "probeA", "dynamic"}:
            last = rows[-1]
            zax.scatter([last["epoch"]], [last["ap"]], color=spec["color"], s=24, zorder=4)
            zax.text(
                last["epoch"] + 2,
                last["ap"],
                f"{last['ap']:.3f}",
                color=spec["color"],
                fontsize=9,
                va="center",
            )

    zax.axhline(
        sar_best["ap"],
        color=RUNS["sar"]["color"],
        linestyle="--",
        linewidth=1.2,
        label=f"SAR baseline best {sar_best['ap']:.5f}",
    )
    zax.axhline(
        rgb_best["ap"],
        color=RUNS["rgb"]["color"],
        linestyle=":",
        linewidth=1.2,
        label=f"RGB baseline best {rgb_best['ap']:.5f}",
    )
    zax.set_ylabel("AP50-95")
    zax.set_xlim(0, 200)
    zax.set_ylim(0, 0.665)
    zax.legend(loc="lower right", ncol=2)

    for key in ["probeA", "dynamic"]:
        epochs, deltas, _ = common_delta(curves[key], control)
        epochs_zoom = []
        deltas_zoom = []
        for ep, d in zip(epochs, deltas):
            if ep <= 200:
                epochs_zoom.append(ep)
                deltas_zoom.append(d)
        zdax.plot(
            epochs_zoom,
            deltas_zoom,
            label=RUNS[key]["label"] + " - det-only",
            color=RUNS[key]["color"],
            linewidth=1.9,
        )
        zdax.fill_between(
            epochs_zoom,
            deltas_zoom,
            0.0,
            where=[d >= 0 for d in deltas_zoom],
            color=RUNS[key]["color"],
            alpha=0.12,
            interpolate=True,
        )
    zdax.axhline(0.0, color="#333333", linewidth=0.9)
    zdax.set_ylabel("Delta AP50-95")
    zdax.set_xlabel("Epoch")
    zdax.set_ylim(-0.02, 0.03)
    zdax.legend(loc="upper right")
    zoom_fig.tight_layout()

    out_zoom_png = FIG_DIR / "ogsod_e800_yoloinit_current_with_sarcurve_v2_zoom200_20260624.png"
    out_zoom_pdf = FIG_DIR / "ogsod_e800_yoloinit_current_with_sarcurve_v2_zoom200_20260624.pdf"
    zoom_fig.savefig(out_zoom_png, bbox_inches="tight")
    zoom_fig.savefig(out_zoom_pdf, bbox_inches="tight")
    print(out_zoom_png)
    print(out_zoom_pdf)


if __name__ == "__main__":
    main()
