from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG_DIR = ROOT.parents[1] / "figures"


RUNS = [
    {
        "key": "img512_std",
        "label": "img512 std",
        "probe": DATA / "img512_probeA_results.csv",
        "control": DATA / "img512_detonly_results.csv",
        "baseline_best": 0.36087,
        "baseline_final": 0.35385,
        "ylim": (0.0, 0.39),
        "dylim": (-0.06, 0.035),
    },
    {
        "key": "img256_std",
        "label": "img256 std",
        "probe": DATA / "img256_std_probeA_results.csv",
        "control": DATA / "img256_std_detonly_results.csv",
        "baseline_best": 0.24956,
        "baseline_final": 0.24939,
        "ylim": (0.0, 0.285),
        "dylim": (-0.03, 0.035),
    },
    {
        "key": "img256_nomix",
        "label": "img256 no-mix auto",
        "probe": DATA / "img256_nomix_probeA_results.csv",
        "control": DATA / "img256_nomix_detonly_results.csv",
        "baseline_best": 0.24956,
        "baseline_final": 0.24939,
        "ylim": (0.0, 0.285),
        "dylim": (-0.065, 0.03),
    },
]


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


def common_delta(probe: list[dict[str, float]], control: list[dict[str, float]]) -> tuple[list[int], list[float]]:
    control_by_epoch = {r["epoch"]: r for r in control}
    epochs: list[int] = []
    deltas: list[float] = []
    for row in probe:
        ctrl = control_by_epoch.get(row["epoch"])
        if ctrl is None:
            continue
        epochs.append(int(row["epoch"]))
        deltas.append(row["ap"] - ctrl["ap"])
    return epochs, deltas


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
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
            "savefig.dpi": 220,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), sharex="col")
    colors = {
        "probe": "#0072B2",
        "control": "#D55E00",
        "baseline_best": "#555555",
        "baseline_final": "#999999",
        "delta": "#009E73",
        "zero": "#333333",
    }

    summary_lines = [
        "group,probe_final,probe_best,control_final,control_best,final_delta,best_delta,late5_delta,late10_delta,late20_delta,positive_epochs,total_common,max_delta,max_delta_epoch"
    ]

    for col, spec in enumerate(RUNS):
        probe = load_curve(spec["probe"])
        control = load_curve(spec["control"])
        ax = axes[0, col]
        dax = axes[1, col]

        ax.plot(
            [r["epoch"] for r in probe],
            [r["ap"] for r in probe],
            label="ProbeA",
            color=colors["probe"],
            linewidth=1.9,
        )
        ax.plot(
            [r["epoch"] for r in control],
            [r["ap"] for r in control],
            label="det-only",
            color=colors["control"],
            linewidth=1.7,
        )
        ax.axhline(
            spec["baseline_best"],
            color=colors["baseline_best"],
            linestyle="--",
            linewidth=1.0,
            label="RGB baseline best",
        )
        ax.axhline(
            spec["baseline_final"],
            color=colors["baseline_final"],
            linestyle=":",
            linewidth=1.0,
            label="RGB baseline final",
        )
        ax.set_ylim(*spec["ylim"])
        ax.set_ylabel("AP50-95" if col == 0 else "")
        ax.text(
            0.02,
            0.96,
            spec["label"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
        )

        epochs, deltas = common_delta(probe, control)
        dax.axhline(0.0, color=colors["zero"], linewidth=0.9)
        dax.plot(epochs, deltas, color=colors["delta"], linewidth=1.8)
        dax.fill_between(
            epochs,
            deltas,
            0.0,
            where=[d >= 0 for d in deltas],
            color=colors["delta"],
            alpha=0.15,
            interpolate=True,
        )
        dax.fill_between(
            epochs,
            deltas,
            0.0,
            where=[d < 0 for d in deltas],
            color=colors["control"],
            alpha=0.10,
            interpolate=True,
        )
        dax.set_ylim(*spec["dylim"])
        dax.set_xlabel("Epoch")
        dax.set_ylabel("ProbeA - det-only" if col == 0 else "")

        probe_best = max(probe, key=lambda r: r["ap"])
        control_best = max(control, key=lambda r: r["ap"])
        if deltas:
            max_i = max(range(len(deltas)), key=lambda i: deltas[i])
            late5 = mean(deltas[-5:])
            late10 = mean(deltas[-10:])
            late20 = mean(deltas[-20:])
            best_delta = probe_best["ap"] - control_best["ap"]
            final_delta = probe[-1]["ap"] - control[-1]["ap"]
            pos = sum(d > 0 for d in deltas)
            dax.text(
                0.02,
                0.06,
                f"final {final_delta:+.4f}, late20 {late20:+.4f}",
                transform=dax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.8},
            )
            summary_lines.append(
                ",".join(
                    [
                        spec["key"],
                        f"{probe[-1]['ap']:.6f}",
                        f"{probe_best['ap']:.6f}",
                        f"{control[-1]['ap']:.6f}",
                        f"{control_best['ap']:.6f}",
                        f"{final_delta:.6f}",
                        f"{best_delta:.6f}",
                        f"{late5:.6f}",
                        f"{late10:.6f}",
                        f"{late20:.6f}",
                        str(pos),
                        str(len(deltas)),
                        f"{deltas[max_i]:.6f}",
                        str(epochs[max_i]),
                    ]
                )
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / "dronevehicle_probeA_yoloinit_vs_detonly_three_groups_20260624.png"
    out_pdf = FIG_DIR / "dronevehicle_probeA_yoloinit_vs_detonly_three_groups_20260624.pdf"
    out_csv = ROOT / "drone_probeA_yoloinit_summary_20260624.csv"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    out_csv.write_text("\n".join(summary_lines) + "\n")
    print(out_png)
    print(out_pdf)
    print(out_csv)


if __name__ == "__main__":
    main()
