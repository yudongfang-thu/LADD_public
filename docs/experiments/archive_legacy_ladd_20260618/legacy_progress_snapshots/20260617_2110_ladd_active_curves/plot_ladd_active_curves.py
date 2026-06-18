from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "raw_csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_curve(path: Path):
    rows = []
    with path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            row = {k.strip(): v for k, v in row.items()}
            try:
                epoch = int(float(row.get("epoch") or row.get("Epoch") or i))
            except Exception:
                epoch = i

            def get_float(keys):
                for key in keys:
                    value = row.get(key)
                    if value not in (None, ""):
                        try:
                            return float(value)
                        except ValueError:
                            pass
                return None

            ap = get_float(["metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95"])
            ap50 = get_float(["metrics/mAP50(B)", "metrics/mAP50", "map50"])
            if ap is not None:
                rows.append({"epoch": epoch, "ap": ap, "ap50": ap50})
    return rows


def baseline_lookup(curve):
    return {p["epoch"]: p["ap"] for p in curve}


series = {
    "n": [
        ("SAR baseline", "n_baseline_sar_results.csv", "black", "-", 2.2),
        ("RGB baseline", "n_baseline_rgb_results.csv", "#7f7f7f", ":", 2.0),
        ("LADD static", "n_static_autodl_results.csv", "#1f77b4", "-", 1.8),
        ("LADD dynamic", "n_dynamic_autodl_results.csv", "#ff7f0e", "-", 1.8),
        ("Probe-A", "n_probeA_4090_results.csv", "#2ca02c", "--", 1.7),
    ],
    "s": [
        ("SAR baseline", "s_baseline_sar_results.csv", "black", "-", 2.2),
        ("RGB baseline", "s_baseline_rgb_results.csv", "#7f7f7f", ":", 2.0),
        ("LADD static", "s_static_4090_results.csv", "#1f77b4", "-", 1.8),
        ("LADD dynamic", "s_dynamic_4090_results.csv", "#ff7f0e", "-", 1.8),
        ("Probe-A", "s_probeA_autodl_results.csv", "#2ca02c", "--", 1.7),
    ],
}


curves = {}
for model, specs in series.items():
    curves[model] = {}
    for label, filename, *_ in specs:
        path = CSV_DIR / filename
        if path.exists():
            curves[model][label] = read_curve(path)


plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 160,
    "savefig.dpi": 220,
})


def plot_ap():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, model in zip(axes, ["n", "s"]):
        for label, filename, color, linestyle, linewidth in series[model]:
            curve = curves[model].get(label, [])
            if not curve:
                continue
            xs = [p["epoch"] for p in curve]
            ys = [p["ap"] for p in curve]
            ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            ax.scatter(xs[-1], ys[-1], color=color, s=18, zorder=3)
            ax.text(xs[-1] + 6, ys[-1], f"{ys[-1]:.3f}", color=color, fontsize=8, va="center")
        ax.set_title(f"YOLO11{model} mosaic100 b64")
        ax.set_xlabel("B-stage epoch")
        ax.set_xlim(0, 805)
        ax.set_ylim(0.40, 0.675)
        if model == "n":
            ax.set_ylabel("mAP50-95(B)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ladd_active_ap_curves_by_model.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ladd_active_ap_curves_by_model.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_delta():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
    for ax, model in zip(axes, ["n", "s"]):
        base = baseline_lookup(curves[model].get("SAR baseline", []))
        for label, filename, color, linestyle, linewidth in series[model]:
            if label == "SAR baseline":
                continue
            curve = curves[model].get(label, [])
            xs, ys = [], []
            for p in curve:
                b = base.get(p["epoch"])
                if b is None:
                    continue
                xs.append(p["epoch"])
                ys.append(p["ap"] - b)
            if not xs:
                continue
            ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            ax.scatter(xs[-1], ys[-1], color=color, s=18, zorder=3)
            ax.text(xs[-1] + 6, ys[-1], f"{ys[-1]:+.3f}", color=color, fontsize=8, va="center")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.set_title(f"YOLO11{model}: same-epoch delta vs SAR baseline")
        ax.set_xlabel("B-stage epoch")
        ax.set_xlim(0, 805)
        if model == "n":
            ax.set_ylabel("Delta mAP50-95(B)")
            ax.set_ylim(-0.05, 0.12)
        else:
            ax.set_ylim(-0.05, 0.10)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ladd_active_delta_vs_baseline_by_model.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ladd_active_delta_vs_baseline_by_model.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary():
    lines = ["model,label,epochs,last_ap,best_ap,best_epoch,last_delta_vs_baseline"]
    for model in ["n", "s"]:
        base = baseline_lookup(curves[model].get("SAR baseline", []))
        for label, *_ in series[model]:
            curve = curves[model].get(label, [])
            if not curve:
                continue
            last = curve[-1]
            best = max(curve, key=lambda p: p["ap"])
            delta = ""
            if label != "SAR baseline" and last["epoch"] in base:
                delta = f"{last['ap'] - base[last['epoch']]:.6f}"
            lines.append(
                f"{model},{label},{len(curve)},{last['ap']:.6f},{best['ap']:.6f},{best['epoch']},{delta}"
            )
    (ROOT / "summary.csv").write_text("\n".join(lines) + "\n")


plot_ap()
plot_delta()
write_summary()
print(f"saved {FIG_DIR}")
