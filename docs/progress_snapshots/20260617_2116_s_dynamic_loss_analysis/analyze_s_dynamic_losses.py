from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "raw_csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    rows = []
    with path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            clean = {k.strip(): v for k, v in row.items()}
            parsed = {}
            for k, v in clean.items():
                if k in ("stage", "bn_stats_mode"):
                    parsed[k] = v
                    continue
                if v in ("", None):
                    parsed[k] = math.nan
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            parsed.setdefault("epoch", float(i))
            rows.append(parsed)
    return rows


def get(row, key):
    v = row.get(key, math.nan)
    return v if isinstance(v, (int, float)) else math.nan


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        y = get(r, key)
        if math.isfinite(y):
            xs.append(get(r, "epoch"))
            ys.append(y)
    return xs, ys


def rolling(values, window=7):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        vals = [v for v in values[lo:i + 1] if math.isfinite(v)]
        out.append(sum(vals) / len(vals) if vals else math.nan)
    return out


def align_by_epoch(rows):
    return {int(get(r, "epoch")): r for r in rows if math.isfinite(get(r, "epoch"))}


dyn = read_csv(CSV_DIR / "s_dynamic_4090_results.csv")
sta = read_csv(CSV_DIR / "s_static_4090_results.csv")
dyn_diag = read_csv(CSV_DIR / "s_dynamic_4090_ladd_diagnostics.csv")
sta_diag = read_csv(CSV_DIR / "s_static_4090_ladd_diagnostics.csv")

ap_key = "metrics/mAP50-95(B)"
ap50_key = "metrics/mAP50(B)"


def largest_negative_jumps(rows, topk=8):
    jumps = []
    prev = None
    for r in rows:
        ep = int(get(r, "epoch"))
        ap = get(r, ap_key)
        if prev and math.isfinite(ap) and math.isfinite(prev[1]):
            jumps.append((ap - prev[1], prev[0], ep, prev[1], ap))
        prev = (ep, ap)
    return sorted(jumps, key=lambda x: x[0])[:topk]


loss_keys = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/kd_loss",
    "train/reach_match_loss",
    "train/reach_rank_loss",
    "train/task_loss",
    "train/t_rec_loss",
    "train/s_rec_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/kd_loss",
    "val/reach_match_loss",
    "val/reach_rank_loss",
    "val/task_loss",
    "val/t_rec_loss",
    "val/s_rec_loss",
    "val/d_pos_mean",
    "val/d_neg_mean",
    "val/rank_gap_mean",
]


def write_jump_report():
    by_ep = align_by_epoch(dyn)
    lines = []
    lines.append("# YOLO11s dynamic AP jump diagnostics")
    lines.append("")
    lines.append("Largest one-epoch negative AP jumps:")
    lines.append("")
    lines.append("| delta_ap | prev_epoch | epoch | prev_ap | ap |")
    lines.append("|---:|---:|---:|---:|---:|")
    jumps = largest_negative_jumps(dyn)
    for delta, prev_ep, ep, prev_ap, ap in jumps:
        lines.append(f"| {delta:.6f} | {prev_ep} | {ep} | {prev_ap:.6f} | {ap:.6f} |")
    lines.append("")
    for delta, prev_ep, ep, prev_ap, ap in jumps[:5]:
        lines.append(f"## Around epoch {ep} jump {delta:.6f}")
        lines.append("")
        lines.append("| metric | e-1 | e | e+1 | e+2 |")
        lines.append("|---|---:|---:|---:|---:|")
        for key in [ap_key, ap50_key] + loss_keys:
            vals = []
            for e in [ep - 1, ep, ep + 1, ep + 2]:
                v = get(by_ep.get(e, {}), key)
                vals.append(f"{v:.6f}" if math.isfinite(v) else "")
            lines.append(f"| {key} | " + " | ".join(vals) + " |")
        lines.append("")
    (ROOT / "jump_diagnostics.md").write_text("\n".join(lines) + "\n")


def plot_main_losses():
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 12), sharex=True)
    colors = {"dynamic": "#ff7f0e", "static": "#1f77b4"}
    panels = [
        ("AP", [ap_key, ap50_key], ["AP", "AP50"]),
        ("Detection train losses", ["train/box_loss", "train/cls_loss", "train/dfl_loss"], None),
        ("LADD train losses", ["train/kd_loss", "train/reach_match_loss", "train/reach_rank_loss", "train/task_loss", "train/t_rec_loss", "train/s_rec_loss"], None),
        ("LADD validation losses", ["val/kd_loss", "val/reach_match_loss", "val/reach_rank_loss", "val/task_loss", "val/t_rec_loss", "val/s_rec_loss"], None),
    ]
    for ax, (title, keys, labels) in zip(axes, panels):
        for run_name, rows, base_color in [("dynamic", dyn, colors["dynamic"]), ("static", sta, colors["static"])]:
            for j, key in enumerate(keys):
                xs, ys = series(rows, key)
                if not ys:
                    continue
                # Smooth dense loss curves but keep AP raw.
                plot_y = ys if title == "AP" else rolling(ys, 7)
                label = f"{run_name} {labels[j] if labels else key.replace('train/', '').replace('val/', '')}"
                alpha = 0.95 if run_name == "dynamic" else 0.55
                linestyle = "-" if run_name == "dynamic" else "--"
                ax.plot(xs, plot_y, label=label, linewidth=1.4, alpha=alpha, linestyle=linestyle)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[-1].set_xlabel("B-stage epoch")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s_dynamic_vs_static_loss_overview.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "s_dynamic_vs_static_loss_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_zoom():
    # Two visible AP drop zones in the current curve.
    windows = [(430, 475), (490, 530)]
    keys = [
        ap_key,
        "train/kd_loss",
        "train/reach_match_loss",
        "train/reach_rank_loss",
        "train/task_loss",
        "train/t_rec_loss",
        "train/s_rec_loss",
        "val/kd_loss",
        "val/task_loss",
        "val/t_rec_loss",
        "val/s_rec_loss",
    ]
    fig, axes = plt.subplots(len(keys), len(windows), figsize=(12, 16), sharex=False)
    for col, (lo, hi) in enumerate(windows):
        for row, key in enumerate(keys):
            ax = axes[row][col]
            xs, ys = series(dyn, key)
            pairs = [(x, y) for x, y in zip(xs, ys) if lo <= x <= hi]
            if pairs:
                px, py = zip(*pairs)
                ax.plot(px, py, color="#ff7f0e", linewidth=1.5)
                ax.scatter(px, py, color="#ff7f0e", s=8)
            if row == 0:
                ax.set_title(f"dynamic zoom {lo}-{hi}")
            if col == 0:
                ax.set_ylabel(key.replace("metrics/", "").replace("train/", "tr/").replace("val/", "val/"), fontsize=8)
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s_dynamic_loss_zoom_drop_windows.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "s_dynamic_loss_zoom_drop_windows.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_diag():
    if not dyn_diag:
        return
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10), sharex=True)
    diag_keys = [
        ("BN running var", ["bn_running_var_mean", "bn_running_var_p95", "bn_running_var_max"]),
        ("BN running mean abs max", ["bn_running_mean_abs_max"]),
        ("Effective weights", ["effective_alpha_kd", "effective_alpha_s_rec", "effective_lambda_reach", "effective_lambda_match_inner", "effective_lambda_rank_inner"]),
        ("Nonfinite flags", ["nonfinite_metrics_or_cmdistill", "nonfinite_bn_stats", "nan_or_inf_detected"]),
    ]
    for ax, (title, keys) in zip(axes, diag_keys):
        for key in keys:
            xs, ys = series(dyn_diag, key)
            if ys:
                ax.plot(xs, ys, label=key, linewidth=1.3)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("B-stage epoch")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "s_dynamic_diagnostics_bn_weights_flags.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "s_dynamic_diagnostics_bn_weights_flags.pdf", bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

write_jump_report()
plot_main_losses()
plot_zoom()
plot_diag()
print("saved", FIG_DIR)
