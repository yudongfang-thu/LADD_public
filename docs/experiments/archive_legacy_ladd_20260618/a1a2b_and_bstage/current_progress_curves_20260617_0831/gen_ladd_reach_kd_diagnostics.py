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

RUNS = {
    "n_static": RAW / "autodl" / "n_static_clean_a1b_b.csv",
    "n_dynamic": RAW / "autodl" / "n_dynamic_clean_a1b_b.csv",
    "s_static": RAW / "4090" / "s_static_clean_a1b_b.csv",
    "s_dynamic": RAW / "4090" / "s_dynamic_clean_a1b_b.csv",
}

METRICS = [
    ("metrics/mAP50-95(B)", "AP"),
    ("train/kd_loss", "KD loss"),
    ("train/reach_match_loss", "reach match"),
    ("train/reach_rank_loss", "reach rank"),
    ("train/t_rec_loss", "teacher rec"),
    ("train/task_loss", "zt task"),
]


def read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["epoch"] = df["epoch"].astype(int)
    for col, _ in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def value_at(df: pd.DataFrame, epoch: int, col: str) -> float:
    row = df[df["epoch"] == epoch]
    if row.empty or col not in row:
        return math.nan
    return float(row.iloc[-1][col])


def make_plot(data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), sharex=True)
    model_styles = {
        "n_static": ("YOLO11n static", "#0072B2", "--"),
        "n_dynamic": ("YOLO11n dynamic", "#0072B2", "-"),
        "s_static": ("YOLO11s static", "#D55E00", "--"),
        "s_dynamic": ("YOLO11s dynamic", "#D55E00", "-"),
    }
    for ax, (col, ylabel) in zip(axes.ravel(), METRICS):
        for key, df in data.items():
            label, color, linestyle = model_styles[key]
            if col not in df:
                continue
            ax.plot(df["epoch"], df[col], label=label, color=color, linestyle=linestyle, linewidth=1.35)
            last = df.iloc[-1]
            ax.scatter([last["epoch"]], [last[col]], color=color, marker="o" if "dynamic" in key else "x", s=18)
        ax.set_ylabel(ylabel)
        ax.axvline(100, color="#999999", linestyle=":", linewidth=0.9, alpha=0.65)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].set_xlabel("B-stage epoch")
    axes[1, 1].set_xlabel("B-stage epoch")
    axes[1, 2].set_xlabel("B-stage epoch")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "ladd_reach_kd_diagnostics.png", dpi=240)
    fig.savefig(FIG / "ladd_reach_kd_diagnostics.pdf")
    plt.close(fig)


def make_tables(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model in ["n", "s"]:
        static = data[f"{model}_static"]
        dynamic = data[f"{model}_dynamic"]
        common = int(min(static["epoch"].max(), dynamic["epoch"].max()))
        for epoch in [50, 100, 150, 200, common]:
            if epoch > common:
                continue
            rec = {"model": model, "epoch": epoch}
            for col, short in METRICS:
                s_val = value_at(static, epoch, col)
                d_val = value_at(dynamic, epoch, col)
                rec[f"static_{short}"] = s_val
                rec[f"dynamic_{short}"] = d_val
                rec[f"delta_{short}_dynamic_minus_static"] = d_val - s_val
            rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "ladd_reach_kd_diagnostics.csv", index=False)

    lines = [
        "# LADD Reach/KD Diagnostics",
        "",
        "This table compares static vs dynamic B-stage losses at matched B-stage epochs.",
        "",
        "| model | epoch | AP static | AP dynamic | AP delta | KD static | KD dynamic | KD delta | reach match dynamic | reach rank dynamic | task dynamic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {model} | {epoch:d} | {aps:.5f} | {apd:.5f} | {apdelta:+.5f} | {kds:.5f} | {kdd:.5f} | {kddelta:+.5f} | {rm:.5f} | {rr:.5f} | {task:.5f} |".format(
                model=row["model"],
                epoch=int(row["epoch"]),
                aps=row["static_AP"],
                apd=row["dynamic_AP"],
                apdelta=row["delta_AP_dynamic_minus_static"],
                kds=row["static_KD loss"],
                kdd=row["dynamic_KD loss"],
                kddelta=row["delta_KD loss_dynamic_minus_static"],
                rm=row["dynamic_reach match"],
                rr=row["dynamic_reach rank"],
                task=row["dynamic_zt task"],
            )
        )
    lines.extend(
        [
            "",
            "Generated figure:",
            "",
            "- `figures/ladd_reach_kd_diagnostics.png` / `.pdf`",
        ]
    )
    (ROOT / "ladd_reach_kd_diagnostics.md").write_text("\n".join(lines) + "\n")
    return table


def main() -> None:
    data = {name: read(path) for name, path in RUNS.items()}
    make_plot(data)
    table = make_tables(data)
    print(table[["model", "epoch", "static_AP", "dynamic_AP", "delta_AP_dynamic_minus_static", "static_KD loss", "dynamic_KD loss", "delta_KD loss_dynamic_minus_static"]].to_string(index=False))
    print(FIG / "ladd_reach_kd_diagnostics.png")


if __name__ == "__main__":
    main()
