#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(v: str | float | int, ndigits: int = 5) -> str:
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(v)


def row(
    evidence_group: str,
    model: str,
    protocol: str,
    schedule: str,
    run_key: str,
    label: str,
    init_source: str,
    rows_recorded: str | int,
    best_ap: str | float,
    best_epoch: str | int,
    last_ap: str | float,
    status: str,
    caveat: str,
    source: str,
) -> dict[str, str]:
    return {
        "evidence_group": evidence_group,
        "model": model,
        "protocol": protocol,
        "schedule": schedule,
        "run_key": run_key,
        "label": label,
        "init_source": init_source,
        "rows_recorded": str(rows_recorded),
        "best_ap": fmt(best_ap),
        "best_epoch": str(best_epoch),
        "last_ap": fmt(last_ap),
        "status": status,
        "caveat": caveat,
        "source": source,
    }


def collect_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    b800_path = ROOT / "ladd/results/b800_restart_20260614/summary/ladd_b800_restart_curve_summary_20260614.csv"
    b800 = {r["key"]: r for r in read_rows(b800_path)}
    for key, init_source, status in [
        ("N0_yoloinit_detonly_B800sched", "YOLO init detector, det-only control", "running snapshot"),
        ("N1_basebest_continue_B800sched", "converged SAR baseline best checkpoint, det-only control", "running snapshot"),
        ("N1_baselast_continue_B800sched", "converged SAR baseline last checkpoint, det-only control", "running snapshot"),
        ("N2_a2best_continue_B800sched", "A2 best full checkpoint, full LADD B", "stopped, NaN"),
        ("N2_a2last_continue_B800sched", "A2 last full checkpoint, full LADD B", "stopped, NaN"),
        ("N3_yoloinit_a2last_decomp_B800sched", "YOLO init detector + A2 teacher decomposition", "running snapshot"),
        ("N4_yoloinit_a2last_decomp_kdwarmup_B800sched", "YOLO init detector + A2 teacher decomposition + KD warmup", "running snapshot"),
    ]:
        r = b800[key]
        rows.append(
            row(
                "no-mosaic B800 restart",
                "n",
                "formal no-mosaic",
                "B800 cosine",
                key,
                r["label"],
                init_source,
                r["epochs_recorded"],
                r["best_ap"],
                r["best_epoch"],
                r["last_finite_ap"],
                status,
                r["note"],
                r["source"],
            )
        )

    entrance_path = ROOT / "docs/experiments/figures/ladd_b_stage_historical_compare_20260614/b_stage_historical_compare_summary_20260614.csv"
    entrance = {r["key"]: r for r in read_rows(entrance_path)}
    for key, model, init_source, schedule in [
        ("N1_current_base_cont", "n", "converged SAR baseline checkpoint, det-only B", "B100 compressed"),
        ("N2_current_a2best_cont", "n", "A2 best checkpoint", "B100 compressed"),
        ("N3_current_sarbase_a2last_decomp", "n", "converged SAR baseline detector + A2 teacher decomposition", "B100 compressed"),
        ("N4_current_kd_ramp", "n", "converged SAR baseline detector + A2 teacher decomposition + KD ramp", "B120 compressed"),
        ("S1_current_base_cont", "s", "converged SAR baseline checkpoint, det-only B", "B100 compressed"),
        ("S2_current_a2best_cont", "s", "A2 best checkpoint", "B100 compressed"),
        ("S3_current_sarbase_a2last_decomp", "s", "converged SAR baseline detector + A2 teacher decomposition", "B100 compressed"),
        ("S4_current_kd_ramp", "s", "converged SAR baseline detector + A2 teacher decomposition + KD ramp", "B120 compressed"),
    ]:
        r = entrance[key]
        rows.append(
            row(
                "no-mosaic B entrance compressed",
                model,
                "formal no-mosaic",
                schedule,
                key,
                r["label"],
                init_source,
                r["epochs"],
                r["best"],
                r["best_epoch"],
                r["last"],
                "completed compressed entrance",
                "Short compressed schedule; useful for entrance trend, not final B800 claim.",
                r["source"],
            )
        )

    split_path = ROOT / "docs/experiments/figures/ladd_capacity_protocol_split_20260617/capacity_protocol_split_summary_20260617.csv"
    split_rows = read_rows(split_path)
    for r in split_rows:
        if "yolo-init" not in r["label"] and "SAR baseline" not in r["label"]:
            continue
        if r["group"] not in {"n_nomosaic", "s_nomosaic"}:
            continue
        rows.append(
            row(
                "protocol split yolo-init check",
                r["group"].split("_")[0],
                "formal no-mosaic",
                "mixed historical/active",
                r["label"],
                r["label"],
                "YOLO init A1/B diagnostic" if "yolo-init" in r["label"] else "same-protocol SAR baseline",
                r["rows"],
                r["best_ap"],
                r["best_epoch"],
                r["last_ap"],
                "historical/diagnostic",
                "Some LADD-like rows include sep/aux contamination; use as directional evidence only.",
                "docs/experiments/figures/ladd_capacity_protocol_split_20260617/capacity_protocol_split_summary_20260617.csv",
            )
        )

    current_path = ROOT / "docs/experiments/ladd_mosaic100_mainline_curves_20260618/summary.csv"
    if current_path.exists():
        for r in read_rows(current_path):
            if r["model"] not in {"n", "s"}:
                continue
            if r["method"] not in {"Static", "Dynamic", "Probe-A"}:
                continue
            rows.append(
                row(
                    "current mosaic100 clean A1B",
                    r["model"],
                    "mosaic100 close@100",
                    "A1 -> B 800",
                    f"{r['model']}_{r['method']}",
                    f"YOLO11{r['model']} {r['method']}",
                    "converged SAR baseline checkpoint -> A1 best -> B",
                    r["epochs_recorded"],
                    r["best_ap"],
                    r["best_ap_epoch"],
                    r["last_ap"],
                    "running/completed current mainline",
                    "Current preferred protocol; not a YOLO-init comparison.",
                    f"docs/experiments/ladd_mosaic100_mainline_curves_20260618/data/{r['source']}",
                )
            )

    return rows


def write_csv(rows: list[dict[str, str]]) -> Path:
    path = OUT_DIR / "init_source_comparison_summary.csv"
    fields = [
        "evidence_group",
        "model",
        "protocol",
        "schedule",
        "run_key",
        "label",
        "init_source",
        "rows_recorded",
        "best_ap",
        "best_epoch",
        "last_ap",
        "status",
        "caveat",
        "source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_readme(rows: list[dict[str, str]]) -> Path:
    path = OUT_DIR / "README.md"
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["evidence_group"], []).append(r)

    lines = [
        "# LADD initialization-source comparison notes",
        "",
        "Date: 2026-06-18",
        "",
        "This note consolidates earlier records about whether LADD/B-stage experiments started from a converged SAR baseline checkpoint or from YOLO initial weights. It does not introduce new training results; it indexes existing lightweight `results.csv` summaries.",
        "",
        "## Main takeaways",
        "",
        "1. We did run YOLO-init diagnostics. Under the no-mosaic B800 restart batch, YOLO-init detector runs were far below converged-SAR-baseline continuation at the available snapshots.",
        "2. The strongest short entrance evidence for split-load LADD used a converged SAR baseline detector plus A2 teacher decomposition, not YOLO init.",
        "3. The current mosaic100 clean A1B mainline uses a converged SAR baseline checkpoint as the A1 starting detector, then B starts from A1 best. It is therefore a SAR-baseline-init A1->B protocol, not a YOLO-init protocol.",
        "4. A fully fair init-source ablation would require the same mosaic100 clean A1B code path with only the initial checkpoint changed. We do not yet have that clean paired ablation; the historical YOLO-init evidence is directional and negative.",
        "",
        "## Summary CSV",
        "",
        "- `init_source_comparison_summary.csv`",
        "",
    ]

    for group, group_rows in grouped.items():
        lines.extend(
            [
                f"## {group}",
                "",
                "| model | schedule | run | init source | rows | best AP @ epoch | last AP | status | caveat |",
                "|---|---|---|---|---:|---:|---:|---|---|",
            ]
        )
        for r in group_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"YOLO11{r['model']}",
                        r["schedule"],
                        r["label"].replace("|", "/"),
                        r["init_source"].replace("|", "/"),
                        r["rows_recorded"],
                        f"{r['best_ap']} @{r['best_epoch']}",
                        r["last_ap"],
                        r["status"],
                        r["caveat"].replace("|", "/"),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- For current paper-facing LADD mainline work, converged SAR-baseline initialization is the more defensible default because it is stable, matches the current launcher, and gives clean detector capacity before decomposition/distillation.",
            "- YOLO-init remains useful as an ablation only if we explicitly want to claim the method can learn from raw YOLO pretrained weights without first training a SAR detector. The existing evidence does not support prioritizing that path.",
            "- Do not mix no-mosaic B800 restart, compressed B100/B120 entrance runs, and current mosaic100 clean A1B runs as one final performance comparison. They answer different questions.",
            "",
            "## Source files",
            "",
            "- `ladd/results/b800_restart_20260614/summary/ladd_b800_restart_curve_summary_20260614.csv`",
            "- `docs/experiments/figures/ladd_b_stage_historical_compare_20260614/b_stage_historical_compare_summary_20260614.csv`",
            "- `docs/experiments/figures/ladd_capacity_protocol_split_20260617/capacity_protocol_split_summary_20260617.csv`",
            "- `docs/experiments/ladd_mosaic100_mainline_curves_20260618/summary.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    rows = collect_rows()
    print(write_csv(rows))
    print(write_readme(rows))


if __name__ == "__main__":
    main()
