#!/usr/bin/env python3
"""Generate SVG diagnostics for OGSOD YOLO-init method-search curves.

This script intentionally uses only the Python standard library because the
local desktop Python environment may not have matplotlib installed.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_SVG = ROOT / "ogsod_yoloinit_dynamic_vs_baselines_20260624.svg"
OUT_SUMMARY = ROOT / "ogsod_yoloinit_dynamic_vs_baselines_20260624_summary.txt"


def metric(row: dict[str, str], names: tuple[str, ...]) -> float:
    stripped = {k.strip(): v for k, v in row.items()}
    for name in names:
        if row.get(name):
            return float(row[name])
        if stripped.get(name):
            return float(stripped[name])
    raise KeyError(names)


def read_curve(path: Path) -> tuple[list[int], list[float], list[float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = list(range(1, len(rows) + 1))
    ap5095 = [metric(row, ("metrics/mAP50-95(B)", "metrics/mAP50-95")) for row in rows]
    ap50 = [metric(row, ("metrics/mAP50(B)", "metrics/mAP50")) for row in rows]
    return epochs, ap5095, ap50


def delta(candidate: list[float], control: list[float]) -> tuple[list[int], list[float]]:
    n = min(len(candidate), len(control))
    return list(range(1, n + 1)), [candidate[i] - control[i] for i in range(n)]


def path_points(xs: list[int], ys: list[float], panel: dict[str, float]) -> str:
    x0, y0, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
    xmin, xmax, ymin, ymax = panel["xmin"], panel["xmax"], panel["ymin"], panel["ymax"]
    pts = []
    for x, y in zip(xs, ys):
        if x < xmin or x > xmax:
            continue
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="#222">{html.escape(text)}</text>'
    )


def draw_panel(panel: dict[str, float], title: str, xlabel: str, ylabel: str, yticks: list[float], xticks: list[int]) -> list[str]:
    x0, y0, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
    xmin, xmax, ymin, ymax = panel["xmin"], panel["xmax"], panel["ymin"], panel["ymax"]
    out = []
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fff" stroke="#222" stroke-width="1"/>')
    for t in yticks:
        py = y0 + h - (t - ymin) / (ymax - ymin) * h
        out.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+w}" y2="{py:.1f}" stroke="#ddd" stroke-width="1"/>')
        out.append(svg_text(x0 - 8, py + 4, f"{t:.3f}" if abs(t) < 0.1 else f"{t:.2f}", 11, "end"))
    for t in xticks:
        px = x0 + (t - xmin) / (xmax - xmin) * w
        out.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+h}" stroke="#eee" stroke-width="1"/>')
        out.append(svg_text(px, y0 + h + 18, str(t), 11, "middle"))
    out.append(svg_text(x0 + w / 2, y0 - 16, title, 15, "middle", "700"))
    out.append(svg_text(x0 + w / 2, y0 + h + 40, xlabel, 13, "middle"))
    out.append(
        f'<text x="{x0 - 55:.1f}" y="{y0 + h / 2:.1f}" font-size="13" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="middle" '
        f'transform="rotate(-90 {x0 - 55:.1f} {y0 + h / 2:.1f})" fill="#222">{html.escape(ylabel)}</text>'
    )
    return out


def draw_curve(panel: dict[str, float], xs: list[int], ys: list[float], color: str, label: str, width: float = 2.0, dash: str = "", opacity: float = 1.0) -> str:
    points = path_points(xs, ys, panel)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}>'
        f'<title>{html.escape(label)}</title></polyline>'
    )


def draw_legend(x: float, y: float, items: list[tuple[str, str, str, float]]) -> list[str]:
    out = []
    for i, (label, color, dash, width) in enumerate(items):
        yy = y + i * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x+26:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>')
        out.append(svg_text(x + 33, yy + 4, label, 11))
    return out


def best_latest(label: str, ys: list[float]) -> str:
    best = max(ys)
    idx = ys.index(best) + 1
    return f"{label}: rows={len(ys)}, latest={ys[-1]:.5f}, best={best:.5f}@{idx}"


def main() -> None:
    curves = {
        "SAR baseline": read_curve(DATA / "baselines" / "sar_yolo11n_formal_nomosaic_e800_s0.csv"),
        "RGB baseline": read_curve(DATA / "baselines" / "rgb_yolo11n_formal_nomosaic_e800_s0.csv"),
        "4090 det-only": read_curve(DATA / "4090" / "detonly_e800.csv"),
        "4090 ProbeA": read_curve(DATA / "4090" / "probeA_e800.csv"),
        "4090 Dynamic e800": read_curve(DATA / "4090" / "dynamic_e800.csv"),
        "4090 oldcommit ProbeA": read_curve(DATA / "4090" / "oldcommit_probeA_e700.csv"),
        "3090 det-only ctrl": read_curve(DATA / "3090" / "detonly_control.csv"),
        "3090 singleproj": read_curve(DATA / "3090" / "singleproj.csv"),
        "3090 wo_s_rec": read_curve(DATA / "3090" / "wo_s_rec.csv"),
        "3090 wo_reach stopped": read_curve(DATA / "3090" / "wo_reach_stopped.csv"),
        "3090 dynamic_plain": read_curve(DATA / "3090" / "dynamic_plain.csv"),
    }

    panels = [
        {"x": 90, "y": 70, "w": 485, "h": 330, "xmin": 0, "xmax": 800, "ymin": 0.0, "ymax": 0.66},
        {"x": 700, "y": 70, "w": 485, "h": 330, "xmin": 0, "xmax": 400, "ymin": -0.006, "ymax": 0.024},
        {"x": 1310, "y": 70, "w": 485, "h": 330, "xmin": 0, "xmax": 200, "ymin": -0.012, "ymax": 0.014},
    ]

    colors = {
        "SAR baseline": "#111111",
        "RGB baseline": "#666666",
        "4090 det-only": "#4C78A8",
        "4090 ProbeA": "#72B7B2",
        "4090 Dynamic e800": "#E45756",
        "4090 oldcommit ProbeA": "#F58518",
        "3090 det-only ctrl": "#B279A2",
        "3090 singleproj": "#54A24B",
        "3090 wo_s_rec": "#C9A300",
        "3090 wo_reach stopped": "#9D755D",
        "3090 dynamic_plain": "#FF9DA6",
    }

    svg: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1860" height="500" viewBox="0 0 1860 500">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    svg += draw_panel(panels[0], "Absolute AP50-95 curves", "Epoch", "AP50-95", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 100, 200, 400, 600, 800])
    for name in [
        "SAR baseline",
        "RGB baseline",
        "4090 det-only",
        "4090 ProbeA",
        "4090 Dynamic e800",
        "4090 oldcommit ProbeA",
        "3090 det-only ctrl",
        "3090 singleproj",
        "3090 wo_s_rec",
        "3090 wo_reach stopped",
        "3090 dynamic_plain",
    ]:
        xs, ys, _ = curves[name]
        dash = "6 4" if "baseline" in name else ("3 4" if "oldcommit" in name or "stopped" in name else "")
        width = 3.0 if name == "4090 Dynamic e800" else (2.3 if name in ("SAR baseline", "RGB baseline", "4090 det-only") else 1.8)
        opacity = 0.92 if name != "3090 dynamic_plain" else 0.75
        svg.append(draw_curve(panels[0], xs, ys, colors[name], name, width, dash, opacity))
    svg += draw_legend(320, 188, [
        ("SAR formal baseline", colors["SAR baseline"], "6 4", 2.3),
        ("RGB formal baseline", colors["RGB baseline"], "6 4", 2.3),
        ("4090 det-only", colors["4090 det-only"], "", 2.3),
        ("4090 Dynamic e800", colors["4090 Dynamic e800"], "", 3.0),
        ("4090 ProbeA", colors["4090 ProbeA"], "", 1.8),
        ("4090 oldcommit ProbeA", colors["4090 oldcommit ProbeA"], "3 4", 1.8),
        ("3090 candidates", "#54A24B", "", 1.8),
    ])

    svg += draw_panel(panels[1], "4090 delta vs same-pipeline det-only", "Matched epoch", "Delta AP50-95", [-0.005, 0, 0.005, 0.010, 0.015, 0.020], [0, 100, 200, 300, 400])
    svg.append(f'<line x1="{panels[1]["x"]}" y1="{panels[1]["y"] + panels[1]["h"] - (0.010 - panels[1]["ymin"]) / (panels[1]["ymax"] - panels[1]["ymin"]) * panels[1]["h"]:.1f}" x2="{panels[1]["x"] + panels[1]["w"]}" y2="{panels[1]["y"] + panels[1]["h"] - (0.010 - panels[1]["ymin"]) / (panels[1]["ymax"] - panels[1]["ymin"]) * panels[1]["h"]:.1f}" stroke="#E45756" stroke-width="1" stroke-dasharray="4 4"/>')
    ctrl = curves["4090 det-only"][1]
    for name in ["4090 ProbeA", "4090 Dynamic e800", "4090 oldcommit ProbeA"]:
        xs, ds = delta(curves[name][1], ctrl)
        svg.append(draw_curve(panels[1], xs, ds, colors[name], name, 3.0 if name == "4090 Dynamic e800" else 2.0, "3 4" if "oldcommit" in name else ""))
    svg += draw_legend(955, 93, [
        ("ProbeA", colors["4090 ProbeA"], "", 2.0),
        ("Dynamic e800", colors["4090 Dynamic e800"], "", 3.0),
        ("oldcommit ProbeA", colors["4090 oldcommit ProbeA"], "3 4", 2.0),
        ("+0.010 screen line", colors["4090 Dynamic e800"], "4 4", 1.0),
    ])

    svg += draw_panel(panels[2], "3090 delta vs same-pipeline det-only", "Matched epoch", "Delta AP50-95", [-0.010, -0.005, 0, 0.005, 0.010], [0, 50, 100, 150, 200])
    svg.append(f'<line x1="{panels[2]["x"]}" y1="{panels[2]["y"] + panels[2]["h"] - (0.010 - panels[2]["ymin"]) / (panels[2]["ymax"] - panels[2]["ymin"]) * panels[2]["h"]:.1f}" x2="{panels[2]["x"] + panels[2]["w"]}" y2="{panels[2]["y"] + panels[2]["h"] - (0.010 - panels[2]["ymin"]) / (panels[2]["ymax"] - panels[2]["ymin"]) * panels[2]["h"]:.1f}" stroke="#E45756" stroke-width="1" stroke-dasharray="4 4"/>')
    ctrl = curves["3090 det-only ctrl"][1]
    for name in ["3090 singleproj", "3090 wo_s_rec", "3090 wo_reach stopped", "3090 dynamic_plain"]:
        xs, ds = delta(curves[name][1], ctrl)
        svg.append(draw_curve(panels[2], xs, ds, colors[name], name, 2.3 if name == "3090 wo_s_rec" else 1.9, "3 4" if "stopped" in name else ""))
    svg += draw_legend(1565, 267, [
        ("singleproj", colors["3090 singleproj"], "", 1.9),
        ("wo_s_rec", colors["3090 wo_s_rec"], "", 2.3),
        ("wo_reach stopped", colors["3090 wo_reach stopped"], "3 4", 1.9),
        ("dynamic_plain", colors["3090 dynamic_plain"], "", 1.9),
    ])

    svg.append("</svg>")
    OUT_SVG.write_text("\n".join(svg))

    lines = [best_latest(name, ys) for name, (_, ys, _) in curves.items()]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_SVG}")
    print(f"Saved {OUT_SUMMARY}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
