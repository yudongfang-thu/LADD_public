#!/usr/bin/env python3
"""Generate SVG diagnostics for DroneVehicle ProbeA curves.

The script intentionally avoids third-party plotting dependencies so the figures
can be regenerated in a minimal local environment.
"""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "plot_data" / "probeA_curves"
FIG_DIR = ROOT / "figures" / "probeA_curves"
AP_KEY = "metrics/mAP50-95(B)"
AP50_KEY = "metrics/mAP50(B)"


COLORS = {
    "probe": "#d62728",
    "control": "#ff7f0e",
    "rgb": "#4d4d4d",
    "teacher": "#1f77b4",
    "delta_control": "#9467bd",
    "delta_rgb": "#2ca02c",
    "grid": "#e6e6e6",
    "axis": "#333333",
    "zero": "#777777",
}


@dataclass
class Curve:
    name: str
    epochs: list[int]
    ap: list[float]
    ap50: list[float]
    color: str
    dash: str | None = None

    @property
    def best(self) -> tuple[int, float]:
        idx = max(range(len(self.ap)), key=self.ap.__getitem__)
        return self.epochs[idx], self.ap[idx]

    @property
    def final(self) -> tuple[int, float]:
        return self.epochs[-1], self.ap[-1]

    def late_mean(self, n: int) -> float:
        vals = self.ap[-min(n, len(self.ap)) :]
        return sum(vals) / len(vals)


def read_curve(filename: str, name: str, color: str, dash: str | None = None) -> Curve:
    rows: list[tuple[int, float, float]] = []
    with (DATA_DIR / filename).open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            rows.append(
                (
                    int(float(row.get("epoch", i))),
                    float(row[AP_KEY]),
                    float(row[AP50_KEY]),
                )
            )
    return Curve(
        name=name,
        epochs=[r[0] for r in rows],
        ap=[r[1] for r in rows],
        ap50=[r[2] for r in rows],
        color=color,
        dash=dash,
    )


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if vmax <= vmin:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + i * step for i in range(count)]


def polyline(points: Iterable[tuple[float, float]], **attrs: str) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    attr = " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f'<polyline points="{pts}" {attr} />'


def line(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    attr = " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attr} />'


def text(x: float, y: float, content: str, **attrs: str) -> str:
    attr = " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr}>{html.escape(content)}</text>'


def circle(x: float, y: float, r: float, **attrs: str) -> str:
    attr = " ".join(f'{k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in attrs.items())
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" {attr} />'


class Panel:
    def __init__(self, x: float, y: float, w: float, h: float, xmin: float, xmax: float, ymin: float, ymax: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def sx(self, val: float) -> float:
        return self.x + (val - self.xmin) / (self.xmax - self.xmin) * self.w

    def sy(self, val: float) -> float:
        return self.y + self.h - (val - self.ymin) / (self.ymax - self.ymin) * self.h

    def plot_points(self, xs: list[int], ys: list[float]) -> list[tuple[float, float]]:
        return [(self.sx(x), self.sy(y)) for x, y in zip(xs, ys)]


def draw_axes(panel: Panel, y_ticks: list[float], title: str, ylabel: str, xlabel: str | None = None) -> list[str]:
    out: list[str] = []
    out.append(line(panel.x, panel.y, panel.x, panel.y + panel.h, stroke=COLORS["axis"], stroke_width="1.2"))
    out.append(line(panel.x, panel.y + panel.h, panel.x + panel.w, panel.y + panel.h, stroke=COLORS["axis"], stroke_width="1.2"))
    for t in y_ticks:
        y = panel.sy(t)
        out.append(line(panel.x, y, panel.x + panel.w, y, stroke=COLORS["grid"], stroke_width="1"))
        out.append(text(panel.x - 8, y + 4, f"{t:.3f}", text_anchor="end", font_size="11", fill="#333"))
    for ep in [1, 50, 100, 150, 200]:
        x = panel.sx(ep)
        out.append(line(x, panel.y + panel.h, x, panel.y + panel.h + 5, stroke=COLORS["axis"], stroke_width="1"))
        out.append(text(x, panel.y + panel.h + 18, str(ep), text_anchor="middle", font_size="11", fill="#333"))
    out.append(text(panel.x + panel.w / 2, panel.y - 14, title, text_anchor="middle", font_size="16", font_weight="700", fill="#111"))
    out.append(text(panel.x - 54, panel.y + panel.h / 2, ylabel, text_anchor="middle", font_size="12", fill="#333", transform=f"rotate(-90 {panel.x - 54:.2f} {panel.y + panel.h / 2:.2f})"))
    if xlabel:
        out.append(text(panel.x + panel.w / 2, panel.y + panel.h + 40, xlabel, text_anchor="middle", font_size="12", fill="#333"))
    return out


def draw_legend(x: float, y: float, items: list[Curve | tuple[str, str, str | None]]) -> list[str]:
    out: list[str] = []
    dx = 0
    for item in items:
        if isinstance(item, Curve):
            label, color, dash = item.name, item.color, item.dash
        else:
            label, color, dash = item
        attrs = {"stroke": color, "stroke_width": "2.4"}
        if dash:
            attrs["stroke_dasharray"] = dash
        out.append(line(x + dx, y, x + dx + 22, y, **attrs))
        out.append(text(x + dx + 28, y + 4, label, font_size="12", fill="#222"))
        dx += 28 + len(label) * 6.6 + 22
    return out


def draw_curve(panel: Panel, curve: Curve, width: str = "2.2") -> list[str]:
    attrs = {
        "fill": "none",
        "stroke": curve.color,
        "stroke_width": width,
        "stroke_linejoin": "round",
        "stroke_linecap": "round",
    }
    if curve.dash:
        attrs["stroke_dasharray"] = curve.dash
    points = panel.plot_points(curve.epochs, curve.ap)
    best_ep, best_ap = curve.best
    return [
        polyline(points, **attrs),
        circle(panel.sx(best_ep), panel.sy(best_ap), 3.0, fill=curve.color, stroke="white", stroke_width="1"),
    ]


def make_delta_curve(name: str, a: Curve, b: Curve, color: str, dash: str | None = None) -> Curve:
    n = min(len(a.epochs), len(b.epochs))
    return Curve(
        name=name,
        epochs=a.epochs[:n],
        ap=[a.ap[i] - b.ap[i] for i in range(n)],
        ap50=[a.ap50[i] - b.ap50[i] for i in range(n)],
        color=color,
        dash=dash,
    )


def summarize(size: str, curves: dict[str, Curve]) -> str:
    probe = curves["probe"]
    control = curves["control"]
    rgb = curves["rgb"]
    d_control = make_delta_curve("ProbeA - det-only", probe, control, COLORS["delta_control"])
    final_delta = probe.final[1] - control.final[1]
    best_delta = probe.best[1] - control.best[1]
    late20_delta = probe.late_mean(20) - control.late_mean(20)
    base_best_delta = probe.best[1] - rgb.best[1]
    base_final_delta = probe.final[1] - rgb.final[1]
    return (
        f"{size}: ProbeA best/final {probe.best[1]:.5f}/{probe.final[1]:.5f}; "
        f"control best/final {control.best[1]:.5f}/{control.final[1]:.5f}; "
        f"RGB baseline best/final {rgb.best[1]:.5f}/{rgb.final[1]:.5f}; "
        f"Probe-control final/best/late20 delta {final_delta:+.5f}/{best_delta:+.5f}/{late20_delta:+.5f}; "
        f"Probe-RGB baseline best/final delta {base_best_delta:+.5f}/{base_final_delta:+.5f}; "
        f"matched positive epochs {sum(1 for v in d_control.ap if v > 0)}/{len(d_control.ap)}."
    )


def draw_diagnostic(size: str, curves: dict[str, Curve], outfile: Path) -> None:
    width, height = 980, 720
    top = Panel(92, 78, 820, 310, 1, 200, 0.0, max(max(c.ap) for c in curves.values()) * 1.08)
    d_control = make_delta_curve("ProbeA - det-only", curves["probe"], curves["control"], COLORS["delta_control"])
    d_rgb = make_delta_curve("ProbeA - RGB baseline", curves["probe"], curves["rgb"], COLORS["delta_rgb"], "5 4")
    d_vals = d_control.ap + d_rgb.ap
    d_pad = max(abs(min(d_vals)), abs(max(d_vals)), 0.01) * 1.12
    bottom = Panel(92, 468, 820, 160, 1, 200, -d_pad, d_pad)

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}</style>',
    ]
    out += draw_axes(top, nice_ticks(top.ymin, top.ymax, 6), f"DroneVehicle sub2k full-val, img{size}", "mAP50-95")
    for key in ["rgb", "teacher", "control", "probe"]:
        out += draw_curve(top, curves[key], "2.4" if key in {"probe", "control"} else "2.0")
    out += draw_legend(100, 42, [curves["probe"], curves["control"], curves["rgb"], curves["teacher"]])

    out += draw_axes(bottom, nice_ticks(bottom.ymin, bottom.ymax, 5), "Method deltas", "Delta", "Epoch")
    out.append(line(bottom.x, bottom.sy(0), bottom.x + bottom.w, bottom.sy(0), stroke=COLORS["zero"], stroke_width="1.2", stroke_dasharray="4 4"))
    out.append(polyline(bottom.plot_points(d_control.epochs, d_control.ap), fill="none", stroke=d_control.color, stroke_width="2.2", stroke_linejoin="round", stroke_linecap="round"))
    out.append(polyline(bottom.plot_points(d_rgb.epochs, d_rgb.ap), fill="none", stroke=d_rgb.color, stroke_width="2.0", stroke_dasharray="5 4", stroke_linejoin="round", stroke_linecap="round"))
    out += draw_legend(100, 438, [d_control, d_rgb])

    summary = summarize(f"img{size}", curves)
    out.append(text(92, 692, summary, font_size="11", fill="#333"))
    out.append("</svg>")
    outfile.write_text("\n".join(out) + "\n")


def draw_overview(all_curves: dict[str, dict[str, Curve]], outfile: Path) -> None:
    width, height = 1180, 430
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}</style>',
    ]
    panels = {
        "512": Panel(78, 78, 470, 270, 1, 200, 0.0, max(max(c.ap) for c in all_curves["512"].values()) * 1.08),
        "256": Panel(665, 78, 470, 270, 1, 200, 0.0, max(max(c.ap) for c in all_curves["256"].values()) * 1.08),
    }
    for size, panel in panels.items():
        out += draw_axes(panel, nice_ticks(panel.ymin, panel.ymax, 6), f"img{size}", "mAP50-95", "Epoch")
        for key in ["rgb", "teacher", "control", "probe"]:
            out += draw_curve(panel, all_curves[size][key], "2.4" if key in {"probe", "control"} else "2.0")
    out += draw_legend(222, 38, [all_curves["512"]["probe"], all_curves["512"]["control"], all_curves["512"]["rgb"], all_curves["512"]["teacher"]])
    out.append(text(width / 2, 24, "ProbeA vs same-pipeline control and fixed baselines on DroneVehicle", text_anchor="middle", font_size="17", font_weight="700", fill="#111"))
    out.append("</svg>")
    outfile.write_text("\n".join(out) + "\n")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_curves = {
        "512": {
            "rgb": read_curve("img512_rgb_baseline.csv", "RGB baseline", COLORS["rgb"]),
            "teacher": read_curve("img512_ir_teacher.csv", "IR teacher", COLORS["teacher"], "6 4"),
            "control": read_curve("img512_detonly_control.csv", "det-only control", COLORS["control"]),
            "probe": read_curve("img512_probeA.csv", "ProbeA", COLORS["probe"]),
        },
        "256": {
            "rgb": read_curve("img256_rgb_baseline.csv", "RGB baseline", COLORS["rgb"]),
            "teacher": read_curve("img256_ir_teacher.csv", "IR teacher", COLORS["teacher"], "6 4"),
            "control": read_curve("img256_detonly_control.csv", "det-only control", COLORS["control"]),
            "probe": read_curve("img256_probeA.csv", "ProbeA", COLORS["probe"]),
        },
    }
    draw_diagnostic("512", all_curves["512"], FIG_DIR / "dronevehicle_probeA_img512_diagnostic.svg")
    draw_diagnostic("256", all_curves["256"], FIG_DIR / "dronevehicle_probeA_img256_diagnostic.svg")
    draw_overview(all_curves, FIG_DIR / "dronevehicle_probeA_img512_img256_overview.svg")
    summary = "\n".join(summarize(f"img{size}", curves) for size, curves in all_curves.items())
    (FIG_DIR / "dronevehicle_probeA_curve_summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
