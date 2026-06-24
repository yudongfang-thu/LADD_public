#!/usr/bin/env python3
"""Generate PNG diagnostics for OGSOD YOLO-init curves using Pillow."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "ogsod_yoloinit_dynamic_vs_baselines_20260624.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(22, True)
F_LABEL = font(17)
F_TICK = font(14)
F_LEG = font(14)


def metric(row: dict[str, str], names: tuple[str, ...]) -> float:
    stripped = {k.strip(): v for k, v in row.items()}
    for name in names:
        if row.get(name):
            return float(row[name])
        if stripped.get(name):
            return float(stripped[name])
    raise KeyError(names)


def read_curve(path: Path) -> tuple[list[int], list[float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        list(range(1, len(rows) + 1)),
        [metric(row, ("metrics/mAP50-95(B)", "metrics/mAP50-95")) for row in rows],
    )


def delta(candidate: list[float], control: list[float]) -> tuple[list[int], list[float]]:
    n = min(len(candidate), len(control))
    return list(range(1, n + 1)), [candidate[i] - control[i] for i in range(n)]


def map_xy(panel: dict[str, float], x: float, y: float) -> tuple[int, int]:
    px = panel["x"] + (x - panel["xmin"]) / (panel["xmax"] - panel["xmin"]) * panel["w"]
    py = panel["y"] + panel["h"] - (y - panel["ymin"]) / (panel["ymax"] - panel["ymin"]) * panel["h"]
    return int(round(px)), int(round(py))


def draw_dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, width: int, dash: int = 10, gap: int = 7) -> None:
    if len(points) < 2:
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        x0, y0 = p0
        x1, y1 = p1
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if length == 0:
            continue
        dx, dy = (x1 - x0) / length, (y1 - y0) / length
        cur = 0.0
        while cur < length:
            end = min(cur + dash, length)
            draw.line(
                [(x0 + dx * cur, y0 + dy * cur), (x0 + dx * end, y0 + dy * end)],
                fill=color,
                width=width,
            )
            cur += dash + gap


def draw_curve(draw: ImageDraw.ImageDraw, panel: dict[str, float], xs: list[int], ys: list[float], color: str, width: int = 3, dashed: bool = False) -> None:
    pts = [map_xy(panel, x, y) for x, y in zip(xs, ys) if panel["xmin"] <= x <= panel["xmax"]]
    if len(pts) < 2:
        return
    if dashed:
        draw_dashed_line(draw, pts, color, width)
    else:
        draw.line(pts, fill=color, width=width, joint="curve")


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, fnt: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    x, y = xy
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, fill=fill, font=fnt)


def draw_panel(draw: ImageDraw.ImageDraw, panel: dict[str, float], title: str, ylabel: str, yticks: list[float], xticks: list[int]) -> None:
    x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
    draw.rectangle([x, y, x + w, y + h], outline="#222", width=2)
    for t in yticks:
        _, py = map_xy(panel, panel["xmin"], t)
        draw.line([(x, py), (x + w, py)], fill="#dddddd", width=1)
        lab = f"{t:.3f}" if abs(t) < 0.1 else f"{t:.2f}"
        draw.text((x - 12 - len(lab) * 7, py - 8), lab, fill="#222", font=F_TICK)
    for t in xticks:
        px, _ = map_xy(panel, t, panel["ymin"])
        draw.line([(px, y), (px, y + h)], fill="#eeeeee", width=1)
        text_center(draw, (px, y + h + 20), str(t), "#222", F_TICK)
    text_center(draw, (x + w // 2, y - 28), title, "#222", F_TITLE)
    text_center(draw, (x + w // 2, y + h + 48), "Epoch", "#222", F_LABEL)
    text_center(draw, (x - 62, y + h // 2), ylabel, "#222", F_LABEL)


def legend(draw: ImageDraw.ImageDraw, x: int, y: int, items: list[tuple[str, str, bool, int]]) -> None:
    for i, (label, color, dashed, width) in enumerate(items):
        yy = y + i * 22
        if dashed:
            draw_dashed_line(draw, [(x, yy), (x + 35, yy)], color, width)
        else:
            draw.line([(x, yy), (x + 35, yy)], fill=color, width=width)
        draw.text((x + 45, yy - 8), label, fill="#222", font=F_LEG)


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

    scale = 2
    img = Image.new("RGB", (1900 * scale, 610 * scale), "white")
    draw = ImageDraw.Draw(img)
    panels = [
        {"x": 90 * scale, "y": 80 * scale, "w": 520 * scale, "h": 360 * scale, "xmin": 0, "xmax": 800, "ymin": 0.0, "ymax": 0.66},
        {"x": 735 * scale, "y": 80 * scale, "w": 500 * scale, "h": 360 * scale, "xmin": 0, "xmax": 400, "ymin": -0.006, "ymax": 0.024},
        {"x": 1370 * scale, "y": 80 * scale, "w": 430 * scale, "h": 360 * scale, "xmin": 0, "xmax": 200, "ymin": -0.012, "ymax": 0.014},
    ]

    # Scale font objects by drawing at 2x via PIL fonts defined at target size*2.
    global F_TITLE, F_LABEL, F_TICK, F_LEG
    F_TITLE = font(22 * scale, True)
    F_LABEL = font(17 * scale)
    F_TICK = font(14 * scale)
    F_LEG = font(14 * scale)

    p = panels[0]
    draw_panel(draw, p, "Absolute AP50-95 curves", "AP50-95", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 100, 200, 400, 600, 800])
    for name in curves:
        dashed = "baseline" in name or "oldcommit" in name or "stopped" in name
        width = 6 if name == "4090 Dynamic e800" else 4
        if name in ("SAR baseline", "RGB baseline", "4090 det-only"):
            width = 5
        draw_curve(draw, p, curves[name][0], curves[name][1], colors[name], width, dashed)
    legend(draw, 335 * scale, 215 * scale, [
        ("SAR formal baseline", colors["SAR baseline"], True, 5),
        ("RGB formal baseline", colors["RGB baseline"], True, 5),
        ("4090 det-only", colors["4090 det-only"], False, 5),
        ("4090 Dynamic e800", colors["4090 Dynamic e800"], False, 6),
        ("4090 ProbeA", colors["4090 ProbeA"], False, 4),
        ("4090 oldcommit ProbeA", colors["4090 oldcommit ProbeA"], True, 4),
        ("3090 candidates", colors["3090 singleproj"], False, 4),
    ])

    p = panels[1]
    draw_panel(draw, p, "4090 delta vs same-pipeline det-only", "Delta AP50-95", [-0.005, 0, 0.005, 0.010, 0.015, 0.020], [0, 100, 200, 300, 400])
    draw_curve(draw, p, [0, 400], [0.010, 0.010], "#E45756", 2, True)
    ctrl = curves["4090 det-only"][1]
    for name in ["4090 ProbeA", "4090 Dynamic e800", "4090 oldcommit ProbeA"]:
        xs, ds = delta(curves[name][1], ctrl)
        draw_curve(draw, p, xs, ds, colors[name], 6 if name == "4090 Dynamic e800" else 4, "oldcommit" in name)
    legend(draw, 990 * scale, 118 * scale, [
        ("ProbeA", colors["4090 ProbeA"], False, 4),
        ("Dynamic e800", colors["4090 Dynamic e800"], False, 6),
        ("oldcommit ProbeA", colors["4090 oldcommit ProbeA"], True, 4),
        ("+0.010 screen", colors["4090 Dynamic e800"], True, 2),
    ])

    p = panels[2]
    draw_panel(draw, p, "3090 delta vs same-pipeline det-only", "Delta AP50-95", [-0.010, -0.005, 0, 0.005, 0.010], [0, 50, 100, 150, 200])
    draw_curve(draw, p, [0, 200], [0.010, 0.010], "#E45756", 2, True)
    ctrl = curves["3090 det-only ctrl"][1]
    for name in ["3090 singleproj", "3090 wo_s_rec", "3090 wo_reach stopped", "3090 dynamic_plain"]:
        xs, ds = delta(curves[name][1], ctrl)
        draw_curve(draw, p, xs, ds, colors[name], 5 if name == "3090 wo_s_rec" else 4, "stopped" in name)
    legend(draw, 1545 * scale, 300 * scale, [
        ("singleproj", colors["3090 singleproj"], False, 4),
        ("wo_s_rec", colors["3090 wo_s_rec"], False, 5),
        ("wo_reach stopped", colors["3090 wo_reach stopped"], True, 4),
        ("dynamic_plain", colors["3090 dynamic_plain"], False, 4),
    ])

    img = img.resize((1900, 610), Image.Resampling.LANCZOS)
    img.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
