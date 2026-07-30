#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10,<4",
# ]
# ///

# ----- How to run -----
# Default outputs:
#   uv run scripts/draw_retina_snn_signal_flow.py
# Custom output stem:
#   uv run scripts/draw_retina_snn_signal_flow.py path/to/output
# ----------------------

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Final, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

type Align = Literal["left", "center", "right"]

WIDTH: Final = 1672
HEIGHT: Final = 941
ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_STEM: Final = ROOT / "report/assets/retina_snn_signal_flow_redrawn"


def text(
    ax: Axes,
    x: float,
    y: float,
    value: str,
    size: float = 14,
    *,
    color: str = "#111111",
    weight: str = "normal",
    align: Align = "center",
    style: str = "normal",
) -> None:
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontstyle=style,
        ha=align,
        va="center",
        family="Arial",
        linespacing=1.2,
    )


def rounded(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    edge: str,
    face: str = "#ffffff",
    radius: float = 18,
    line_width: float = 1.2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.01,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=line_width,
        )
    )


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#222222",
    size: float = 18,
    width: float = 2.4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=size,
            linewidth=width,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def stage(ax: Axes, number: int, x: float, title: str) -> None:
    ax.add_patch(Circle((x, 146), 17, facecolor="#163b63", edgecolor="none"))
    text(ax, x, 146, str(number), 17, color="white", weight="bold")
    text(ax, x, 187, title, 16, weight="bold")


def divider(ax: Axes, x1: float, x2: float, y: float, color: str) -> None:
    ax.plot([x1, x2], [y, y], color=color, linewidth=1, dashes=(1.5, 2))


def receptive_field(ax: Axes, x: float, y: float, color: str, radius: float, dashed: bool = False) -> None:
    for scale, alpha in ((1.0, 0.06), (0.68, 0.12), (0.38, 0.22), (0.14, 0.75)):
        circle = Circle(
            (x, y),
            radius * scale,
            facecolor=color if scale < 1 else "none",
            edgecolor=color,
            linewidth=1.1,
            alpha=alpha if scale < 1 else 0.9,
        )
        if dashed and scale == 1:
            circle.set_linestyle((0, (3, 2)))
        ax.add_patch(circle)


def draw_scene(ax: Axes) -> None:
    rounded(ax, 20, 238, 170, 406, "#333333", "#ffffff", 17)
    rounded(ax, 31, 258, 148, 174, "#b5b5b5", "#eeeeee", 10)
    ax.add_patch(Polygon([(32, 357), (73, 302), (109, 346), (141, 287), (179, 342), (179, 431), (32, 431)], facecolor="#a1a1a1", edgecolor="none"))
    ax.add_patch(Polygon([(32, 377), (73, 326), (111, 364), (141, 315), (179, 356), (179, 431), (32, 431)], facecolor="#686868", edgecolor="none"))
    ax.add_patch(Polygon([(32, 385), (179, 356), (179, 431), (32, 431)], facecolor="#d8d8d8", edgecolor="none"))
    ax.plot([32, 179], [392, 392], color="#f4f4f4", linewidth=1)
    for px, height in ((42, 68), (50, 48), (162, 39), (170, 55)):
        ax.plot([px, px], [425, 425 - height], color="#262626", linewidth=2)
        for offset in range(8, height, 9):
            ax.plot([px - offset * 0.18, px + offset * 0.18], [425 - offset, 425 - offset], color="#262626", linewidth=2)
    text(ax, 105, 467, "Natural scene\n(grayscale)", 13)
    divider(ax, 31, 179, 506, "#777777")
    points = [(42, 569), (51, 531), (65, 542), (81, 547), (99, 530), (117, 556), (138, 543), (158, 550)]
    ax.plot([p[0] for p in points], [p[1] for p in points], color="#111111", linewidth=1.4)
    ax.scatter([p[0] for p in points[:-1]], [p[1] for p in points[:-1]], s=18, color="#111111")
    arrow(ax, points[-2], (171, 547), "#111111", 13, 1.4)
    text(ax, 105, 602, "Eye movement\n(trace)", 13)


def draw_cones(ax: Axes) -> None:
    rounded(ax, 394, 220, 200, 441, "#2b7c23", "#f4faef", 24)
    labels = ("L", "M", "L", "S", "M", "L", "M", "L")
    colors = {"L": ("#f6ded1", "#d64a1d"), "M": ("#e8f0d5", "#387122"), "S": ("#dcdaf2", "#344584")}
    for row in range(7):
        for col in range(6):
            px = 418 + col * 30 + 5 * math.sin(row * 1.7 + col)
            py = 259 + row * 31 + 4 * math.cos(col * 1.4 + row)
            cone = labels[(row * 6 + col) % len(labels)]
            face, edge = colors[cone]
            ax.add_patch(Circle((px, py), 11, facecolor=face, edgecolor=edge, linewidth=1))
            text(ax, px, py, cone, 10, color=edge)
    text(ax, 494, 494, "L", 15, color="#d64a1d", weight="bold")
    text(ax, 514, 494, "/", 15)
    text(ax, 530, 494, "M", 15, color="#387122", weight="bold")
    text(ax, 552, 494, "/", 15)
    text(ax, 568, 494, "S", 15, color="#344584", weight="bold")
    text(ax, 494, 525, "Photoreceptors", 14, color="#b82d16")
    text(ax, 494, 557, "Irregular spatial sampling", 12, style="italic")
    divider(ax, 397, 591, 590, "#387122")
    text(ax, 494, 623, "Cone spacing / aperture:\neccentricity-dependent", 11)


def draw_bipolar(ax: Axes, y: float, color: str, face: str, title_value: str) -> None:
    rounded(ax, 854, y, 240, 199, color, face, 22)
    text(ax, 974, y + 30, title_value, 16, color=color, weight="bold")
    text(ax, 870, y + 64, "Local center-surround", 12, align="left")
    text(ax, 870, y + 96, "Relay to inner retina", 12, align="left")
    receptive_field(ax, 1053, y + 86, color, 32)
    divider(ax, 865, 1083, y + 136, color)
    text(ax, 974, y + 159, "RF center: tens of μm", 11)
    text(ax, 974, y + 183, "subtype-dependent", 11, style="italic")


def draw_rgc(ax: Axes, y: float, color: str, face: str, title_value: str) -> None:
    rounded(ax, 1339, y, 182, 247, color, face, 20)
    text(ax, 1430, y + 29, title_value, 16, color=color, weight="bold")
    text(ax, 1360, y + 60, "Retinal ganglion cell", 12, align="left")
    text(ax, 1360, y + 91, "Output to brain", 12, align="left")
    receptive_field(ax, 1480, y + 128, color, 30)
    divider(ax, 1349, 1511, y + 166, color)
    text(ax, 1430, y + 188, "RF center: ~30–300 μm", 11)
    text(ax, 1430, y + 214, "depends on cell type,", 10, style="italic")
    text(ax, 1430, y + 234, "species, eccentricity", 10, style="italic")


def draw_legend(ax: Axes) -> None:
    rounded(ax, 1240, 734, 379, 187, "#333333", "#ffffff", 20)
    text(ax, 1262, 756, "Legend", 12, weight="bold", align="left")
    arrow(ax, (1265, 781), (1314, 781), "#111111", 14, 2)
    text(ax, 1337, 781, "Solid arrow: excitatory / forward flow", 10, align="left")
    ax.plot([1265, 1312], [809, 809], color="#111111", linewidth=2.3)
    ax.plot([1312, 1312], [799, 819], color="#111111", linewidth=2.3)
    text(ax, 1337, 809, "Bar-ended line: inhibition", 10, align="left")
    curved = FancyArrowPatch((1270, 851), (1315, 851), connectionstyle="arc3,rad=-0.55", arrowstyle="-|>", mutation_scale=13, linewidth=2, color="#111111")
    ax.add_patch(curved)
    text(ax, 1337, 838, "Curved arrow: feedback", 10, align="left")
    receptive_field(ax, 1290, 886, "#777777", 24, True)
    text(ax, 1337, 886, "Concentric circles: receptive-field scale\n(center to surround)", 10, align="left")


def build_figure() -> tuple[Figure, Axes]:
    figure, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    figure.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    text(ax, WIDTH / 2, 40, "Retina SNN Signal Flow", 38, weight="bold")
    for number, x, title_value in ((1, 105, "Visual\nstimulus"), (2, 289, "ISETBio"), (3, 484, "Cone mosaic"), (4, 706, "H1 horizontal cell"), (5, 972, "Bipolar cells"), (6, 1205, "Amacrine cell"), (7, 1422, "RGC"), (8, 1598, "")):
        stage(ax, number, x, title_value)
    draw_scene(ax)
    rounded(ax, 222, 236, 139, 407, "#c58a00", "#fffaf0", 19)
    text(ax, 232, 304, "•  Optics", 13, align="left")
    text(ax, 232, 394, "•  Cone mosaic\n   sampling", 13, align="left")
    text(ax, 232, 495, "•  Time-varying\n   cone responses", 13, align="left")
    draw_cones(ax)
    rounded(ax, 635, 220, 170, 442, "#775197", "#f9f4fb", 23)
    receptive_field(ax, 720, 300, "#6d3d91", 40, True)
    text(ax, 720, 382, "Lateral surround", 12)
    text(ax, 720, 423, "Feedback modulation", 12)
    text(ax, 720, 477, "Wide spatial\nintegration", 13)
    divider(ax, 645, 795, 527, "#775197")
    text(ax, 720, 574, "Approx. RF diameter:\n~100–300 μm", 12)
    text(ax, 720, 624, "varies by species &\neccentricity", 11, style="italic")
    draw_bipolar(ax, 219, "#e65500", "#fff7f1", "ON bipolar cell")
    draw_bipolar(ax, 468, "#2166ac", "#f2f7fd", "OFF bipolar cell")
    rounded(ax, 1136, 220, 157, 455, "#087c78", "#f2faf8", 22)
    receptive_field(ax, 1215, 289, "#087c78", 39, True)
    text(ax, 1215, 371, "Inhibitory modulation", 11)
    text(ax, 1215, 404, "Temporal shaping", 11)
    text(ax, 1215, 438, "Local circuit", 11)
    ax.add_patch(Circle((1183, 491), 18, facecolor="white", edgecolor="#333333", linewidth=1.3))
    ax.plot([1183, 1183], [491, 480], color="#333333", linewidth=1.2)
    ax.plot([1183, 1192], [491, 497], color="#333333", linewidth=1.2)
    text(ax, 1218, 491, "delay", 11, align="left")
    divider(ax, 1147, 1282, 537, "#087c78")
    text(ax, 1215, 571, "Local field:\n~50–200 μm (approx.)", 11)
    text(ax, 1215, 630, "high subtype\ndiversity", 11, style="italic")
    draw_rgc(ax, 213, "#e65500", "#fff7f1", "ON RGC")
    draw_rgc(ax, 474, "#2166ac", "#f2f7fd", "OFF RGC")
    rounded(ax, 1556, 376, 107, 152, "#333333", "#ffffff", 16)
    text(ax, 1609, 451, "Retinal\noutput /\nspike\nresponses", 14, weight="bold")
    for start, end in (((190, 407), (221, 407)), ((361, 407), (393, 407)), ((594, 407), (634, 407))):
        arrow(ax, start, end, "#333333", 18, 2.5)
    arrow(ax, (805, 293), (853, 293), "#d84a00", 18, 2.5)
    arrow(ax, (805, 563), (853, 563), "#2166ac", 18, 2.5)
    arrow(ax, (1094, 299), (1135, 299), "#d84a00", 18, 2.5)
    arrow(ax, (1094, 563), (1135, 563), "#2166ac", 18, 2.5)
    for y, color in ((308, "#d84a00"), (578, "#2166ac")):
        ax.plot([1293, 1332], [y, y], color=color, linewidth=2.5)
        ax.plot([1332, 1332], [y - 12, y + 12], color=color, linewidth=2.5)
    arrow(ax, (1521, 314), (1568, 314), "#111111", 18, 2.5)
    arrow(ax, (1521, 582), (1568, 582), "#111111", 18, 2.5)
    feedback_left = FancyArrowPatch((550, 708), (470, 658), connectionstyle="arc3,rad=-0.22", arrowstyle="-|>", mutation_scale=16, linewidth=2, color="#333333")
    feedback_right = FancyArrowPatch((550, 708), (632, 658), connectionstyle="arc3,rad=0.22", arrowstyle="-|>", mutation_scale=16, linewidth=2, color="#333333")
    ax.add_patch(feedback_left)
    ax.add_patch(feedback_right)
    text(ax, 551, 744, "Lateral feedback /\nsurround modulation", 10, style="italic")
    rounded(ax, 36, 875, 901, 47, "#333333", "#ffffff", 8)
    text(ax, 53, 899, "*  RF sizes are approximate physiological ranges and vary with species, eccentricity, subtype, and measurement method.", 10, align="left")
    draw_legend(ax)
    return figure, ax


def render(output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    figure, _ = build_figure()
    figure.savefig(png_path, dpi=100, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)
    return png_path, svg_path


def main() -> None:
    output_stem = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_STEM
    if len(sys.argv) > 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [OUTPUT_STEM]")
    png_path, svg_path = render(output_stem)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
