#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10,<4",
# ]
# ///

# ----- How to run -----
# uv run figures/retina_signal_flow/generate_retina_signal_flow.py
# ----------------------

from __future__ import annotations

# noqa: SIZE_OK - static scientific-figure composition and coordinates

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

if __package__:
    from .drawing import (
        AMACRINE,
        AMACRINE_FILL,
        BORDER,
        CANVAS,
        CONE_FILL,
        CONE_STAGE,
        GUIDE,
        INHIBITORY,
        INHIBITORY_FILL,
        INK,
        INPUT,
        INPUT_FILL,
        L_CONE,
        L_FILL,
        M_CONE,
        M_FILL,
        MUTED,
        OFF,
        OFF_FILL,
        ON,
        ON_FILL,
        OUTPUT,
        OUTPUT_FILL,
        PROCESSING,
        PROCESSING_FILL,
        S_CONE,
        S_FILL,
        BoxStyle,
        Diagram,
        Rect,
        RFSpec,
        StageSpec,
        Stroke,
        TextStyle,
    )
    from .footer import draw_footer
else:
    from drawing import (
        AMACRINE,
        AMACRINE_FILL,
        BORDER,
        CANVAS,
        CONE_FILL,
        CONE_STAGE,
        GUIDE,
        INHIBITORY,
        INHIBITORY_FILL,
        INK,
        INPUT,
        INPUT_FILL,
        L_CONE,
        L_FILL,
        M_CONE,
        M_FILL,
        MUTED,
        OFF,
        OFF_FILL,
        ON,
        ON_FILL,
        OUTPUT,
        OUTPUT_FILL,
        PROCESSING,
        PROCESSING_FILL,
        S_CONE,
        S_FILL,
        BoxStyle,
        Diagram,
        Rect,
        RFSpec,
        StageSpec,
        Stroke,
        TextStyle,
    )
    from footer import draw_footer

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

WIDTH: Final = 4800
HEIGHT: Final = 1980
DPI: Final = 300
DEFAULT_STEM: Final = Path(__file__).resolve().parent / "output/retina_signal_flow"

matplotlib.rcParams.update(
    {
        "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True, slots=True)
class CellSpec:
    rect: Rect
    title: str
    subtitle: str
    accent: str
    fill: str


def draw_headers(diagram: Diagram) -> None:
    diagram.text((WIDTH / 2, 85), "Retina SNN Signal Flow", TextStyle(34, INK, "bold"))
    stages = (
        StageSpec(Rect(60, 190, 620, 95), "1. Input & ISETBio", INPUT, INPUT_FILL),
        StageSpec(Rect(720, 190, 760, 95), "2. Cone sampling", CONE_STAGE, CONE_FILL),
        StageSpec(Rect(1520, 190, 1540, 95), "3. Horizontal & bipolar processing", PROCESSING, PROCESSING_FILL),
        StageSpec(Rect(3100, 190, 650, 95), "4. Amacrine modulation", AMACRINE, AMACRINE_FILL),
        StageSpec(Rect(3790, 190, 950, 95), "5. RGC & optic nerve output", OUTPUT, OUTPUT_FILL),
    )
    for stage in stages:
        diagram.stage(stage)
    for x in (700, 1500, 3080, 3770):
        diagram.line(((x, 315), (x, 1460)), Stroke(GUIDE, 1.0, True))


def draw_input(diagram: Diagram) -> None:
    image_rect = Rect(90, 820, 260, 430)
    diagram.box(image_rect, BoxStyle("#FAFAFA", INPUT, 1.1, 18))
    diagram.text((220, 875), "Natural\nimage", TextStyle(11, INK, "bold"))
    diagram.ax.add_patch(Circle((285, 1010), 27, facecolor="#E3E3E3", edgecolor="none"))
    diagram.ax.add_patch(
        Polygon(
            [(100, 1190), (190, 1060), (260, 1135), (335, 1030), (345, 1190)],
            facecolor="#8B8D91",
            edgecolor="none",
        )
    )
    diagram.ax.add_patch(
        Polygon(
            [(100, 1190), (210, 1100), (290, 1170), (345, 1140), (345, 1190)],
            facecolor="#C6C7C9",
            edgecolor="none",
        )
    )
    isetbio = Rect(430, 900, 220, 270)
    diagram.box(isetbio, BoxStyle(INPUT_FILL, INPUT, 1.1, 18))
    diagram.text((540, 975), "ISETBio", TextStyle(12, INK, "bold"))
    diagram.text((540, 1070), "Optical image\nCone excitation", TextStyle(8.3, INK))
    diagram.arrow((350, 1035), (430, 1035), Stroke(INK, 1.8))
    diagram.arrow((650, 1035), (760, 1035), Stroke(INK, 1.8))


def draw_cones(diagram: Diagram) -> None:
    diagram.text((1100, 405), "L / M / S cones", TextStyle(13, INK, "bold")).set_gid(
        "cone-stage-label"
    )
    rows = (
        (620, "L", L_CONE, L_FILL, ("L cone",)),
        (900, "M", M_CONE, M_FILL, ("M cone",)),
        (1180, "S", S_CONE, S_FILL, ("S cone",)),
    )
    for y, symbol, accent, fill, lines in rows:
        diagram.ax.add_patch(Circle((850, y), 68, facecolor=fill, edgecolor=accent, linewidth=1.8))
        diagram.text((850, y), symbol, TextStyle(13, accent, "bold"))
        for index, value in enumerate(lines):
            diagram.text(
                (1010, y - 66 + index * 33),
                value,
                TextStyle(8.3 if index else 9.5, MUTED if value.startswith("[") else INK, "bold" if index == 0 else "normal", "left"),
            )
def draw_cell_panel(diagram: Diagram, spec: CellSpec) -> None:
    diagram.box(spec.rect, BoxStyle(spec.fill, spec.accent, 1.25, 20))
    diagram.text(
        (spec.rect.x + spec.rect.width / 2, spec.rect.y + spec.rect.height * 0.40),
        spec.title,
        TextStyle(11.5, INK, "bold"),
    )
    diagram.text(
        (spec.rect.x + spec.rect.width / 2, spec.rect.y + spec.rect.height * 0.68),
        spec.subtitle,
        TextStyle(7.8, INK),
    )


def draw_processing(diagram: Diagram) -> None:
    diagram.arrow((1460, 670), (2440, 670), Stroke(ON, 1.9)).set_gid("cone-to-on-bipolar")
    diagram.arrow((1460, 1300), (2440, 1300), Stroke(OFF, 1.9)).set_gid("cone-to-off-bipolar")
    diagram.text((2140, 620), "Sign-inverting · mGluR6", TextStyle(7.6, INK))
    diagram.text((2140, 1350), "Sign-preserving · AMPA/kainate", TextStyle(7.6, INK))

    h1 = Rect(1600, 800, 610, 410)
    diagram.box(h1, BoxStyle(PROCESSING_FILL, PROCESSING, 1.25, 22))
    diagram.text((1905, 875), "H1 horizontal network", TextStyle(12, INHIBITORY, "bold"))
    diagram.text((1905, 955), "Lateral spatial integration\nCone-terminal feedback", TextStyle(8.2, INK))
    diagram.line(((1700, 1085), (2110, 1085)), Stroke(INHIBITORY, 2.4))
    for x in (1725, 1850, 1975, 2100):
        diagram.ax.add_patch(Circle((x, 1085), 28, facecolor="#8B71A7", edgecolor=INHIBITORY, linewidth=1.1))
    diagram.arrow((1460, 730), (1600, 900), Stroke(INHIBITORY, 1.6))
    diagram.arrow((1460, 960), (1600, 1000), Stroke(INHIBITORY, 1.6))
    diagram.arrow((1460, 1180), (1600, 1110), Stroke(MUTED, 1.4, True))
    diagram.text((1520, 1170), "S: weak / absent H1 input", TextStyle(7.1, MUTED, italic=True))
    diagram.curve((1650, 835), (1470, 610), -0.25, Stroke(INHIBITORY, 1.6))
    diagram.text((1600, 720), "feedback", TextStyle(7.1, INHIBITORY, italic=True))

    draw_cell_panel(diagram, CellSpec(Rect(2440, 540, 340, 260), "ON bipolar", "Sign-inverting\nmGluR6", ON, ON_FILL))
    diagram.rf_tile(RFSpec(Rect(2820, 540, 190, 260), "RF\nON-center", ON, ON_FILL))
    draw_cell_panel(diagram, CellSpec(Rect(2440, 1170, 340, 260), "OFF bipolar", "Sign-preserving\nAMPA / kainate", OFF, OFF_FILL))
    diagram.rf_tile(RFSpec(Rect(2820, 1170, 190, 260), "RF\nOFF-center", OFF, OFF_FILL))
def draw_amacrine(diagram: Diagram) -> None:
    diagram.arrow((3010, 670), (3830, 670), Stroke(ON, 2.0)).set_gid("direct-on-bipolar-to-rgc")
    diagram.arrow((3010, 1300), (3830, 1300), Stroke(OFF, 2.0)).set_gid("direct-off-bipolar-to-rgc")
    diagram.text((3410, 620), "Direct excitation", TextStyle(7.4, ON, "bold"))
    diagram.text((3410, 1350), "Direct excitation", TextStyle(7.4, OFF, "bold"))

    panel = Rect(3180, 800, 470, 410)
    diagram.box(panel, BoxStyle(INHIBITORY_FILL, INHIBITORY, 1.25, 22))
    diagram.text((3415, 875), "Local amacrine cell", TextStyle(11.5, INHIBITORY, "bold"))
    diagram.text((3415, 940), "Inhibitory modulation", TextStyle(8, INK))
    diagram.ax.add_patch(Circle((3415, 1040), 30, facecolor="#8B71A7", edgecolor=INHIBITORY, linewidth=1.1))
    diagram.line(((3260, 1040), (3570, 1040)), Stroke(INHIBITORY, 1.7))
    for offset in (-120, -55, 55, 120):
        diagram.line(((3415, 1040), (3415 + offset, 1095)), Stroke(INHIBITORY, 1.3))
    diagram.text((3415, 1160), "Subtype-dependent timing", TextStyle(7.6, INHIBITORY, italic=True))

    diagram.path_arrow(((3010, 750), (3100, 750), (3100, 880), (3180, 880)), Stroke(INHIBITORY, 1.5)).set_gid(
        "on-bipolar-to-amacrine"
    )
    diagram.path_arrow(((3010, 1210), (3100, 1210), (3100, 1110), (3180, 1110)), Stroke(INHIBITORY, 1.5)).set_gid(
        "off-bipolar-to-amacrine"
    )
    for suffix, artist in zip(
        ("line", "bar"),
        diagram.t_bar((3650, 900), (3810, 790), Stroke(INHIBITORY, 1.7)),
        strict=True,
    ):
        artist.set_gid(f"amacrine-to-on-rgc-inhibition-{suffix}")
    for suffix, artist in zip(
        ("line", "bar"),
        diagram.t_bar((3650, 1090), (3810, 1180), Stroke(INHIBITORY, 1.7)),
        strict=True,
    ):
        artist.set_gid(f"amacrine-to-off-rgc-inhibition-{suffix}")
def draw_output(diagram: Diagram) -> None:
    draw_cell_panel(diagram, CellSpec(Rect(3830, 540, 300, 260), "ON RGC", "Spike output", ON, ON_FILL))
    diagram.rf_tile(RFSpec(Rect(4160, 540, 190, 260), "RF\nON-center", ON, ON_FILL))
    draw_cell_panel(diagram, CellSpec(Rect(3830, 1170, 300, 260), "OFF RGC", "Spike output", OFF, OFF_FILL))
    diagram.rf_tile(RFSpec(Rect(4160, 1170, 190, 260), "RF\nOFF-center", OFF, OFF_FILL))

    optic = Rect(4430, 820, 300, 390)
    diagram.box(optic, BoxStyle("#FAFAFA", INPUT, 1.1, 20))
    diagram.text((4580, 875), "Optic nerve", TextStyle(11.5, INK, "bold"))
    diagram.text((4580, 930), "Spike trains", TextStyle(7.6, MUTED))
    for x in range(4480, 4690, 32):
        diagram.line(((x, 1010), (x, 1140)), Stroke(INK, 0.7))
        for y in (1025 + (x % 3) * 7, 1070 + (x % 5) * 5, 1120 - (x % 4) * 6):
            diagram.line(((x - 7, y), (x + 7, y)), Stroke(INK, 0.8))
    diagram.curve((4350, 670), (4430, 900), 0.22, Stroke(INK, 1.6))
    diagram.curve((4350, 1300), (4430, 1130), -0.22, Stroke(INK, 1.6))
def build_figure() -> tuple[Figure, Axes]:
    figure, ax = plt.subplots(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    figure.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.set_facecolor(CANVAS)
    ax.axis("off")
    diagram = Diagram(ax)
    draw_headers(diagram)
    draw_input(diagram)
    draw_cones(diagram)
    draw_processing(diagram)
    draw_amacrine(diagram)
    draw_output(diagram)
    draw_footer(diagram)
    return figure, ax


def render(output_stem: Path = DEFAULT_STEM) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    figure, _ = build_figure()
    figure.savefig(png_path, dpi=DPI, facecolor=CANVAS, metadata={"Software": "matplotlib"})
    figure.savefig(svg_path, facecolor=CANVAS, metadata={"Creator": "matplotlib"})
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor=CANVAS, metadata={"Creator": "matplotlib"})
    plt.close(figure)
    return png_path, svg_path


def main() -> None:
    output_stem = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_STEM
    if len(sys.argv) > 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [OUTPUT_STEM]")
    for path in (*render(output_stem), output_stem.with_suffix(".pdf")):
        print(path)


if __name__ == "__main__":
    main()
