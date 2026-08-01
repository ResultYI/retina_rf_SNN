#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10,<4",
# ]
# ///

# ----- How to run -----
# uv run figures/retina_signal_flow/generate_training_evaluation_flow.py
# ----------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if __package__:
    from .drawing import (
        BORDER,
        CANVAS,
        INK,
        INPUT,
        INPUT_FILL,
        MUTED,
        OUTPUT,
        OUTPUT_FILL,
        PROCESSING,
        PROCESSING_FILL,
        SURFACE,
        BoxStyle,
        Diagram,
        Rect,
        Stroke,
        TextStyle,
    )
else:
    from drawing import (
        BORDER,
        CANVAS,
        INK,
        INPUT,
        INPUT_FILL,
        MUTED,
        OUTPUT,
        OUTPUT_FILL,
        PROCESSING,
        PROCESSING_FILL,
        SURFACE,
        BoxStyle,
        Diagram,
        Rect,
        Stroke,
        TextStyle,
    )

WIDTH: Final = 5000
HEIGHT: Final = 1650
DPI: Final = 300
DEFAULT_STEM: Final = (
    Path(__file__).resolve().parent
    / "output"
    / "retina_snn_training_evaluation_flow"
)


@dataclass(frozen=True, slots=True)
class Node:
    rect: Rect
    label: str
    accent: str
    fill: str
    text_size: float = 9.2


NODES: Final = (
    Node(Rect(80, 590, 430, 300), "Experimental\nstimulus", INPUT, INPUT_FILL),
    Node(
        Rect(620, 590, 540, 300),
        "Species-matched\ninput model",
        INPUT,
        INPUT_FILL,
    ),
    Node(Rect(1270, 590, 450, 300), "Retina SNN", PROCESSING, PROCESSING_FILL),
    Node(
        Rect(1830, 590, 600, 300),
        "Recorded-cell\nRGC output\nspike probability /\nfiring rate",
        OUTPUT,
        OUTPUT_FILL,
        8.5,
    ),
    Node(
        Rect(2540, 590, 470, 300),
        "Likelihood loss\nvs recorded spikes",
        PROCESSING,
        SURFACE,
    ),
    Node(Rect(3120, 590, 400, 300), "Freeze\nmodel", PROCESSING, PROCESSING_FILL),
    Node(
        Rect(3630, 590, 520, 300),
        "STA / Jacobian /\nGLM readout",
        OUTPUT,
        SURFACE,
    ),
    Node(
        Rect(4260, 590, 620, 300),
        "Dynamic effective RF",
        OUTPUT,
        OUTPUT_FILL,
    ),
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_node(diagram: Diagram, node: Node) -> None:
    diagram.box(node.rect, BoxStyle(node.fill, node.accent, 2.2, 24))
    diagram.text(
        (
            node.rect.x + node.rect.width / 2,
            node.rect.y + node.rect.height / 2,
        ),
        node.label,
        TextStyle(node.text_size, INK, "bold"),
    )


def annotation(
    diagram: Diagram,
    rect: Rect,
    label: str,
    element_id: str,
    *,
    accent: str = BORDER,
) -> None:
    patch = diagram.box(rect, BoxStyle(SURFACE, accent, 1.5, 18))
    patch.set_gid(element_id)
    text = diagram.text(
        (rect.x + rect.width / 2, rect.y + rect.height / 2),
        label,
        TextStyle(8.2, INK, "bold"),
    )
    text.set_gid(f"{element_id}-label")


def build_figure() -> tuple[Figure, Axes]:
    configure_matplotlib()
    figure, ax = plt.subplots(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    figure.patch.set_facecolor(CANVAS)
    figure.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor(CANVAS)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    diagram = Diagram(ax)

    diagram.text(
        (WIDTH / 2, 100),
        "Retina SNN Training & RF Evaluation",
        TextStyle(18, INK, "bold"),
    )
    diagram.line(((80, 485), (3010, 485)), Stroke(PROCESSING, 2.4))
    diagram.text((1545, 450), "TRAINING", TextStyle(8, PROCESSING, "bold"))
    diagram.line(((3120, 485), (4880, 485)), Stroke(OUTPUT, 2.4))
    diagram.text(
        (4000, 450),
        "FROZEN-MODEL EVALUATION",
        TextStyle(8, OUTPUT, "bold"),
    )

    for node in NODES:
        draw_node(diagram, node)
    for left, right in zip(NODES, NODES[1:]):
        diagram.arrow(
            (left.rect.x + left.rect.width + 18, 740),
            (right.rect.x - 18, 740),
            Stroke(INK, 2.4),
        )

    annotation(
        diagram,
        Rect(600, 215, 1680, 145),
        "Fixed metadata: cell type · ON/OFF polarity · eccentricity",
        "fixed-metadata",
        accent=INPUT,
    )
    diagram.arrow((890, 360), (890, 555), Stroke(INPUT, 1.6))

    mapping = diagram.text(
        (2130, 530),
        "One model output unit\nper recorded RGC",
        TextStyle(8.2, OUTPUT, "bold"),
    )
    mapping.set_gid("one-to-one-mapping")

    annotation(
        diagram,
        Rect(2540, 215, 470, 145),
        "Recorded spikes\n= supervision",
        "recorded-spike-supervision",
        accent=PROCESSING,
    )
    diagram.arrow((2775, 360), (2775, 555), Stroke(PROCESSING, 1.8))

    no_rf = diagram.text(
        (2775, 1040),
        "RF is not in the training loss",
        TextStyle(8.2, MUTED, "bold"),
    )
    no_rf.set_gid("rf-post-hoc-no-loss")
    diagram.line(((2540, 985), (3010, 985)), Stroke(BORDER, 1.4))

    annotation(
        diagram,
        Rect(3625, 1010, 1260, 145),
        "RF extracted after freezing",
        "rf-post-hoc",
        accent=OUTPUT,
    )
    diagram.arrow((3890, 975), (3890, 910), Stroke(OUTPUT, 1.6))
    diagram.arrow((4570, 975), (4570, 910), Stroke(OUTPUT, 1.6))
    return figure, ax


def render(output_stem: Path) -> tuple[Path, Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = build_figure()
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=DPI, facecolor=CANVAS)
    figure.savefig(svg_path, facecolor=CANVAS)
    figure.savefig(pdf_path, facecolor=CANVAS)
    plt.close(figure)
    return png_path, svg_path, pdf_path


def main() -> None:
    render(DEFAULT_STEM)


if __name__ == "__main__":
    main()
