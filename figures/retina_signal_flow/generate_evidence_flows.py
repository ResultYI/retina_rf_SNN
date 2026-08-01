#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10,<4",
# ]
# ///

# ----- How to run -----
# uv run figures/retina_signal_flow/generate_evidence_flows.py
# ----------------------

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

if __package__:
    from .drawing import CANVAS, INK
    from .generate_retina_signal_flow import DPI, HEIGHT, WIDTH, build_figure
else:
    from drawing import CANVAS, INK
    from generate_retina_signal_flow import DPI, HEIGHT, WIDTH, build_figure

SUPPORTED_COLOR: Final = "#18864B"
GAP_COLOR: Final = "#C92A2A"
DEFAULT_OUTPUT_DIR: Final = Path(__file__).resolve().parent / "output"


class EvidenceSpecies(StrEnum):
    HUMAN = "human"
    MACAQUE = "macaque"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    GAP = "gap"


@dataclass(frozen=True, slots=True)
class EvidenceMarker:
    marker_id: str
    x: float
    y: float
    status: EvidenceStatus


@dataclass(frozen=True, slots=True)
class EvidenceOutput:
    species: EvidenceSpecies
    png_path: Path
    svg_path: Path
    pdf_path: Path


def marker(marker_id: str, x: float, y: float, status: EvidenceStatus) -> EvidenceMarker:
    return EvidenceMarker(marker_id, x, y, status)


def markers_for(species: EvidenceSpecies) -> tuple[EvidenceMarker, ...]:
    match species:
        case EvidenceSpecies.HUMAN:
            return (
                marker("cone-data", 1400, 405, EvidenceStatus.SUPPORTED),
                marker("h1-connection", 1550, 925, EvidenceStatus.GAP),
                marker("h1-internal", 2180, 825, EvidenceStatus.GAP),
                marker("cone-on-bipolar", 2310, 670, EvidenceStatus.SUPPORTED),
                marker("cone-off-bipolar", 2310, 1300, EvidenceStatus.SUPPORTED),
                marker("on-bipolar-rf", 2990, 550, EvidenceStatus.GAP),
                marker("off-bipolar-rf", 2990, 1180, EvidenceStatus.GAP),
                marker("on-direct-rgc", 3720, 670, EvidenceStatus.SUPPORTED),
                marker("off-direct-rgc", 3720, 1300, EvidenceStatus.SUPPORTED),
                marker("amacrine-input", 3135, 1000, EvidenceStatus.GAP),
                marker("amacrine-internal", 3615, 1190, EvidenceStatus.GAP),
                marker("amacrine-output", 3740, 950, EvidenceStatus.GAP),
                marker("on-rgc-anatomy", 4100, 550, EvidenceStatus.SUPPORTED),
                marker("off-rgc-anatomy", 4100, 1180, EvidenceStatus.SUPPORTED),
                marker("on-rgc-rf", 4330, 550, EvidenceStatus.GAP),
                marker("off-rgc-rf", 4330, 1180, EvidenceStatus.GAP),
                marker("optic-output", 4700, 825, EvidenceStatus.SUPPORTED),
            )
        case EvidenceSpecies.MACAQUE:
            return (
                marker("cone-data", 1400, 405, EvidenceStatus.SUPPORTED),
                marker("h1-input", 1550, 925, EvidenceStatus.SUPPORTED),
                marker("h1-feedback", 1550, 650, EvidenceStatus.SUPPORTED),
                marker("h1-internal", 2180, 825, EvidenceStatus.SUPPORTED),
                marker("cone-on-bipolar", 2310, 670, EvidenceStatus.SUPPORTED),
                marker("cone-off-bipolar", 2310, 1300, EvidenceStatus.SUPPORTED),
                marker("on-bipolar-rf", 2990, 550, EvidenceStatus.SUPPORTED),
                marker("off-bipolar-rf", 2990, 1180, EvidenceStatus.SUPPORTED),
                marker("on-direct-rgc", 3720, 670, EvidenceStatus.SUPPORTED),
                marker("off-direct-rgc", 3720, 1300, EvidenceStatus.SUPPORTED),
                marker("on-amacrine-input", 3135, 835, EvidenceStatus.SUPPORTED),
                marker("off-amacrine-input", 3135, 1165, EvidenceStatus.SUPPORTED),
                marker("amacrine-internal", 3615, 1190, EvidenceStatus.GAP),
                marker("amacrine-output", 3740, 950, EvidenceStatus.SUPPORTED),
                marker("on-rgc-rf", 4330, 550, EvidenceStatus.SUPPORTED),
                marker("off-rgc-rf", 4330, 1180, EvidenceStatus.SUPPORTED),
                marker("optic-output", 4700, 825, EvidenceStatus.SUPPORTED),
            )
        case unreachable:
            assert_never(unreachable)


def draw_marker(ax: Axes, species: EvidenceSpecies, evidence: EvidenceMarker) -> None:
    gid = f"{species.value}-{evidence.status.value}-{evidence.marker_id}"
    match evidence.status:
        case EvidenceStatus.SUPPORTED:
            artist = Circle(
                (evidence.x, evidence.y),
                24,
                facecolor="#FFFFFF",
                edgecolor=SUPPORTED_COLOR,
                linewidth=3.2,
                zorder=30,
            )
            artist.set_gid(gid)
            ax.add_patch(artist)
        case EvidenceStatus.GAP:
            artist = ax.text(
                evidence.x,
                evidence.y,
                "×",
                color=GAP_COLOR,
                fontsize=17,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "pad": 0.08, "alpha": 0.94},
                zorder=30,
            )
            artist.set_gid(gid)
        case unreachable:
            assert_never(unreachable)


def draw_evidence_header(ax: Axes, species: EvidenceSpecies) -> None:
    title = f"{species.value.title()} evidence map"
    ax.text(
        WIDTH / 2,
        145,
        title,
        color=INK,
        fontsize=12,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#FFFFFF", "edgecolor": "#B8B8B8"},
        zorder=30,
    )
    supported = Circle(
        (3890, 145),
        17,
        facecolor="#FFFFFF",
        edgecolor=SUPPORTED_COLOR,
        linewidth=2.6,
        zorder=30,
    )
    supported.set_gid(f"{species.value}-legend-supported")
    ax.add_patch(supported)
    ax.text(3930, 145, "Evidence available", color=INK, fontsize=7.2, verticalalignment="center")
    gap = ax.text(
        4370,
        145,
        "×",
        color=GAP_COLOR,
        fontsize=14,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        zorder=30,
    )
    gap.set_gid(f"{species.value}-legend-gap")
    ax.text(4410, 145, "Direct evidence gap", color=INK, fontsize=7.2, verticalalignment="center")


def build_evidence_figure(species: EvidenceSpecies) -> tuple[Figure, Axes]:
    figure, ax = build_figure()
    draw_evidence_header(ax, species)
    for evidence in markers_for(species):
        draw_marker(ax, species, evidence)
    return figure, ax


def render(species: EvidenceSpecies, output_dir: Path) -> EvidenceOutput:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"retina_signal_flow_evidence_{species.value}"
    output = EvidenceOutput(species, stem.with_suffix(".png"), stem.with_suffix(".svg"), stem.with_suffix(".pdf"))
    figure, _ = build_evidence_figure(species)
    figure.savefig(output.png_path, dpi=DPI, facecolor=CANVAS, metadata={"Software": "matplotlib"})
    figure.savefig(output.svg_path, facecolor=CANVAS, metadata={"Creator": "matplotlib"})
    figure.savefig(output.pdf_path, facecolor=CANVAS, metadata={"Creator": "matplotlib"})
    plt.close(figure)
    return output


def render_all(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[EvidenceOutput, EvidenceOutput]:
    return (
        render(EvidenceSpecies.HUMAN, output_dir),
        render(EvidenceSpecies.MACAQUE, output_dir),
    )


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_OUTPUT_DIR
    if len(sys.argv) > 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [OUTPUT_DIR]")
    for output in render_all(output_dir):
        print(output.png_path)
        print(output.svg_path)
        print(output.pdf_path)


if __name__ == "__main__":
    main()
