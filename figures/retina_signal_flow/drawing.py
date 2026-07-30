from __future__ import annotations

from dataclasses import dataclass
from textwrap import wrap
from typing import Final, Literal

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text

if __package__:
    from .content import CardContent
else:
    from content import CardContent

type Align = Literal["left", "center", "right"]

CANVAS: Final = "#FDFDFC"
SURFACE: Final = "#FFFFFF"
INK: Final = "#17181B"
MUTED: Final = "#5F6269"
BORDER: Final = "#A7A9AE"
GUIDE: Final = "#D6D7DA"

INPUT: Final = "#666A70"
INPUT_FILL: Final = "#F3F3F4"
CONE_STAGE: Final = "#78906B"
CONE_FILL: Final = "#F1F5EE"
PROCESSING: Final = "#6D568D"
PROCESSING_FILL: Final = "#F4F1F8"
AMACRINE: Final = "#9B6A1E"
AMACRINE_FILL: Final = "#FBF4E8"
OUTPUT: Final = "#2C6294"
OUTPUT_FILL: Final = "#EEF4FA"

L_CONE: Final = "#B64C52"
L_FILL: Final = "#F5DADC"
M_CONE: Final = "#59834F"
M_FILL: Final = "#DDEAD9"
S_CONE: Final = "#3F6FA8"
S_FILL: Final = "#DBE7F5"
ON: Final = "#C27A12"
ON_FILL: Final = "#FFF3DA"
OFF: Final = "#2F67A3"
OFF_FILL: Final = "#E7F0FA"
INHIBITORY: Final = "#5B3D77"
INHIBITORY_FILL: Final = "#F1ECF6"


def wrap_card_line(value: str, width: int) -> tuple[str, ...]:
    lines = wrap(value, width=width, break_long_words=False, break_on_hyphens=False)
    return tuple(lines) if lines else ("",)


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class TextStyle:
    size: float = 8.5
    color: str = INK
    weight: str = "normal"
    align: Align = "center"
    italic: bool = False


@dataclass(frozen=True, slots=True)
class Stroke:
    color: str = INK
    width: float = 1.8
    dashed: bool = False


@dataclass(frozen=True, slots=True)
class BoxStyle:
    face: str = SURFACE
    edge: str = BORDER
    width: float = 1.1
    radius: float = 18


@dataclass(frozen=True, slots=True)
class StageSpec:
    rect: Rect
    label: str
    accent: str
    fill: str


@dataclass(frozen=True, slots=True)
class CardSpec:
    rect: Rect
    content: CardContent
    accent: str
    fill: str
    element_id: str | None = None


@dataclass(frozen=True, slots=True)
class RFSpec:
    rect: Rect
    label: str
    accent: str
    fill: str


@dataclass(frozen=True, slots=True)
class Diagram:
    ax: Axes

    def text(self, position: tuple[float, float], value: str, style: TextStyle = TextStyle()) -> Text:
        return self.ax.text(
            *position,
            value,
            fontsize=style.size,
            color=style.color,
            fontweight=style.weight,
            fontstyle="italic" if style.italic else "normal",
            ha=style.align,
            va="center",
            linespacing=1.2,
        )

    def box(self, rect: Rect, style: BoxStyle = BoxStyle()) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            (rect.x, rect.y),
            rect.width,
            rect.height,
            boxstyle=f"round,pad=0.01,rounding_size={style.radius}",
            facecolor=style.face,
            edgecolor=style.edge,
            linewidth=style.width,
        )
        self.ax.add_patch(patch)
        return patch

    def line(self, points: tuple[tuple[float, float], ...], stroke: Stroke = Stroke()) -> Line2D:
        line = self.ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=stroke.color,
            linewidth=stroke.width,
            solid_capstyle="round",
            solid_joinstyle="round",
        )[0]
        if stroke.dashed:
            line.set_dashes((6, 6))
        return line

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        stroke: Stroke = Stroke(),
    ) -> FancyArrowPatch:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=stroke.width,
            color=stroke.color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3",
        )
        if stroke.dashed:
            arrow.set_linestyle((0, (6, 6)))
        self.ax.add_patch(arrow)
        return arrow

    def path_arrow(
        self,
        points: tuple[tuple[float, float], ...],
        stroke: Stroke = Stroke(),
    ) -> FancyArrowPatch:
        self.line(points[:-1], stroke)
        return self.arrow(points[-2], points[-1], stroke)

    def curve(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        bend: float,
        stroke: Stroke,
    ) -> FancyArrowPatch:
        arrow = FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={bend}",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=stroke.width,
            color=stroke.color,
            linestyle=(0, (6, 6)) if stroke.dashed else "solid",
        )
        self.ax.add_patch(arrow)
        return arrow

    def t_bar(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        stroke: Stroke,
    ) -> tuple[Line2D, Line2D]:
        connection = self.line((start, end), stroke)
        bar = self.line(((end[0], end[1] - 22), (end[0], end[1] + 22)), stroke)
        return connection, bar

    def stage(self, spec: StageSpec) -> None:
        self.box(spec.rect, BoxStyle(spec.fill, spec.accent, 1.25, 18))
        self.text(
            (spec.rect.x + spec.rect.width / 2, spec.rect.y + spec.rect.height / 2),
            spec.label,
            TextStyle(11.5, INK, "bold"),
        )

    def card(self, spec: CardSpec) -> None:
        patch = self.box(spec.rect, BoxStyle(spec.fill, spec.accent, 1.0, 14))
        if spec.element_id:
            patch.set_gid(spec.element_id)
        left = spec.rect.x + 24
        title = self.text(
            (left, spec.rect.y + 32),
            spec.content.title,
            TextStyle(7.5, INK, "bold", "left"),
        )
        if spec.element_id:
            title.set_gid(f"{spec.element_id}-title")
        self.line(
            ((left, spec.rect.y + 58), (spec.rect.x + spec.rect.width - 24, spec.rect.y + 58)),
            Stroke(spec.accent, 0.9),
        )
        y = spec.rect.y + 91
        max_chars = max(18, int((spec.rect.width - 48) / 16.5))
        for value in spec.content.lines:
            is_label = value.startswith(("HUMAN", "MACAQUE", "MODEL"))
            is_citation = value.startswith("[")
            for line in wrap_card_line(value, max_chars):
                self.text(
                    (left, y),
                    line,
                    TextStyle(
                        6.6 if is_citation else 7.0,
                        MUTED if is_citation else INK,
                        "bold" if is_label else "normal",
                        "left",
                    ),
                )
                y += 31
            if is_label:
                y += 4

    def rf_tile(self, spec: RFSpec) -> None:
        self.box(spec.rect, BoxStyle(SURFACE, spec.accent, 1.0, 12))
        cx = spec.rect.x + spec.rect.width / 2
        cy = spec.rect.y + spec.rect.height * 0.67
        self.text((cx, spec.rect.y + 39), spec.label, TextStyle(6.8, INK, "bold"))
        self.ax.add_patch(Circle((cx, cy), 55, facecolor=spec.fill, edgecolor=spec.accent, linewidth=1.0))
        self.ax.add_patch(Circle((cx, cy), 25, facecolor=spec.accent, edgecolor=spec.accent, linewidth=1.0))
