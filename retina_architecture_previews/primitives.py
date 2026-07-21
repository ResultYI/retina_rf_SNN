from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageDraw

from retina_architecture_previews.design import (
    Box,
    Color,
    Font,
    PALETTE,
    Point,
    TYPE,
    font,
)


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (2400, 1350), PALETTE.canvas)
    return image, ImageDraw.Draw(image)


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: Box,
    *,
    fill: Color = PALETTE.surface,
    outline: Color = PALETTE.border,
    radius: int = 24,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font: Font, width: int) -> str:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=text_font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def wrapped_text(
    draw: ImageDraw.ImageDraw,
    position: Point,
    text: str,
    *,
    text_font: Font,
    fill: Color,
    max_width: int,
    spacing: int = 7,
    anchor: str | None = None,
    align: str = "left",
) -> None:
    draw.multiline_text(
        position,
        wrap_text(draw, text, text_font, max_width),
        font=text_font,
        fill=fill,
        spacing=spacing,
        anchor=anchor,
        align=align,
    )


def page_header(draw: ImageDraw.ImageDraw, number: str, title: str, lead: str) -> None:
    draw.text((96, 70), number, font=font(24, bold=True), fill=PALETTE.teal)
    draw.text((150, 57), title, font=font(TYPE.title, bold=True), fill=PALETTE.ink)
    draw.text((150, 142), lead, font=font(TYPE.lead), fill=PALETTE.muted)
    draw.line((96, 210, 2304, 210), fill=PALETTE.border, width=3)


def section_label(draw: ImageDraw.ImageDraw, position: Point, text: str) -> None:
    draw.text(position, text.upper(), font=font(TYPE.section, bold=True), fill=PALETTE.muted)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    *,
    color: Color = PALETTE.ink,
    width: int = 8,
    head: int = 22,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        int(end[0] - head * math.cos(angle) + head * 0.55 * math.sin(angle)),
        int(end[1] - head * math.sin(angle) - head * 0.55 * math.cos(angle)),
    )
    right = (
        int(end[0] - head * math.cos(angle) - head * 0.55 * math.sin(angle)),
        int(end[1] - head * math.sin(angle) + head * 0.55 * math.cos(angle)),
    )
    draw.polygon((end, left, right), fill=color)


def elbow_arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Point],
    *,
    color: Color = PALETTE.structure,
    width: int = 5,
    head: int = 18,
) -> None:
    if len(points) < 2:
        return
    draw.line(tuple(points), fill=color, width=width, joint="curve")
    arrow(draw, points[-2], points[-1], color=color, width=width, head=head)


def inhibitory_line(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    *,
    width: int = 7,
    bar: int = 26,
) -> None:
    draw.line((*start, *end), fill=PALETTE.inhibition, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0]) + math.pi / 2
    offset = (int(bar * math.cos(angle)), int(bar * math.sin(angle)))
    draw.line(
        (end[0] - offset[0], end[1] - offset[1], end[0] + offset[0], end[1] + offset[1]),
        fill=PALETTE.inhibition,
        width=width,
    )


def cone_triangle(
    draw: ImageDraw.ImageDraw,
    center: Point,
    size: int,
    fill: Color,
    *,
    outline: Color | None = None,
) -> None:
    x, y = center
    points = ((x, y - size), (x - size, y + size), (x + size, y + size))
    draw.polygon(points, fill=fill, outline=outline)


def rod(draw: ImageDraw.ImageDraw, center: Point, width: int, height: int) -> None:
    x, y = center
    draw.rounded_rectangle(
        (x - width // 2, y - height // 2, x + width // 2, y + height // 2),
        radius=max(2, width // 4),
        fill=PALETTE.rod,
    )


def landscape(draw: ImageDraw.ImageDraw, box: Box, *, shift: int = 0) -> None:
    x0, y0, x1, y1 = box
    span_x = x1 - x0
    span_y = y1 - y0
    mid = y0 + int(span_y * 0.58)
    draw.rectangle(box, fill=PALETTE.sky)
    sun_size = max(10, min(50, span_y // 4))
    sun_right = min(x1 - 4, x1 - max(6, span_x // 12) + shift)
    sun_top = y0 + max(5, span_y // 10)
    draw.ellipse((sun_right - sun_size, sun_top, sun_right, sun_top + sun_size), fill=(225, 185, 96))
    first_peak = max(y0 + 12, mid - 74)
    second_peak = max(y0 + 12, mid - 98)
    draw.polygon(
        (
            (x0, min(y1, mid + int(span_y * 0.16))),
            (x0 + int(span_x * 0.28) + shift, first_peak),
            (x0 + int(span_x * 0.52) + shift, mid),
            (x0 + int(span_x * 0.72) + shift, second_peak),
            (x1, min(y1, mid + int(span_y * 0.24))),
        ),
        fill=PALETTE.ridge,
    )
    draw.rectangle((x0, mid, x1, y1), fill=PALETTE.ground)
    trunk_x = x0 + int(span_x * 0.74) + shift
    trunk_half_width = max(3, span_x // 50)
    crown_radius = max(7, min(36, span_x // 8, span_y // 4))
    draw.rectangle((trunk_x - trunk_half_width, mid - 4, trunk_x + trunk_half_width, y1 - 6), fill=(112, 84, 62))
    draw.ellipse((trunk_x - crown_radius, mid - crown_radius * 2, trunk_x + crown_radius, mid + 4), fill=(73, 112, 79))
    draw.rectangle(box, outline=PALETTE.border, width=2)


def sampled_landscape(draw: ImageDraw.ImageDraw, box: Box, *, shift: int = 0) -> None:
    landscape(draw, box, shift=shift)
    x0, y0, x1, y1 = box
    spacing = 30
    for row_index, y in enumerate(range(y0 + 18, y1 - 12, spacing)):
        offset = spacing // 2 if row_index % 2 else 0
        for column_index, x in enumerate(range(x0 + 18 + offset, x1 - 12, spacing)):
            type_index = (row_index * 5 + column_index * 3) % 17
            fill = PALETTE.s_cone if type_index == 0 else PALETTE.l_cone if type_index % 2 else PALETTE.m_cone
            cone_triangle(draw, (x, y), 5, fill)


def waveform(
    draw: ImageDraw.ImageDraw,
    box: Box,
    *,
    sustained: bool,
    color: Color,
) -> None:
    x0, y0, x1, y1 = box
    baseline = y1 - 10
    draw.line((x0, baseline, x1, baseline), fill=PALETTE.border, width=2)
    if sustained:
        points = ((x0, baseline), (x0 + 22, y0 + 10), (x0 + 54, y0 + 22), (x1, y0 + 26))
    else:
        points = ((x0, baseline), (x0 + 18, y0 + 8), (x0 + 42, baseline), (x1, baseline))
    draw.line(points, fill=color, width=6, joint="curve")
