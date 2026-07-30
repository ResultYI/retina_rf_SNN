from __future__ import annotations

if __package__:
    from .drawing import (
        BORDER,
        INHIBITORY,
        INK,
        MUTED,
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
        INHIBITORY,
        INK,
        MUTED,
        SURFACE,
        BoxStyle,
        Diagram,
        Rect,
        Stroke,
        TextStyle,
    )


def draw_footer(diagram: Diagram) -> None:
    diagram.box(Rect(1030, 1550, 2740, 300), BoxStyle(SURFACE, BORDER, 1.1, 16))
    diagram.text((1100, 1605), "Connection legend", TextStyle(10, INK, "bold", "left"))

    items = (
        (1170, "excitation"),
        (1690, "inhibition"),
        (2160, "feedback"),
        (2610, "weak input"),
        (3160, "network coupling"),
    )
    for x, label in items:
        diagram.text((x + 150, 1760), label, TextStyle(8, INK, align="left"))

    diagram.arrow((1170, 1690), (1270, 1690), Stroke(INK, 1.8))
    diagram.t_bar((1690, 1690), (1790, 1690), Stroke(INK, 1.8))
    diagram.curve((2160, 1700), (2260, 1700), -0.35, Stroke(INHIBITORY, 1.7))
    diagram.arrow((2610, 1690), (2710, 1690), Stroke(MUTED, 1.6, True))
    diagram.line(((3160, 1683), (3260, 1683)), Stroke(INK, 2.2))
    diagram.line(((3160, 1697), (3260, 1697)), Stroke(INK, 1.0))
