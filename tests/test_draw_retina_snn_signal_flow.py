from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from figures.retina_signal_flow.generate_retina_signal_flow import HEIGHT, WIDTH, render


def element_with_id(svg_path: Path, element_id: str) -> ElementTree.Element:
    root = ElementTree.parse(svg_path).getroot()
    match = next((element for element in root.iter() if element.get("id") == element_id), None)
    assert match is not None
    return match


def test_renders_complete_scientific_figure(tmp_path: Path) -> None:
    # Given
    output_stem = tmp_path / "retina_snn_signal_flow"

    # When
    png_path, svg_path = render(output_stem)
    pdf_path = output_stem.with_suffix(".pdf")

    # Then
    with Image.open(png_path) as image:
        assert image.size == (WIDTH, HEIGHT)
        assert image.info["dpi"][0] >= 299

    svg = svg_path.read_text(encoding="utf-8")
    svg_text = " ".join(ElementTree.parse(svg_path).getroot().itertext())
    required_text = (
        "Retina SNN Signal Flow",
        "L / M / S cones",
        "H1 horizontal network",
        "ON bipolar",
        "OFF bipolar",
        "Local amacrine",
        "ON RGC",
        "OFF RGC",
        "Optic nerve",
        "Connection legend",
    )
    assert all(value in svg_text for value in required_text)
    assert "<text" in svg
    assert "<image" not in svg
    assert "marmoset" not in svg.lower()
    assert "cone mosaic" not in svg.lower()
    assert "MODEL" not in svg_text
    assert "Timing / mechanism" not in svg_text
    assert "Timing notation" not in svg_text
    assert "et al." not in svg_text
    assert "HUMAN" not in svg_text
    assert "MACAQUE" not in svg_text
    assert "Tpeak" not in svg_text
    assert "?" not in svg_text
    assert "S: weak / absent H1 input" in svg
    for element_id in (
        "direct-on-bipolar-to-rgc",
        "direct-off-bipolar-to-rgc",
        "on-bipolar-to-amacrine",
        "off-bipolar-to-amacrine",
        "amacrine-to-on-rgc-inhibition-line",
        "amacrine-to-on-rgc-inhibition-bar",
        "amacrine-to-off-rgc-inhibition-line",
        "amacrine-to-off-rgc-inhibition-bar",
    ):
        element = element_with_id(svg_path, element_id)
        assert any(child.tag.endswith("path") for child in element.iter())
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_omits_evidence_cards_from_flow_figure(tmp_path: Path) -> None:
    # Given
    output_stem = tmp_path / "retina_snn_signal_flow"

    # When
    _, svg_path = render(output_stem)
    svg_text = " ".join(ElementTree.parse(svg_path).getroot().itertext())

    # Then
    for removed_heading in (
        "CONE RESPONSE TIMING",
        "CONE DENSITY & COMPOSITION",
        "H1 HORIZONTAL CELL",
        "MIDGET BIPOLAR",
        "DIFFUSE BIPOLAR",
        "LOCAL AMACRINE PATHWAY",
        "RGC · HUMAN",
        "RGC · MACAQUE",
    ):
        assert removed_heading not in svg_text
