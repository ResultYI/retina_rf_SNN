from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from figures.retina_signal_flow.generate_training_evaluation_flow import HEIGHT, WIDTH, render


def ids_with_prefix(svg_path: Path, prefix: str) -> tuple[str, ...]:
    root = ElementTree.parse(svg_path).getroot()
    return tuple(
        element_id
        for element in root.iter()
        if (element_id := element.get("id")) is not None and element_id.startswith(prefix)
    )


def test_renders_training_and_post_hoc_rf_flow(tmp_path: Path) -> None:
    # Given
    output_stem = tmp_path / "retina_snn_training_evaluation_flow"

    # When
    png_path, svg_path, pdf_path = render(output_stem)

    # Then
    with Image.open(png_path) as image:
        assert image.size == (WIDTH, HEIGHT)
        assert image.info["dpi"][0] >= 299

    svg = svg_path.read_text(encoding="utf-8")
    svg_text = " ".join(
        " ".join(ElementTree.parse(svg_path).getroot().itertext()).split()
    )
    required_text = (
        "Experimental stimulus",
        "Species-matched input model",
        "Retina SNN",
        "Recorded-cell RGC output",
        "Likelihood loss",
        "Freeze model",
        "STA / Jacobian / GLM",
        "Dynamic effective RF",
        "One model output unit per recorded RGC",
        "Fixed metadata: cell type · ON/OFF polarity · eccentricity",
        "Recorded spikes = supervision",
        "RF is not in the training loss",
        "RF extracted after freezing",
    )
    assert all(value in svg_text for value in required_text)
    assert "<text" in svg
    assert "<image" not in svg
    for prefix in ("one-to-one-mapping", "fixed-metadata", "recorded-spike-supervision", "rf-post-hoc"):
        assert ids_with_prefix(svg_path, prefix)
    assert pdf_path.read_bytes().startswith(b"%PDF")
