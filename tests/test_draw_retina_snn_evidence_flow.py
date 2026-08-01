from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from figures.retina_signal_flow.generate_evidence_flows import (
    EvidenceSpecies,
    HEIGHT,
    WIDTH,
    render_all,
)


def ids_with_prefix(svg_path: Path, prefix: str) -> tuple[str, ...]:
    root = ElementTree.parse(svg_path).getroot()
    return tuple(
        element_id
        for element in root.iter()
        if (element_id := element.get("id")) is not None and element_id.startswith(prefix)
    )


def test_renders_species_evidence_maps_from_the_base_flow(tmp_path: Path) -> None:
    # Given
    output_dir = tmp_path / "evidence"

    # When
    human, macaque = render_all(output_dir)

    # Then
    assert human.species is EvidenceSpecies.HUMAN
    assert macaque.species is EvidenceSpecies.MACAQUE
    for output in (human, macaque):
        with Image.open(output.png_path) as image:
            assert image.size == (WIDTH, HEIGHT)
            assert image.info["dpi"][0] >= 299
        assert output.pdf_path.read_bytes().startswith(b"%PDF")
        svg_text = " ".join(ElementTree.parse(output.svg_path).getroot().itertext())
        assert "Evidence available" in svg_text
        assert "Direct evidence gap" in svg_text
        assert ids_with_prefix(output.svg_path, f"{output.species.value}-supported-")
        assert ids_with_prefix(output.svg_path, f"{output.species.value}-gap-")

    assert len(ids_with_prefix(human.svg_path, "human-gap-")) > len(
        ids_with_prefix(human.svg_path, "human-supported-")
    )
    assert ids_with_prefix(human.svg_path, "human-supported-optic-output")
    assert not ids_with_prefix(human.svg_path, "human-gap-optic-output")
    assert len(ids_with_prefix(macaque.svg_path, "macaque-gap-")) == 1
