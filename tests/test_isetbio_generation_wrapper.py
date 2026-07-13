from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_isetbio_h5 import (
    collect_sources,
    load_generation_config,
    matlab_string,
)
from scripts.isetbio_stage1 import StageMinusOneError


def test_generation_config_parses_flat_yaml_when_paths_are_relative(
    tmp_path: Path,
) -> None:
    # Given
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "h5"
    image_dir.mkdir()
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        "\n".join(
            (
                "input_path: images",
                "output_dir: h5",
                "max_items: 2",
                "time_steps: 16",
                "dt_ms: 5.0",
                "field_of_view_deg: 0.5",
                "export_crop_fov_deg: 0.2",
                "achromatic_stimulus_enabled: true",
                "treat_child_directories_as_sequences: true",
                "reproducibility_check: false",
            )
        ),
        encoding="utf-8",
    )

    # When
    config = load_generation_config(config_path)

    # Then
    assert config.input_path == image_dir
    assert config.output_dir == output_dir
    assert config.max_items == 2
    assert config.time_steps == 16
    assert config.dt_ms == 5.0
    assert config.export_crop_fov_deg == 0.2
    assert config.achromatic_stimulus_enabled
    assert config.sequence_root
    assert not config.reproducibility_check


def test_collect_sources_uses_sorted_image_files(tmp_path: Path) -> None:
    # Given
    for name in ("b.jpg", "a.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")

    # When
    sources = collect_sources(tmp_path, max_items=2, sequence_directory=False)

    # Then
    assert [source.name for source in sources] == ["a.png", "b.jpg"]


def test_collect_sources_uses_sorted_child_directories_as_sequences(
    tmp_path: Path,
) -> None:
    # Given
    (tmp_path / "sequence_b").mkdir()
    (tmp_path / "sequence_a").mkdir()
    (tmp_path / "still.png").write_bytes(b"x")

    # When
    sources = collect_sources(
        tmp_path,
        max_items=2,
        sequence_directory=False,
        sequence_root=True,
    )

    # Then
    assert [source.name for source in sources] == ["sequence_a", "sequence_b"]


def test_matlab_string_escapes_single_quotes() -> None:
    assert matlab_string(Path("a'b")) == "'a''b'"


def test_generation_config_rejects_crop_larger_than_source_fov(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "input_path: input.png\noutput_dir: h5\nfield_of_view_deg: 0.5\n"
        "export_crop_fov_deg: 0.6\n",
        encoding="utf-8",
    )

    with pytest.raises(StageMinusOneError, match="must not exceed"):
        load_generation_config(config_path)
