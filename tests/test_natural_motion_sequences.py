from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.build_natural_motion_sequences import (
    NaturalSequenceConfig,
    build_natural_sequences,
)


def test_build_natural_sequences_writes_source_disjoint_motion_splits(
    tmp_path: Path,
) -> None:
    # Given
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(3):
        image = Image.new("RGB", (32, 32))
        for x in range(32):
            for y in range(32):
                image.putpixel((x, y), ((x + index * 20) % 256, y * 4, x * y % 256))
        image.save(raw / f"source_{index}.png")
    config = NaturalSequenceConfig(
        input_dir=raw,
        output_dir=tmp_path / "sequences",
        train_count=1,
        val_count=1,
        test_count=1,
        frame_count=4,
        image_size=16,
        seed=11,
    )

    # When
    records = build_natural_sequences(config)

    # Then
    assert {record.split for record in records} == {"train", "val", "test"}
    assert len({record.source_path for record in records}) == 3
    assert all(len(tuple(record.sequence_dir.glob("*.png"))) == 4 for record in records)
    assert (config.output_dir / "sequence_manifest.csv").is_file()
    first = Image.open(records[0].sequence_dir / "0000.png")
    last = Image.open(records[0].sequence_dir / "0003.png")
    assert list(first.getdata()) != list(last.getdata())
