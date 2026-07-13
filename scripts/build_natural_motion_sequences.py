from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.image_folder_stimulus import iter_image_paths


class NaturalSequenceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NaturalSequenceConfig:
    input_dir: Path
    output_dir: Path
    train_count: int = 8
    val_count: int = 2
    test_count: int = 2
    frame_count: int = 32
    image_size: int = 256
    seed: int = 7
    max_shift_fraction: float = 0.10
    max_zoom_fraction: float = 0.06
    luminance_fraction: float = 0.08
    occlusion_fraction: float = 0.20

    def __post_init__(self) -> None:
        if min(self.train_count, self.val_count, self.test_count) < 0:
            raise NaturalSequenceError("Split counts must be non-negative")
        if self.train_count < 1:
            raise NaturalSequenceError("train_count must be positive")
        if self.frame_count < 2 or self.image_size < 8:
            raise NaturalSequenceError("frame_count and image_size are too small")
        fractions = (
            self.max_shift_fraction,
            self.max_zoom_fraction,
            self.luminance_fraction,
            self.occlusion_fraction,
        )
        if any(value < 0 or value >= 0.5 for value in fractions):
            raise NaturalSequenceError("Sequence fractions must lie in [0, 0.5)")


@dataclass(frozen=True, slots=True)
class NaturalSequenceRecord:
    split: str
    source_path: Path
    sequence_dir: Path
    seed: int


def build_natural_sequences(
    config: NaturalSequenceConfig,
) -> tuple[NaturalSequenceRecord, ...]:
    sources = list(iter_image_paths(config.input_dir, recursive=True))
    requested = config.train_count + config.val_count + config.test_count
    if len(sources) < requested:
        raise NaturalSequenceError(
            f"Need {requested} source images, found {len(sources)} in {config.input_dir}"
        )
    random.Random(config.seed).shuffle(sources)
    assignments = _split_assignments(config)
    records = []
    for index, (source, split) in enumerate(zip(sources[:requested], assignments)):
        sequence_dir = config.output_dir / split / f"{index:03d}_{source.stem}"
        sequence_seed = config.seed + index
        _write_sequence(source, sequence_dir, config, sequence_seed)
        records.append(NaturalSequenceRecord(split, source, sequence_dir, sequence_seed))
    _write_manifest(config.output_dir / "sequence_manifest.csv", records)
    return tuple(records)


def _split_assignments(config: NaturalSequenceConfig) -> tuple[str, ...]:
    return (
        ("train",) * config.train_count
        + ("val",) * config.val_count
        + ("test",) * config.test_count
    )


def _write_sequence(
    source_path: Path,
    sequence_dir: Path,
    config: NaturalSequenceConfig,
    seed: int,
) -> None:
    rng = random.Random(seed)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        base = _prepare_base(image.convert("RGB"), config)
    parameters = _sample_motion_parameters(rng, config)
    for frame_index in range(config.frame_count):
        frame = _make_frame(frame_index, base, parameters, config)
        frame.save(sequence_dir / f"{frame_index:04d}.png")


def _prepare_base(image: Image.Image, config: NaturalSequenceConfig) -> Image.Image:
    padding = math.ceil(
        config.image_size
        * (1 + 2 * config.max_shift_fraction + 2 * config.max_zoom_fraction)
    )
    return ImageOps.fit(
        image,
        (padding, padding),
        method=Image.Resampling.LANCZOS,
    )


def _sample_motion_parameters(
    rng: random.Random,
    config: NaturalSequenceConfig,
) -> tuple[float, ...]:
    shift = config.image_size * rng.uniform(
        config.max_shift_fraction / 2,
        config.max_shift_fraction,
    )
    angle = rng.uniform(0, 2 * math.pi)
    zoom = rng.uniform(-config.max_zoom_fraction, config.max_zoom_fraction)
    luminance_phase = rng.uniform(0, 2 * math.pi)
    occluder_size = config.image_size * config.occlusion_fraction
    occluder_x = rng.uniform(0.1, 0.7) * config.image_size
    occluder_y = rng.uniform(0.1, 0.7) * config.image_size
    return (
        shift * math.cos(angle),
        shift * math.sin(angle),
        zoom,
        luminance_phase,
        occluder_size,
        occluder_x,
        occluder_y,
    )


def _make_frame(
    frame_index: int,
    base: Image.Image,
    parameters: tuple[float, ...],
    config: NaturalSequenceConfig,
) -> Image.Image:
    progress = frame_index / (config.frame_count - 1)
    smooth = 0.5 - 0.5 * math.cos(math.pi * progress)
    dx, dy, zoom, phase, side, occluder_x, occluder_y = parameters
    scale = 1 + zoom * (2 * smooth - 1)
    crop_size = round(config.image_size / scale)
    center = base.width / 2
    left = round(center + dx * (2 * smooth - 1) - crop_size / 2)
    top = round(center + dy * (2 * smooth - 1) - crop_size / 2)
    left = min(max(left, 0), base.width - crop_size)
    top = min(max(top, 0), base.height - crop_size)
    frame = base.crop((left, top, left + crop_size, top + crop_size)).resize(
        (config.image_size, config.image_size),
        Image.Resampling.LANCZOS,
    )
    luminance = 1 + config.luminance_fraction * math.sin(2 * math.pi * progress + phase)
    frame = ImageEnhance.Brightness(frame).enhance(luminance)
    if side > 0 and 0.25 <= progress <= 0.75:
        mean = tuple(round(value) for value in ImageStat.Stat(frame).mean)
        draw = ImageDraw.Draw(frame)
        draw.rectangle(
            (occluder_x, occluder_y, occluder_x + side, occluder_y + side),
            fill=mean,
        )
    return frame


def _write_manifest(path: Path, records: Sequence[NaturalSequenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "source_path", "sequence_dir", "seed"))
        for record in records:
            writer.writerow(
                (record.split, record.source_path, record.sequence_dir, record.seed)
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    records = build_natural_sequences(
        NaturalSequenceConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            frame_count=args.frame_count,
            image_size=args.image_size,
            seed=args.seed,
        )
    )
    print(f"sequence_manifest={args.output_dir / 'sequence_manifest.csv'}")
    print(f"sequences={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
