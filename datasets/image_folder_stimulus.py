from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets.raw_stimulus_dataset import (
    DownloadSpec,
    RawStimulusDataset,
    RawStimulusDatasetError,
    ensure_dataset_root,
)
from datasets.transforms import is_image_path


@dataclass(frozen=True, slots=True)
class ImageFolderStimulusConfig:
    root: Path
    recursive: bool = False
    max_items: int | None = None
    download: DownloadSpec | None = None
    allow_download: bool = False

    def __post_init__(self) -> None:
        if self.max_items is not None and self.max_items < 1:
            raise RawStimulusDatasetError("max_items must be positive when set")


class ImageFolderStimulusDataset(RawStimulusDataset):
    def __init__(self, config: ImageFolderStimulusConfig) -> None:
        root = ensure_dataset_root(
            Path(config.root),
            config.download,
            config.allow_download,
        )
        paths = tuple(iter_image_paths(root, config.recursive))
        if config.max_items is not None:
            paths = paths[: config.max_items]
        super().__init__(paths)


def iter_image_paths(root: Path, recursive: bool = False) -> tuple[Path, ...]:
    if not root.is_dir():
        raise RawStimulusDatasetError(f"image root must be a directory: {root}")
    pattern = "**/*" if recursive else "*"
    return tuple(
        sorted(path for path in root.glob(pattern) if path.is_file() and is_image_path(path))
    )
