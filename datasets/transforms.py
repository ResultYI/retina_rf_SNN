from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})


class DatasetTransformError(ValueError):
    pass


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def stimulus_kind(path: Path) -> str:
    if is_image_path(path):
        return "image"
    if is_video_path(path):
        return "video"
    raise DatasetTransformError(f"Unsupported stimulus extension: {path.suffix}")
