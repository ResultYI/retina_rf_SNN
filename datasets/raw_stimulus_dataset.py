from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from datasets.transforms import stimulus_kind


class RawStimulusDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    url: str
    target_path: Path
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RawStimulusSample:
    source_path: Path
    source_id: str
    kind: str


class RawStimulusDataset(Dataset[RawStimulusSample]):
    def __init__(self, sources: Sequence[Path]) -> None:
        paths = tuple(Path(source) for source in sources)
        if not paths:
            raise RawStimulusDatasetError("RawStimulusDataset needs at least one source")
        for path in paths:
            if not path.exists():
                raise RawStimulusDatasetError(f"stimulus path does not exist: {path}")
            stimulus_kind(path)
        self._sources = paths

    def __len__(self) -> int:
        return len(self._sources)

    def __getitem__(self, index: int) -> RawStimulusSample:
        path = self._sources[index]
        return RawStimulusSample(path, path.name, stimulus_kind(path))


def ensure_dataset_root(
    root: Path,
    download: DownloadSpec | None,
    allow_download: bool,
) -> Path:
    if root.exists():
        return root
    if download is None or not allow_download:
        raise RawStimulusDatasetError(
            f"Missing raw stimulus root: {root}. Pass allow_download=True to download."
        )
    download_file(download)
    if not root.exists():
        raise RawStimulusDatasetError(
            f"Downloaded {download.target_path}, but dataset root is still missing: {root}"
        )
    return root


def download_file(spec: DownloadSpec) -> Path:
    spec.target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(spec.url, spec.target_path)
    if spec.sha256 is not None and _sha256(spec.target_path) != spec.sha256:
        raise RawStimulusDatasetError(f"SHA256 mismatch for {spec.target_path}")
    return spec.target_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
