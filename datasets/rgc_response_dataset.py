from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset


class RGCResponseDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RGCResponseSample:
    path: Path
    sample_id: str


class RGCResponseDataset(Dataset[RGCResponseSample]):
    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = tuple(Path(path) for path in paths)
        if not self._paths:
            raise RGCResponseDatasetError("RGCResponseDataset needs at least one file")
        for path in self._paths:
            if not path.is_file():
                raise RGCResponseDatasetError(f"RGC response file does not exist: {path}")

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> RGCResponseSample:
        path = self._paths[index]
        return RGCResponseSample(path=path, sample_id=path.stem)
