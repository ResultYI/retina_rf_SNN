from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from itertools import product
from typing import Final, assert_never

import numpy as np


TARGET_LABELS: Final = (
    "ON parasol",
    "ON midget",
    "OFF parasol",
    "OFF midget",
)


@unique
class CellSelection(StrEnum):
    COLOCALIZED_QUARTET = "colocalized_quartet"
    ALL_QUALITY_1_TARGETS = "all_quality_1_targets"


@dataclass(frozen=True, slots=True)
class CellCatalog:
    labels: tuple[str, ...]
    classes: np.ndarray
    units: np.ndarray
    electrode_grid: np.ndarray


@dataclass(frozen=True, slots=True)
class KaramanlisCellSelectionError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def electrode_grid() -> np.ndarray:
    x_values, y_values = np.meshgrid(np.arange(16), np.arange(15, -1, -1))
    keep = np.ones((16, 16), dtype=bool)
    keep[0, 0] = keep[-1, 0] = keep[0, -1] = keep[-1, -1] = False
    return np.column_stack(
        (
            x_values.flatten(order="F")[keep.flatten(order="F")],
            y_values.flatten(order="F")[keep.flatten(order="F")],
        )
    ).astype(np.float64)


def select_cells(catalog: CellCatalog, selection: CellSelection) -> np.ndarray:
    match selection:
        case CellSelection.COLOCALIZED_QUARTET:
            return _select_colocalized_cells(catalog)
        case CellSelection.ALL_QUALITY_1_TARGETS:
            label_ids = tuple(catalog.labels.index(label) + 1 for label in TARGET_LABELS)
            rows = np.flatnonzero(
                np.isin(catalog.classes, label_ids) & (catalog.units[:, 3] == 1)
            )
            if rows.size == 0:
                raise KaramanlisCellSelectionError("no quality-1 target cells found")
            return rows
        case unreachable:
            assert_never(unreachable)


def _select_colocalized_cells(catalog: CellCatalog) -> np.ndarray:
    groups = []
    for target in TARGET_LABELS:
        label_id = catalog.labels.index(target) + 1
        rows = np.flatnonzero(
            (catalog.classes == label_id) & (catalog.units[:, 3] == 1)
        )
        if rows.size == 0:
            raise KaramanlisCellSelectionError(
                f"no quality-1 {target} cell found"
            )
        groups.append(rows)
    best_score: tuple[float, float] | None = None
    best_rows: tuple[int, ...] | None = None
    for rows in product(*groups):
        points = catalog.electrode_grid[catalog.units[np.asarray(rows), 1] - 1]
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        score = (
            float(distances.max()),
            float(((points - points.mean(0)) ** 2).sum()),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_rows = tuple(int(row) for row in rows)
    if best_rows is None:
        raise KaramanlisCellSelectionError(
            "cell selection produced no valid quartet"
        )
    return np.asarray(best_rows, dtype=np.int64)


__all__ = [
    "CellCatalog",
    "CellSelection",
    "KaramanlisCellSelectionError",
    "electrode_grid",
    "select_cells",
]
