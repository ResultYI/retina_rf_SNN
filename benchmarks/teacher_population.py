from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from data.rgc_response import CellMetadata


_POPULATION_REVISION: Final = "hierarchical-synthetic-teacher-v1"
_GROUP_TYPES: Final = ("midget", "midget", "parasol", "parasol")
_GROUP_POLARITIES: Final = (0, 1, 0, 1)
_GROUP_HIGH_SCALES: Final = (0.85, 0.90, 1.10, 1.15)
_TYPE_COMPONENTS: Final = (-0.125, -0.125, 0.125, 0.125)
_POLARITY_COMPONENTS: Final = (-0.025, 0.025, -0.025, 0.025)
_GROUP_IDS: Final = ("midget-on", "midget-off", "parasol-on", "parasol-off")
_COMPONENT_IDS: Final = ("population", "type", "polarity", "cell_residual")


class SyntheticTeacherError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TeacherPopulationConfig:
    cells_per_type_polarity: int = 4
    residual_bound: float = 0.03
    residual_seed: int = 0

    def __post_init__(self) -> None:
        if self.cells_per_type_polarity < 1:
            raise SyntheticTeacherError("cells_per_type_polarity must be positive")
        if not np.isfinite(self.residual_bound) or self.residual_bound < 0:
            raise SyntheticTeacherError(
                "cell residual bound must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class TeacherPopulation:
    cells: CellMetadata
    group_ids: tuple[str, ...]
    replicate_ids: tuple[str, ...]
    high_scales: np.ndarray
    residuals: np.ndarray
    type_components: np.ndarray
    polarity_components: np.ndarray
    generation_seed: int
    residual_seed: int
    config: TeacherPopulationConfig


def build_teacher_population(
    cone_positions_degs: np.ndarray,
    config: TeacherPopulationConfig,
    *,
    generation_seed: int,
) -> TeacherPopulation:
    replicate_count = config.cells_per_type_polarity
    center_indices = np.rint(
        np.linspace(0, cone_positions_degs.shape[0] - 1, replicate_count)
    ).astype(np.int64)
    if len(set(int(index) for index in center_indices)) != replicate_count:
        raise SyntheticTeacherError(
            "cells_per_type_polarity needs distinct cone-supported centers"
        )
    residual_seed = int(generation_seed) + config.residual_seed
    residual_template = _cell_residuals(config, residual_seed)
    ids: list[str] = []
    type_ids: list[str] = []
    group_ids: list[str] = []
    replicate_ids: list[str] = []
    polarities: list[int] = []
    high_scales: list[float] = []
    residuals: list[float] = []
    type_components: list[float] = []
    polarity_components: list[float] = []
    positions: list[np.ndarray] = []
    for group, (type_id, polarity, group_id) in enumerate(
        zip(_GROUP_TYPES, _GROUP_POLARITIES, _GROUP_IDS, strict=True)
    ):
        for replicate, center_index in enumerate(center_indices):
            residual = float(residual_template[replicate])
            ids.append(f"synthetic-{group_id}-r{replicate}")
            type_ids.append(type_id)
            group_ids.append(group_id)
            replicate_ids.append(f"r{replicate}")
            polarities.append(polarity)
            high_scales.append(_GROUP_HIGH_SCALES[group] + residual)
            residuals.append(residual)
            type_components.append(_TYPE_COMPONENTS[group])
            polarity_components.append(_POLARITY_COMPONENTS[group])
            positions.append(cone_positions_degs[int(center_index)])
    return TeacherPopulation(
        cells=CellMetadata(
            ids=tuple(ids),
            type_ids=tuple(type_ids),
            polarities=np.asarray(polarities, dtype=np.int64),
            positions_degs=np.stack(positions).astype(np.float32),
            eccentricities_deg=np.full(len(ids), 4.0, dtype=np.float32),
        ),
        group_ids=tuple(group_ids),
        replicate_ids=tuple(replicate_ids),
        high_scales=np.asarray(high_scales, dtype=np.float32),
        residuals=np.asarray(residuals, dtype=np.float32),
        type_components=np.asarray(type_components, dtype=np.float32),
        polarity_components=np.asarray(polarity_components, dtype=np.float32),
        generation_seed=int(generation_seed),
        residual_seed=residual_seed,
        config=config,
    )


def teacher_population_metadata(
    population: TeacherPopulation,
) -> dict[str, np.ndarray]:
    return {
        "cell_group_id": np.asarray(population.group_ids),
        "cell_replicate_id": np.asarray(population.replicate_ids),
        "component_id": np.asarray(_COMPONENT_IDS),
        "revision": np.asarray((_POPULATION_REVISION,)),
        "generation_seed": np.asarray((population.generation_seed,), dtype=np.int64),
        "residual_seed": np.asarray((population.residual_seed,), dtype=np.int64),
        "cells_per_type_polarity": np.asarray(
            (population.config.cells_per_type_polarity,),
            dtype=np.int64,
        ),
        "residual_bound": np.asarray((population.config.residual_bound,), dtype=np.float32),
        "context_high_scale": population.high_scales,
        "context_gain_population_component": np.ones_like(population.high_scales),
        "context_gain_type_component": population.type_components,
        "context_gain_polarity_component": population.polarity_components,
        "context_gain_cell_residual": population.residuals,
    }


def _cell_residuals(config: TeacherPopulationConfig, residual_seed: int) -> np.ndarray:
    count = config.cells_per_type_polarity
    if count == 1 or config.residual_bound == 0:
        return np.zeros(count, dtype=np.float32)
    rng = np.random.default_rng(residual_seed)
    residuals = rng.uniform(
        -config.residual_bound / count,
        config.residual_bound / count,
        count - 1,
    ).astype(np.float32)
    return np.append(
        residuals,
        np.asarray([-residuals.sum(dtype=np.float32)], dtype=np.float32),
    )


__all__ = [
    "SyntheticTeacherError",
    "TeacherPopulation",
    "TeacherPopulationConfig",
    "build_teacher_population",
    "teacher_population_metadata",
]
