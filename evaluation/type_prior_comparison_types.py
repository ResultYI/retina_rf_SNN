from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import numpy as np

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap: TypeAlias = dict[str, JsonValue]
Variant: TypeAlias = Literal[
    "type_aware",
    "type_blind",
    "cell_only",
    "shuffled_type",
    "balanced_shuffled_type",
]
ValueStatus: TypeAlias = Literal[
    "supported",
    "not_supported",
    "not_identifiable",
    "significant_disadvantage",
]
OverallStatus: TypeAlias = Literal["supported", "mixed", "not_supported", "not_identifiable"]

HISTORY_CONTRACTS: Final = ("zero", "matched_observed", "standard_train_rate")
BASE_VARIANTS: Final[tuple[Variant, Variant]] = (
    "type_aware",
    "type_blind",
)
SHUFFLE_VARIANTS: Final[tuple[Variant, Variant]] = (
    "shuffled_type",
    "balanced_shuffled_type",
)
REFERENCE_VARIANT: Final[Variant] = "cell_only"
MIN_SEEDS: Final = 2
MIN_SOURCE_PAIRS: Final = 3


class TypePriorComparisonError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunInput:
    path: Path
    variant: Variant
    seed: int
    budget: int
    fingerprint: str
    source_pairs: int
    nll: float
    initialized_nll: float
    bits: float
    calibration: float
    per_cell_nll: tuple[float, ...]
    initialized_per_cell_nll: tuple[float, ...]
    teacher_error: float | None
    parameter_delta: tuple[float, ...]
    cell_ids: tuple[str, ...]
    cone_positions: tuple[tuple[float, ...], ...]
    lag_order: str
    low_kernel: np.ndarray
    high_kernel: np.ndarray
    initialized_low_kernel: np.ndarray
    initialized_high_kernel: np.ndarray
    matched_initialization: bool
    shuffle_contract: str
    effective_type_labels: tuple[str, ...]
    initial_effective_parameters: tuple[float, ...]
    observed_type_labels: tuple[str, ...]
    cell_polarities: tuple[int, ...]


RunGrid: TypeAlias = dict[tuple[int, int], dict[Variant, RunInput]]


__all__ = [
    "BASE_VARIANTS",
    "HISTORY_CONTRACTS",
    "JsonMap",
    "JsonValue",
    "MIN_SEEDS",
    "MIN_SOURCE_PAIRS",
    "OverallStatus",
    "REFERENCE_VARIANT",
    "SHUFFLE_VARIANTS",
    "RunGrid",
    "RunInput",
    "TypePriorComparisonError",
    "ValueStatus",
    "Variant",
]
