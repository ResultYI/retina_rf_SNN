#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

# ─── How to run ───
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
# 2. Run: uv run aggregation_core.py
# ──────────────────

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, assert_never

import numpy as np
from numpy.typing import NDArray


Group = Literal["MC_ON", "MC_OFF", "PC_ON", "PC_OFF"]
GROUPS: Final[tuple[Group, ...]] = ("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Stratum:
    indices: IntArray
    coefficient: float


@dataclass(frozen=True, slots=True)
class AggregationSpec:
    name: str
    strata: tuple[Stratum, ...]


def polarity(group: Group) -> Literal["ON", "OFF"]:
    match group:
        case "MC_ON" | "PC_ON":
            return "ON"
        case "MC_OFF" | "PC_OFF":
            return "OFF"
        case unreachable:
            assert_never(unreachable)


def point_estimate(values: FloatArray, specification: AggregationSpec) -> float:
    return sum(stratum.coefficient * float(values[stratum.indices].mean())
               for stratum in specification.strata)


def control_subtracted(values_by_width: FloatArray) -> FloatArray:
    return values_by_width - values_by_width[0]
