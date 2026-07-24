from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import yaml


YamlValue: TypeAlias = (
    str | int | float | bool | None | list["YamlValue"] | dict[str, "YamlValue"]
)


class TypePriorConfigurationError(ValueError):
    pass


_PARAMETERS = (
    "spatial_sigma",
    "sustained_mix",
    "membrane_tau_ms",
    "adaptation_tau_ms",
    "adaptation_gain",
    "amacrine_gain",
    "threshold",
    "subunit_tau_ms",
    "subunit_gain",
)


@dataclass(frozen=True, slots=True)
class ParameterPrior:
    mean: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        values = (self.mean, self.lower, self.upper)
        if not all(math.isfinite(value) for value in values):
            raise TypePriorConfigurationError("Type prior values must be finite")
        if not self.lower < self.mean < self.upper:
            raise TypePriorConfigurationError(
                "Type prior mean must lie strictly inside its bounds"
            )


@dataclass(frozen=True, slots=True)
class RGCTypePrior:
    type_id: str
    spatial_sigma: ParameterPrior
    sustained_mix: ParameterPrior
    membrane_tau_ms: ParameterPrior
    adaptation_tau_ms: ParameterPrior
    adaptation_gain: ParameterPrior
    amacrine_gain: ParameterPrior
    threshold: ParameterPrior
    subunit_tau_ms: ParameterPrior
    subunit_gain: ParameterPrior

    def parameter(self, name: str) -> ParameterPrior:
        if name not in _PARAMETERS:
            raise TypePriorConfigurationError(f"Unknown RGC prior parameter: {name}")
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class RGCTypePriors:
    cell_residual_scale: float
    cell_residual_weight: float
    type_prior_weight: float
    types: tuple[RGCTypePrior, ...]

    @property
    def type_ids(self) -> tuple[str, ...]:
        return tuple(prior.type_id for prior in self.types)

    def for_type(self, type_id: str) -> RGCTypePrior:
        for prior in self.types:
            if prior.type_id == type_id:
                return prior
        raise TypePriorConfigurationError(f"No type prior configured for {type_id!r}")


def load_type_priors(
    path: str | Path,
    *,
    required_type_ids: tuple[str, ...],
) -> RGCTypePriors:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypePriorConfigurationError("Type prior root must be a mapping")
    expected = {
        "cell_residual_scale",
        "cell_residual_weight",
        "type_prior_weight",
        "types",
    }
    _require_keys(raw, expected, "type priors")
    raw_types = raw["types"]
    if not isinstance(raw_types, dict) or not raw_types:
        raise TypePriorConfigurationError("types must be a non-empty mapping")
    required = set(required_type_ids)
    configured = set(raw_types)
    if configured != required:
        raise TypePriorConfigurationError(
            f"Type prior coverage mismatch: required={sorted(required)}, "
            f"configured={sorted(configured)}"
        )
    priors = tuple(
        _parse_type(type_id, raw_types[type_id]) for type_id in sorted(configured)
    )
    residual_scale = float(raw["cell_residual_scale"])
    residual_weight = float(raw["cell_residual_weight"])
    type_prior_weight = float(raw["type_prior_weight"])
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise TypePriorConfigurationError("cell_residual_scale must be positive")
    if not math.isfinite(residual_weight) or residual_weight < 0:
        raise TypePriorConfigurationError("cell_residual_weight must be non-negative")
    if not math.isfinite(type_prior_weight) or type_prior_weight < 0:
        raise TypePriorConfigurationError("type_prior_weight must be non-negative")
    return RGCTypePriors(
        residual_scale,
        residual_weight,
        type_prior_weight,
        priors,
    )


def _parse_type(type_id: str, raw: YamlValue) -> RGCTypePrior:
    if not isinstance(type_id, str) or not type_id:
        raise TypePriorConfigurationError("type_id must be a non-empty string")
    if not isinstance(raw, dict):
        raise TypePriorConfigurationError(f"types.{type_id} must be a mapping")
    _require_keys(raw, set(_PARAMETERS), f"types.{type_id}")
    parameters = {
        name: _parse_parameter(raw[name], f"types.{type_id}.{name}")
        for name in _PARAMETERS
    }
    return RGCTypePrior(type_id=type_id, **parameters)


def _parse_parameter(raw: YamlValue, path: str) -> ParameterPrior:
    if not isinstance(raw, dict):
        raise TypePriorConfigurationError(f"{path} must be a mapping")
    _require_keys(raw, {"mean", "lower", "upper"}, path)
    return ParameterPrior(
        mean=float(raw["mean"]),
        lower=float(raw["lower"]),
        upper=float(raw["upper"]),
    )


def _require_keys(
    raw: dict[str, YamlValue],
    expected: set[str],
    path: str,
) -> None:
    keys = set(raw)
    if keys != expected:
        raise TypePriorConfigurationError(
            f"{path} keys mismatch: missing={sorted(expected - keys)}, "
            f"unknown={sorted(keys - expected)}"
        )


__all__ = [
    "ParameterPrior",
    "RGCTypePrior",
    "RGCTypePriors",
    "TypePriorConfigurationError",
    "load_type_priors",
]
