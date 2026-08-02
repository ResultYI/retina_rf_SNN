from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import math
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, TypeVar, get_type_hints

import yaml


class ResponseConfigurationError(ValueError):
    pass


ParameterSharingMode: TypeAlias = Literal[
    "type_aware",
    "type_blind",
    "cell_only",
    "shuffled_type",
    "balanced_shuffled_type",
]
_PARAMETER_SHARING_MODES: Final = frozenset(
    (
        "type_aware",
        "type_blind",
        "cell_only",
        "shuffled_type",
        "balanced_shuffled_type",
    )
)


@dataclass(frozen=True, slots=True)
class ResponseDataConfig:
    train_glob: str
    validation_glob: str
    test_glob: str
    sequence_steps: int


@dataclass(frozen=True, slots=True)
class ResponseModelConfig:
    type_prior_path: str
    support_radius_degs: float
    readout_rate_tau_ms: float
    surrogate_slope: float
    parameter_sharing_mode: ParameterSharingMode = "type_aware"
    matched_initialization: bool = False
    enable_response_bias: bool = False
    enable_synaptic_gain: bool = False
    synaptic_gain_min: float = 0.1
    synaptic_gain_max: float = 4.0
    synaptic_gain_init: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parameter_sharing_mode, str)
            or self.parameter_sharing_mode not in _PARAMETER_SHARING_MODES
        ):
            raise ResponseConfigurationError(
                "parameter_sharing_mode must be one of "
                f"{sorted(_PARAMETER_SHARING_MODES)}"
            )
        if not isinstance(self.matched_initialization, bool):
            raise ResponseConfigurationError(
                "matched_initialization must be a boolean"
            )
        if not isinstance(self.enable_response_bias, bool):
            raise ResponseConfigurationError("enable_response_bias must be a boolean")
        if not isinstance(self.enable_synaptic_gain, bool):
            raise ResponseConfigurationError("enable_synaptic_gain must be a boolean")
        gains = (
            self.synaptic_gain_min,
            self.synaptic_gain_max,
            self.synaptic_gain_init,
        )
        if not all(math.isfinite(value) for value in gains):
            raise ResponseConfigurationError("synaptic gain bounds must be finite")
        if not self.synaptic_gain_min < self.synaptic_gain_init < self.synaptic_gain_max:
            raise ResponseConfigurationError(
                "synaptic_gain_init must lie inside synaptic gain bounds"
            )


@dataclass(frozen=True, slots=True)
class ResponseTrainingConfig:
    burn_in_steps: int
    differentiable_steps: int
    checkpoint_block_steps: int
    batch_size: int
    max_optimizer_steps: int
    learning_rate: float
    gradient_clip_norm: float
    validation_interval_steps: int
    learn_cell_residuals: bool = True
    supervised_tail_steps: int | None = None
    response_bias_lr: float = 0.01
    rgc_lr: float = 0.001
    stage0_calibration_enabled: bool = False
    freeze_threshold: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("response_bias_lr", self.response_bias_lr),
            ("rgc_lr", self.rgc_lr),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ResponseConfigurationError(
                    f"{name} must be finite and positive"
                )
        if self.supervised_tail_steps is not None and not (
            1 <= self.supervised_tail_steps <= self.differentiable_steps
        ):
            raise ResponseConfigurationError(
                "supervised tail must fit within the differentiable window"
            )
        if not isinstance(self.stage0_calibration_enabled, bool):
            raise ResponseConfigurationError(
                "stage0_calibration_enabled must be a boolean"
            )
        if not isinstance(self.freeze_threshold, bool):
            raise ResponseConfigurationError("freeze_threshold must be a boolean")

    @property
    def supervision_slice(self) -> slice:
        if self.supervised_tail_steps is None:
            return slice(None)
        return slice(-self.supervised_tail_steps, None)


@dataclass(frozen=True, slots=True)
class ResponseEvaluationConfig:
    rf_lag_steps: int
    recovery_delays_ms: tuple[int, ...]
    rf_finite_difference_checks: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.rf_finite_difference_checks, bool):
            raise ResponseConfigurationError(
                "rf_finite_difference_checks must be a boolean"
            )

    @property
    def finite_difference_tolerance(self) -> float | None:
        if self.rf_finite_difference_checks:
            return 0.05
        return None


@dataclass(frozen=True, slots=True)
class ResponseExperimentConfig:
    seed: int
    data: ResponseDataConfig
    model: ResponseModelConfig
    training: ResponseTrainingConfig
    evaluation: ResponseEvaluationConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ResponseConfigurationError("seed must be non-negative")
        if self.data.sequence_steps != (
            self.training.burn_in_steps + self.training.differentiable_steps
        ):
            raise ResponseConfigurationError(
                "sequence_steps must equal burn-in plus differentiable steps"
            )
        if (
            self.training.differentiable_steps
            % self.training.checkpoint_block_steps
        ):
            raise ResponseConfigurationError(
                "differentiable steps must divide into checkpoint blocks"
            )
        positive = (
            self.data.sequence_steps,
            self.model.support_radius_degs,
            self.model.readout_rate_tau_ms,
            self.model.surrogate_slope,
            self.model.synaptic_gain_min,
            self.model.synaptic_gain_max,
            self.model.synaptic_gain_init,
            self.training.burn_in_steps,
            self.training.differentiable_steps,
            self.training.batch_size,
            self.training.max_optimizer_steps,
            self.training.learning_rate,
            self.training.response_bias_lr,
            self.training.rgc_lr,
            self.training.gradient_clip_norm,
            self.training.validation_interval_steps,
            self.evaluation.rf_lag_steps,
        )
        if not all(value > 0 for value in positive):
            raise ResponseConfigurationError(
                "Positive response configuration values are invalid"
            )
        if not isinstance(self.training.learn_cell_residuals, bool):
            raise ResponseConfigurationError(
                "learn_cell_residuals must be a boolean"
            )
        if (
            self.training.stage0_calibration_enabled
            and not self.model.enable_response_bias
        ):
            raise ResponseConfigurationError(
                "stage0 calibration requires enabled response bias"
            )


_ConfigType = TypeVar("_ConfigType")


def load_response_config(path: str | Path) -> ResponseExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ResponseConfigurationError("Configuration root must be a mapping")
    return _materialize(ResponseExperimentConfig, raw, "config")


def _materialize(
    cls: type[_ConfigType],
    raw: dict[str, Any],
    path: str,
) -> _ConfigType:
    expected = {field.name for field in fields(cls)}
    keys = set(raw)
    required = {
        field.name
        for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING
    }
    if not required.issubset(keys) or not keys.issubset(expected):
        raise ResponseConfigurationError(
            f"{path} keys mismatch: missing={sorted(required - keys)}, "
            f"unknown={sorted(keys - expected)}"
        )
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in raw:
            continue
        expected_type = hints[field.name]
        value = raw[field.name]
        if hasattr(expected_type, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ResponseConfigurationError(
                    f"{path}.{field.name} must be a mapping"
                )
            value = _materialize(expected_type, value, f"{path}.{field.name}")
        elif expected_type == tuple[int, ...]:
            if not isinstance(value, list) or not all(
                isinstance(item, int) for item in value
            ):
                raise ResponseConfigurationError(
                    f"{path}.{field.name} must be a list of integers"
                )
            value = tuple(value)
        values[field.name] = value
    return cls(**values)


__all__ = [
    "ResponseConfigurationError",
    "ResponseDataConfig",
    "ResponseEvaluationConfig",
    "ResponseExperimentConfig",
    "ResponseModelConfig",
    "ResponseTrainingConfig",
    "ParameterSharingMode",
    "load_response_config",
]
