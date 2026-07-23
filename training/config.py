from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DataConfig:
    train_glob: str
    validation_glob: str
    sequence_steps: int
    noise_std_min: float
    noise_std_max: float
    noise_transition_probability: float
    context_transition_probability: float
    context_transition_latest_step: int
    context_gain_min: float
    context_gain_max: float


@dataclass(frozen=True, slots=True)
class ModelConfig:
    units_per_center: int
    support_radius_spacing_multiplier: float
    sigma_min_spacing_multiplier: float
    sigma_initial_spacing_multiplier: float
    sigma_max_spacing_multiplier: float
    readout_rate_tau_ms: float
    max_tau_ms: float
    surrogate_slope: float
    decoder_gain_max: float
    adaptation_gain_max: float
    amacrine_gain_max: float
    subunit_gain_max: float
    debug_checks: bool


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    burn_in_steps: int
    differentiable_steps: int
    context_only_steps: int
    supervised_steps: int
    checkpoint_block_steps: int
    batch_size: int
    gradient_clip_norm: float
    max_optimizer_steps: int
    reconstruction_bootstrap_steps: int
    budget_ramp_end_step: int
    core_lr: float
    decoder_lr: float
    validation_interval_steps: int


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    energy_budget_ratio: float
    rho_energy: float
    dual_lr: float
    dual_max: float
    wiring_weight: float
    variance_weight: float
    phenotype_repulsion_weight: float
    homeostasis_weight: float
    variance_floor: float
    phenotype_temperature: float
    homeostasis_rate_min: float


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    dynamic_rf_max_sources: int
    dynamic_rf_units_per_polarity: int
    dynamic_rf_lag_steps: int
    recovery_delays_ms: tuple[int, ...]
    minimum_representation_skill: float
    maximum_energy_budget_ratio: float
    local_linear_baseline: str
    dynamic_rf_fd_max_relative_error: float
    dynamic_rf_kernel_norm_min: float
    dynamic_rf_reset_error_max: float
    dynamic_rf_min_valid_sources: int
    dynamic_rf_min_valid_records_per_source: int
    dynamic_rf_min_valid_record_fraction_per_source: float
    dynamic_rf_shape_distance_min: float
    dynamic_rf_gain_log_shift_min: float
    dynamic_rf_recovery_fraction_max: float
    dynamic_rf_bootstrap_iterations: int
    rgc_min_cluster_fraction: float
    rgc_min_silhouette: float
    rgc_min_relative_radius_difference: float
    rgc_min_sustained_difference: float
    rgc_min_flicker_difference: float
    rgc_min_trained_to_initial_separation_ratio: float
    rgc_min_absolute_separation_gain: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    objective: ObjectiveConfig
    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigurationError("seed must be non-negative")
        if self.data.sequence_steps != (
            self.training.burn_in_steps + self.training.differentiable_steps
        ):
            raise ConfigurationError("sequence_steps must equal burn-in plus differentiable steps")
        if self.training.context_only_steps + self.training.supervised_steps != self.training.differentiable_steps:
            raise ConfigurationError("context-only plus supervised steps must equal differentiable steps")
        if self.training.differentiable_steps % self.training.checkpoint_block_steps:
            raise ConfigurationError("differentiable steps must divide into checkpoint blocks")
        probabilities = (
            self.data.noise_transition_probability,
            self.data.context_transition_probability,
        )
        if not all(0 <= value <= 1 for value in probabilities):
            raise ConfigurationError("transition probabilities must lie inside [0,1]")
        supervised_onset = (
            self.training.burn_in_steps + self.training.context_only_steps
        )
        earliest_transition = max(1, self.data.sequence_steps // 3)
        transition_width = max(2, self.data.sequence_steps // 64)
        if not (
            earliest_transition <= self.data.context_transition_latest_step
            and self.data.context_transition_latest_step + transition_width
            < supervised_onset
        ):
            raise ConfigurationError(
                "context transitions must finish before supervised onset"
            )
        if not 0 < self.data.noise_std_min <= self.data.noise_std_max:
            raise ConfigurationError("noise bounds are invalid")
        if not 0 < self.data.context_gain_min <= self.data.context_gain_max:
            raise ConfigurationError("context gain bounds are invalid")
        if not 0 < self.objective.energy_budget_ratio <= self.evaluation.maximum_energy_budget_ratio:
            raise ConfigurationError("energy budget ratios are inconsistent")
        if self.evaluation.local_linear_baseline != "disabled":
            raise ConfigurationError("local_linear_baseline currently supports only disabled")
        positive_values = (
            self.model.units_per_center,
            self.model.decoder_gain_max,
            self.model.adaptation_gain_max,
            self.model.amacrine_gain_max,
            self.model.subunit_gain_max,
            self.training.batch_size,
            self.training.gradient_clip_norm,
            self.training.max_optimizer_steps,
            self.training.validation_interval_steps,
            self.training.core_lr,
            self.training.decoder_lr,
            self.objective.rho_energy,
            self.objective.dual_lr,
            self.objective.dual_max,
            self.objective.wiring_weight,
            self.objective.variance_weight,
            self.objective.phenotype_repulsion_weight,
            self.objective.homeostasis_weight,
            self.objective.phenotype_temperature,
            self.evaluation.dynamic_rf_max_sources,
            self.evaluation.dynamic_rf_units_per_polarity,
            self.evaluation.dynamic_rf_lag_steps,
            self.evaluation.dynamic_rf_fd_max_relative_error,
            self.evaluation.dynamic_rf_kernel_norm_min,
            self.evaluation.dynamic_rf_reset_error_max,
            self.evaluation.dynamic_rf_min_valid_sources,
            self.evaluation.dynamic_rf_min_valid_records_per_source,
            self.evaluation.dynamic_rf_shape_distance_min,
            self.evaluation.dynamic_rf_gain_log_shift_min,
            self.evaluation.dynamic_rf_recovery_fraction_max,
            self.evaluation.dynamic_rf_bootstrap_iterations,
            self.evaluation.rgc_min_cluster_fraction,
            self.evaluation.rgc_min_silhouette,
            self.evaluation.rgc_min_relative_radius_difference,
            self.evaluation.rgc_min_sustained_difference,
            self.evaluation.rgc_min_flicker_difference,
            self.evaluation.rgc_min_trained_to_initial_separation_ratio,
            self.evaluation.rgc_min_absolute_separation_gain,
        )
        if not all(math.isfinite(float(value)) and value > 0 for value in positive_values):
            raise ConfigurationError("positive configuration values are invalid")
        valid_fraction = self.evaluation.dynamic_rf_min_valid_record_fraction_per_source
        if not 0 < valid_fraction <= 1:
            raise ConfigurationError(
                "dynamic RF minimum valid record fraction must lie inside (0,1]"
            )

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


_ConfigType = TypeVar("_ConfigType")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    config = _from_mapping(ExperimentConfig, raw, "config")
    if not isinstance(config, ExperimentConfig):
        raise ConfigurationError("Failed to materialize experiment configuration")
    return config


def _from_mapping(cls: type[_ConfigType], raw: dict[str, Any], path: str) -> _ConfigType:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path} must be a mapping")
    expected = {field.name for field in fields(cls)}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown:
        raise ConfigurationError(f"{path} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ConfigurationError(f"{path} is missing keys: {sorted(missing)}")
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for field in fields(cls):
        expected_type = hints[field.name]
        value = raw[field.name]
        if is_dataclass(expected_type):
            value = _from_mapping(expected_type, value, f"{path}.{field.name}")
        elif expected_type == tuple[int, ...]:
            if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
                raise ConfigurationError(f"{path}.{field.name} must be a list of integers")
            value = tuple(value)
        values[field.name] = value
    return cls(**values)


__all__ = [
    "ConfigurationError",
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ObjectiveConfig",
    "TrainingConfig",
    "load_config",
]
