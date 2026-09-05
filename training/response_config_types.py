from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Literal, TypeAlias


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
    enable_direct_readout: bool = False
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
        for name, value in (
            ("enable_response_bias", self.enable_response_bias),
            ("enable_synaptic_gain", self.enable_synaptic_gain),
            ("enable_direct_readout", self.enable_direct_readout),
        ):
            if not isinstance(value, bool):
                raise ResponseConfigurationError(f"{name} must be a boolean")
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


__all__ = [
    "ParameterSharingMode",
    "ResponseConfigurationError",
    "ResponseDataConfig",
    "ResponseModelConfig",
]
