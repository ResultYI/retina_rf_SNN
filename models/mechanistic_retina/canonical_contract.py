from __future__ import annotations

from models.mechanistic_retina.causal_contract import CANONICAL_CAUSAL_CONTRACT
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticConfigError,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.spatial_contract import CANONICAL_SPATIAL_CONTRACT


def validate_canonical_config(config: MechanisticRetinaConfig) -> None:
    if config.architecture_mode != ArchitectureMode.MECHANISM_IDENTIFIABLE:
        raise MechanisticConfigError(
            "Canonical V1 requires mechanism_identifiable architecture mode"
        )
    if config.causal_contract != CANONICAL_CAUSAL_CONTRACT:
        raise MechanisticConfigError("Canonical V1 requires the shared-BC causal contract")
    if config.spatial_contract != CANONICAL_SPATIAL_CONTRACT:
        raise MechanisticConfigError("Canonical V1 requires overlapping full-disk spatial support")
