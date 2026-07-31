from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum, unique
from typing import assert_never


class InputIdentityError(ValueError):
    pass


@unique
class DatasetKind(StrEnum):
    LEGACY_UNSPECIFIED = "legacy_unspecified"
    REAL_RECORDING = "real_recording"
    SYNTHETIC_METHOD_VALIDATION = "synthetic_method_validation"


@dataclass(frozen=True, slots=True)
class InputIdentity:
    dataset_kind: DatasetKind
    species: str
    optics_species: str
    mosaic_species: str
    photoreceptor_mode: str
    chromatic_mode: str
    light_level: str
    mean_luminance_cd_m2: float
    cone_types: tuple[int, ...]
    response_units: str
    mosaic_id: str
    mosaic_fingerprint: str
    stimulus_source_fingerprints: tuple[str, ...]
    generator_name: str
    generator_revision: str
    cone_bin_reference: str
    spike_bin_reference: str
    stimulus_to_spike_offset_bins: int

    def __post_init__(self) -> None:
        text_values = (
            self.species,
            self.optics_species,
            self.mosaic_species,
            self.photoreceptor_mode,
            self.chromatic_mode,
            self.light_level,
            self.response_units,
            self.mosaic_id,
            self.mosaic_fingerprint,
            self.generator_name,
            self.generator_revision,
            self.cone_bin_reference,
            self.spike_bin_reference,
        )
        if not all(value.strip() for value in text_values):
            raise InputIdentityError("Input identity text fields must be non-empty")
        if (
            not math.isfinite(self.mean_luminance_cd_m2)
            or self.mean_luminance_cd_m2 < 0
        ):
            raise InputIdentityError("mean_luminance_cd_m2 must be finite and non-negative")
        if any(not value.strip() for value in self.stimulus_source_fingerprints):
            raise InputIdentityError("Stimulus source fingerprints must be non-empty")

    def compatibility_key(self) -> tuple[str, ...]:
        return (
            self.dataset_kind.value,
            self.species,
            self.optics_species,
            self.mosaic_species,
            self.photoreceptor_mode,
            self.chromatic_mode,
            self.light_level,
            repr(self.mean_luminance_cd_m2),
            ",".join(str(value) for value in self.cone_types),
            self.response_units,
            self.mosaic_id,
            self.mosaic_fingerprint,
            self.generator_name,
            self.generator_revision,
            self.cone_bin_reference,
            self.spike_bin_reference,
            str(self.stimulus_to_spike_offset_bins),
        )

    def with_sources(
        self,
        fingerprints: tuple[str, ...],
        *,
        generator_name: str | None = None,
        generator_revision: str | None = None,
    ) -> InputIdentity:
        return replace(
            self,
            stimulus_source_fingerprints=fingerprints,
            generator_name=generator_name or self.generator_name,
            generator_revision=generator_revision or self.generator_revision,
        )


def legacy_input_identity() -> InputIdentity:
    return InputIdentity(
        dataset_kind=DatasetKind.LEGACY_UNSPECIFIED,
        species="unknown",
        optics_species="unknown",
        mosaic_species="unknown",
        photoreceptor_mode="unknown",
        chromatic_mode="unknown",
        light_level="unknown",
        mean_luminance_cd_m2=0.0,
        cone_types=(),
        response_units="unknown",
        mosaic_id="unknown",
        mosaic_fingerprint="unknown",
        stimulus_source_fingerprints=(),
        generator_name="legacy",
        generator_revision="unknown",
        cone_bin_reference="unknown",
        spike_bin_reference="unknown",
        stimulus_to_spike_offset_bins=0,
    )


def synthetic_input_identity(
    cone_count: int,
    source_fingerprints: tuple[str, ...],
) -> InputIdentity:
    return InputIdentity(
        dataset_kind=DatasetKind.SYNTHETIC_METHOD_VALIDATION,
        species="synthetic",
        optics_species="synthetic",
        mosaic_species="synthetic",
        photoreceptor_mode="cone_only",
        chromatic_mode="achromatic",
        light_level="photopic",
        mean_luminance_cd_m2=0.0,
        cone_types=(0,) * cone_count,
        response_units="normalized_synthetic_input",
        mosaic_id="synthetic-memory",
        mosaic_fingerprint="synthetic-memory",
        stimulus_source_fingerprints=source_fingerprints,
        generator_name="point_process_teacher",
        generator_revision="1",
        cone_bin_reference="interval_end",
        spike_bin_reference="interval_end",
        stimulus_to_spike_offset_bins=0,
    )


def validate_experiment_input(identity: InputIdentity, dt_ms: float) -> None:
    if not math.isclose(dt_ms, 5.0, rel_tol=0.0, abs_tol=1e-6):
        raise InputIdentityError("Canonical response fitting requires 5 ms bins")
    signal_contract = (
        identity.photoreceptor_mode == "cone_only"
        and identity.chromatic_mode == "achromatic"
        and identity.light_level == "photopic"
        and identity.cone_bin_reference == "interval_end"
        and identity.spike_bin_reference == "interval_end"
        and identity.stimulus_to_spike_offset_bins == 0
    )
    match identity.dataset_kind:
        case DatasetKind.SYNTHETIC_METHOD_VALIDATION:
            if not signal_contract:
                raise InputIdentityError(
                    "Synthetic canonical input must be cone-only, achromatic, "
                    "photopic, and interval-end aligned"
                )
            return
        case DatasetKind.REAL_RECORDING:
            required = (
                signal_contract
                and identity.species == "macaque"
                and identity.optics_species == "macaque"
                and identity.mosaic_species == "macaque"
            )
            if not required:
                raise InputIdentityError(
                    "Real canonical input must be macaque, cone-only, achromatic, "
                    "photopic, and interval-end aligned"
                )
        case DatasetKind.LEGACY_UNSPECIFIED:
            raise InputIdentityError(
                "Legacy response input lacks species and mosaic provenance"
            )
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "DatasetKind",
    "InputIdentity",
    "InputIdentityError",
    "legacy_input_identity",
    "synthetic_input_identity",
    "validate_experiment_input",
]
