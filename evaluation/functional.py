from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class FunctionalComparisonError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FunctionalResponse:
    chirp_frequency_hz: torch.Tensor
    chirp_amplitude: torch.Tensor
    contrast: torch.Tensor
    contrast_response: torch.Tensor
    grating_spatial_frequency_cpd: torch.Tensor
    grating_response: torch.Tensor


@dataclass(frozen=True, slots=True)
class FunctionalSummary:
    chirp_peak_hz: float
    contrast_gain: float
    grating_preference_cpd: float


@dataclass(frozen=True, slots=True)
class FunctionalAgreementTolerance:
    max_frequency_octave_error: float
    max_contrast_gain_relative_error: float

    def __post_init__(self) -> None:
        values = (
            self.max_frequency_octave_error,
            self.max_contrast_gain_relative_error,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise FunctionalComparisonError(
                "Functional agreement tolerances must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class FunctionalAgreement:
    chirp_peak_octave_error: float
    contrast_gain_relative_error: float
    grating_preference_octave_error: float
    passed: bool


def summarize_functional_response(
    response: FunctionalResponse,
) -> FunctionalSummary:
    pairs = (
        (response.chirp_frequency_hz, response.chirp_amplitude),
        (response.contrast, response.contrast_response),
        (response.grating_spatial_frequency_cpd, response.grating_response),
    )
    if any(
        x.ndim != 1
        or y.shape != x.shape
        or x.numel() < 2
        or not torch.isfinite(x).all()
        or not torch.isfinite(y).all()
        for x, y in pairs
    ):
        raise FunctionalComparisonError("Functional response curves are invalid")
    contrast_design = torch.stack(
        (response.contrast, torch.ones_like(response.contrast)),
        dim=1,
    )
    contrast_fit = torch.linalg.lstsq(
        contrast_design,
        response.contrast_response,
    ).solution
    return FunctionalSummary(
        chirp_peak_hz=float(
            response.chirp_frequency_hz[response.chirp_amplitude.argmax()]
        ),
        contrast_gain=float(contrast_fit[0]),
        grating_preference_cpd=float(
            response.grating_spatial_frequency_cpd[
                response.grating_response.argmax()
            ]
        ),
    )


def compare_functional_summaries(
    model: FunctionalSummary,
    human_reference: FunctionalSummary,
    tolerance: FunctionalAgreementTolerance,
) -> FunctionalAgreement:
    values = (
        model.chirp_peak_hz,
        model.contrast_gain,
        model.grating_preference_cpd,
        human_reference.chirp_peak_hz,
        human_reference.contrast_gain,
        human_reference.grating_preference_cpd,
    )
    if not all(math.isfinite(value) for value in values):
        raise FunctionalComparisonError("Functional summaries must be finite")
    positive_frequencies = (
        model.chirp_peak_hz,
        model.grating_preference_cpd,
        human_reference.chirp_peak_hz,
        human_reference.grating_preference_cpd,
    )
    if min(positive_frequencies) <= 0 or human_reference.contrast_gain == 0:
        raise FunctionalComparisonError(
            "Reference frequencies and contrast gain must be non-zero"
        )
    chirp_error = abs(
        math.log2(model.chirp_peak_hz / human_reference.chirp_peak_hz)
    )
    grating_error = abs(
        math.log2(
            model.grating_preference_cpd
            / human_reference.grating_preference_cpd
        )
    )
    contrast_error = abs(
        model.contrast_gain - human_reference.contrast_gain
    ) / abs(human_reference.contrast_gain)
    return FunctionalAgreement(
        chirp_peak_octave_error=chirp_error,
        contrast_gain_relative_error=contrast_error,
        grating_preference_octave_error=grating_error,
        passed=(
            chirp_error <= tolerance.max_frequency_octave_error
            and grating_error <= tolerance.max_frequency_octave_error
            and contrast_error <= tolerance.max_contrast_gain_relative_error
        ),
    )
