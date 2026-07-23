from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from evaluation.decoder_diagnostics import (
    decode_with_fit,
    fit_global_decoder,
)


class GlobalProbeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GlobalProbeInputs:
    train_readout: torch.Tensor
    train_target: torch.Tensor
    train_source_ids: Sequence[str]
    validation_readout: torch.Tensor
    validation_target: torch.Tensor
    validation_source_ids: Sequence[str]


class RepresentationExamples(Protocol):
    rates: torch.Tensor
    generator_potential: torch.Tensor
    target: torch.Tensor
    source_ids: Sequence[str]


@dataclass(frozen=True, slots=True)
class GlobalReadoutGeometry:
    spatial_weights: torch.Tensor
    gain_max: float


@dataclass(frozen=True, slots=True)
class GlobalSourceMSE:
    source_id: str
    mse: float


@dataclass(frozen=True, slots=True)
class GlobalProbeResult:
    source_cv_mse: float
    validation_mse: float
    validation_source_mse: tuple[GlobalSourceMSE, ...]


@dataclass(frozen=True, slots=True)
class GlobalProbePair:
    rate: GlobalProbeResult
    generator: GlobalProbeResult


@torch.no_grad()
def fit_global_probe(
    inputs: GlobalProbeInputs,
    geometry: GlobalReadoutGeometry,
) -> GlobalProbeResult:
    _validate_inputs(inputs)
    fit = fit_global_decoder(
        inputs.train_readout,
        inputs.train_target,
        geometry.spatial_weights,
        gain_max=geometry.gain_max,
    )
    validation_prediction = decode_with_fit(
        inputs.validation_readout,
        geometry.spatial_weights,
        fit.unit_gain,
        fit.cone_bias,
    )
    return GlobalProbeResult(
        source_cv_mse=_source_cross_validated_mse(inputs, geometry),
        validation_mse=float(
            (validation_prediction - inputs.validation_target).square().mean()
        ),
        validation_source_mse=_source_mse(
            validation_prediction,
            inputs.validation_target,
            inputs.validation_source_ids,
        ),
    )


def fit_global_probe_pair(
    train: RepresentationExamples,
    validation: RepresentationExamples,
    geometry: GlobalReadoutGeometry,
) -> GlobalProbePair:
    return GlobalProbePair(
        rate=fit_global_probe(
            GlobalProbeInputs(
                train_readout=train.rates,
                train_target=train.target,
                train_source_ids=train.source_ids,
                validation_readout=validation.rates,
                validation_target=validation.target,
                validation_source_ids=validation.source_ids,
            ),
            geometry,
        ),
        generator=fit_global_probe(
            GlobalProbeInputs(
                train_readout=train.generator_potential,
                train_target=train.target,
                train_source_ids=train.source_ids,
                validation_readout=validation.generator_potential,
                validation_target=validation.target,
                validation_source_ids=validation.source_ids,
            ),
            geometry,
        ),
    )


def _source_cross_validated_mse(
    inputs: GlobalProbeInputs,
    geometry: GlobalReadoutGeometry,
) -> float:
    unique_sources = tuple(dict.fromkeys(inputs.train_source_ids))
    if len(unique_sources) < 2:
        return fit_global_decoder(
            inputs.train_readout,
            inputs.train_target,
            geometry.spatial_weights,
            gain_max=geometry.gain_max,
        ).train_mse
    squared_error = 0.0
    element_count = 0
    for source_id in unique_sources:
        held_out = torch.tensor(
            [value == source_id for value in inputs.train_source_ids],
            device=inputs.train_readout.device,
        )
        fit = fit_global_decoder(
            inputs.train_readout[~held_out],
            inputs.train_target[~held_out],
            geometry.spatial_weights,
            gain_max=geometry.gain_max,
        )
        prediction = decode_with_fit(
            inputs.train_readout[held_out],
            geometry.spatial_weights,
            fit.unit_gain,
            fit.cone_bias,
        )
        squared_error += float(
            (prediction - inputs.train_target[held_out]).square().sum()
        )
        element_count += prediction.numel()
    return squared_error / element_count


def _source_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_ids: Sequence[str],
) -> tuple[GlobalSourceMSE, ...]:
    rows: list[GlobalSourceMSE] = []
    for source_id in dict.fromkeys(source_ids):
        selected = torch.tensor(
            [value == source_id for value in source_ids],
            device=target.device,
        )
        rows.append(
            GlobalSourceMSE(
                source_id=source_id,
                mse=float(
                    (prediction[selected] - target[selected]).square().mean()
                ),
            )
        )
    return tuple(rows)


def _validate_inputs(inputs: GlobalProbeInputs) -> None:
    if len(inputs.train_source_ids) != inputs.train_readout.shape[0]:
        raise GlobalProbeError("Training source IDs do not match readout rows")
    if (
        len(inputs.validation_source_ids)
        != inputs.validation_readout.shape[0]
    ):
        raise GlobalProbeError(
            "Validation source IDs do not match readout rows"
        )


__all__ = [
    "GlobalProbeError",
    "GlobalProbeInputs",
    "GlobalProbePair",
    "GlobalProbeResult",
    "GlobalReadoutGeometry",
    "GlobalSourceMSE",
    "RepresentationExamples",
    "fit_global_probe",
    "fit_global_probe_pair",
]
