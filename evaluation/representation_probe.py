from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from evaluation.decoder_diagnostics import (
    DecoderFit,
    decode_with_fit,
    fit_regularized_tied_decoder_probe,
)


@dataclass(frozen=True, slots=True)
class ProbePrediction:
    fit: DecoderFit
    validation_prediction: torch.Tensor
    source_cv_mse: float


def fit_probe_prediction(
    train_readout: torch.Tensor,
    train_target: torch.Tensor,
    train_source_ids: Sequence[str],
    validation_readout: torch.Tensor,
    spatial_weights: torch.Tensor,
    *,
    gain_max: float,
    prior_gain: torch.Tensor,
) -> ProbePrediction:
    fit = fit_regularized_tied_decoder_probe(
        train_readout,
        train_target,
        spatial_weights,
        gain_max=gain_max,
        prior_gain=prior_gain,
    )
    return ProbePrediction(
        fit=fit,
        validation_prediction=decode_with_fit(
            validation_readout,
            spatial_weights,
            fit.unit_gain,
            fit.cone_bias,
        ),
        source_cv_mse=source_cross_validated_probe_mse(
            train_readout,
            train_target,
            train_source_ids,
            spatial_weights,
            gain_max=gain_max,
            prior_gain=prior_gain,
        ),
    )


def source_cross_validated_probe_mse(
    readout: torch.Tensor,
    target: torch.Tensor,
    source_ids: Sequence[str],
    spatial_weights: torch.Tensor,
    *,
    gain_max: float,
    prior_gain: torch.Tensor,
) -> float:
    unique_sources = tuple(dict.fromkeys(source_ids))
    if len(unique_sources) < 2:
        return fit_regularized_tied_decoder_probe(
            readout,
            target,
            spatial_weights,
            gain_max=gain_max,
            prior_gain=prior_gain,
        ).train_mse
    squared_error = 0.0
    element_count = 0
    for source_id in unique_sources:
        held_out = torch.tensor(
            [value == source_id for value in source_ids],
            device=readout.device,
        )
        fit = fit_regularized_tied_decoder_probe(
            readout[~held_out],
            target[~held_out],
            spatial_weights,
            gain_max=gain_max,
            prior_gain=prior_gain,
        )
        prediction = decode_with_fit(
            readout[held_out],
            spatial_weights,
            fit.unit_gain,
            fit.cone_bias,
        )
        squared_error += float((prediction - target[held_out]).square().sum())
        element_count += prediction.numel()
    return squared_error / element_count


def source_mse_by_id(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_ids: Sequence[str],
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for source_id in dict.fromkeys(source_ids):
        selected = torch.tensor(
            [value == source_id for value in source_ids],
            device=target.device,
        )
        rows[source_id] = float(
            (prediction[selected] - target[selected]).square().mean()
        )
    return rows


__all__ = [
    "ProbePrediction",
    "fit_probe_prediction",
    "source_cross_validated_probe_mse",
    "source_mse_by_id",
]
