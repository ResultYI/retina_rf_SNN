from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
import torch
from torch.nn import functional as F

from baselines.center_surround_ln import CONTEXT_BINS, CenterSurroundLN, LNError
from baselines.spatial_contrast_official import (
    convolve_stimulus_with_kernels_for_sc, vectorized_softplus,
    vectorized_softplus_derivative,
)

FloatArray = NDArray[np.float64]


class ActivationCheck(TypedDict):
    center_activation_double_max_abs_error: float
    center_activation_native_max_abs_error: float


@dataclass(frozen=True, slots=True)
class CenterFilter:
    spatial: FloatArray
    temporal: FloatArray
    gaussian: FloatArray
    amplitude: float

    @classmethod
    def from_ln(cls, model: CenterSurroundLN) -> CenterFilter:
        with torch.no_grad():
            spatial = model.spatial_components()[0].numpy().astype(np.float64)
            temporal = model.temporal_kernels()[0].numpy().astype(np.float64)
            gaussian = model.gaussians()[0].numpy().astype(np.float64)
            amplitude = float(model.amplitudes()[0])
        if not all(np.isfinite(x).all() for x in (spatial, temporal, gaussian)):
            raise LNError("STOP: center filter contains nonfinite values")
        if spatial.shape != (17, 17) or temporal.shape != (60,) or spatial.sum() <= 0:
            raise LNError("STOP: center filter shape or amplitude cannot be recovered")
        if np.any(spatial < 0) or np.linalg.norm(temporal) == 0:
            raise LNError("STOP: invalid center filter")
        return cls(spatial, temporal, gaussian, amplitude)

    @property
    def official_temporal(self) -> FloatArray:
        return self.temporal * self.spatial.sum()


def features_for_sequences(
    center: CenterFilter, cones: torch.Tensor,
) -> tuple[FloatArray, ActivationCheck]:
    if cones.ndim != 3 or cones.shape[-1] != 289 or not torch.isfinite(cones).all():
        raise LNError("STOP: stimulus violates [sequence,time,289] contract")
    unique, inverse = np.unique(cones.numpy(), axis=0, return_inverse=True)
    time = cones.shape[1]
    stimulus = np.pad(unique.astype(np.float64).reshape(-1, time, 17, 17),
                      ((0, 0), (CONTEXT_BINS-1, 0), (0, 0), (0, 0)))
    imean, lsc = convolve_stimulus_with_kernels_for_sc(
        stimulus, center.spatial, center.official_temporal, len(stimulus), stimulus_smoothing=None,
    )
    errors = []
    for dtype in (torch.float64, torch.float32):
        x = torch.from_numpy(unique).to(dtype)
        spatial = torch.from_numpy(center.spatial).to(dtype).flatten()
        kernel = torch.from_numpy(center.temporal).to(dtype).flip(0)[None, None]
        expected = F.conv1d(F.pad((x @ spatial)[:, None], (CONTEXT_BINS-1, 0)), kernel)[:, 0].numpy()
        errors.append(float(np.max(np.abs(expected-imean))))
        tolerance = 1e-10 if dtype == torch.float64 else 1e-5
        np.testing.assert_allclose(imean, expected, atol=tolerance, rtol=tolerance,
                                   err_msg="STOP: center-only activation polarity/scale mismatch")
    features = np.stack((imean, lsc), axis=-1)[inverse]
    if not np.isfinite(features).all():
        raise LNError("STOP: SC features are nonfinite")
    return features, ActivationCheck(center_activation_double_max_abs_error=errors[0],
                                    center_activation_native_max_abs_error=errors[1])


@dataclass(frozen=True, slots=True)
class FittingData:
    standardized: FloatArray
    events: FloatArray
    mean: FloatArray
    std: FloatArray
    maximum_count: int

    @classmethod
    def from_arrays(
        cls, features: FloatArray, counts: NDArray[np.int64], mask: NDArray[np.bool_],
    ) -> FittingData:
        selected = features[mask[..., 0]]
        mean, std = selected.mean(axis=0), selected.std(axis=0)
        if not np.isfinite(selected).all() or np.any(std <= 0):
            raise LNError("STOP: fitting-only Z-score has a nonfinite or zero scale")
        observed = counts[mask]
        events = (observed > 0).astype(np.float64)
        if not 0 < events.sum() < events.size:
            raise LNError("STOP: fitting target needs both binary outcomes")
        return cls(((selected-mean)/std).T, events, mean, std, int(counts.max()))


def bernoulli_objective(parameters: FloatArray, fitting: FittingData) -> tuple[float, FloatArray]:
    x = fitting.standardized[:parameters.size-2]
    rate = vectorized_softplus(x, parameters)
    derivative = vectorized_softplus_derivative(x, parameters)
    spike = fitting.events > 0
    terms = rate.copy()
    slope = np.ones_like(rate)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        terms[spike] = -np.log(-np.expm1(-rate[spike]))
        slope[spike] = -1 / np.expm1(rate[spike])
        gradient = slope @ derivative / fitting.events.sum()
    return float(terms.sum() / fitting.events.sum()), gradient


@dataclass(frozen=True, slots=True)
class OutputFit:
    parameters: tuple[float, ...]
    initial_parameters: tuple[float, ...]
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    gradient: tuple[float, ...]
    train_nll: float
    mean: tuple[float, float]
    std: tuple[float, float]

    def expected_counts(self, features: FloatArray) -> FloatArray:
        standardized = ((features-np.asarray(self.mean))/np.asarray(self.std)).reshape(-1, 2).T
        values = vectorized_softplus(standardized[:len(self.parameters)-2], self.parameters)
        return values.reshape(features.shape[:-1])


def fit_output(fitting: FittingData, *, spatial_contrast: bool) -> OutputFit:
    initial = np.array([float(fitting.maximum_count), -2., 1., 0.])
    if not spatial_contrast:
        initial = initial[:3]
    bounds = [(0., None), *[(None, None)] * (len(initial)-1)]
    result = minimize(bernoulli_objective, initial, args=(fitting,), jac=True,
                      bounds=bounds, method="L-BFGS-B", options=None)
    value, gradient = bernoulli_objective(result.x, fitting)
    if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(result.x).all():
        raise LNError("STOP: nonfinite optimizer endpoint; no restart permitted")
    return OutputFit(tuple(result.x.tolist()), tuple(initial.tolist()), bool(result.success),
                     int(result.status), str(result.message), int(result.nit), int(result.nfev),
                     tuple(gradient.tolist()), value*float(fitting.events.mean()),
                     (float(fitting.mean[0]), float(fitting.mean[1])),
                     (float(fitting.std[0]), float(fitting.std[1])))


def reported_nll(rate: FloatArray, events: torch.Tensor, mask: torch.Tensor) -> float:
    values = rate[mask.numpy()[..., 0]]
    spike = events.numpy()[mask.numpy()] > 0
    terms = values.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms[spike] = -np.log(-np.expm1(-values[spike]))
    if not np.isfinite(terms).all():
        raise LNError("STOP: reported Bernoulli NLL is nonfinite")
    return float(terms.mean())
