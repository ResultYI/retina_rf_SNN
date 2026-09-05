from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, assert_never

import torch
from torch.nn import functional as F

from data.rgc_response import ResponseTargetKind
from evaluation.response_metrics import ResponseMetrics, compute_response_metrics
from evaluation.response_predictions import ResponsePredictionTensors
from loss.rgc_response import response_nll
from models.response_snn import ResponseRetinaModel

CalibrationMode: TypeAlias = Literal["intercept", "affine"]
_UNIT_SOFTPLUS_RAW: Final = math.log(math.expm1(1.0))


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LogitCalibrationRequest:
    train: ResponsePredictionTensors
    validation: ResponsePredictionTensors
    target_kind: ResponseTargetKind
    baseline_rates: torch.Tensor
    mode: CalibrationMode
    max_iterations: int = 50


@dataclass(frozen=True, slots=True)
class LogitCalibrationResult:
    mode: CalibrationMode
    scales: tuple[float, ...]
    intercepts: tuple[float, ...]
    train_metrics: ResponseMetrics
    validation_metrics: ResponseMetrics


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationRequest:
    model: ResponseRetinaModel
    train: ResponsePredictionTensors
    validation: ResponsePredictionTensors
    target_kind: ResponseTargetKind
    baseline_rates: torch.Tensor
    max_iterations: int = 50
    tolerance: float = 0.001


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationResult:
    initial_thresholds: tuple[float, ...]
    calibrated_thresholds: tuple[float, ...]
    changed_parameter_names: tuple[str, ...]
    train_metrics: ResponseMetrics
    validation_metrics: ResponseMetrics
    validation_baseline_gap: float
    passed: bool


def fit_logit_calibration(
    request: LogitCalibrationRequest,
) -> LogitCalibrationResult:
    _validate_request(request.target_kind, request.max_iterations)
    _validate_predictions(request.train, request.baseline_rates)
    _validate_predictions(request.validation, request.baseline_rates)
    cell_count = request.train.logits.shape[-1]
    device = request.train.logits.device
    intercepts = torch.zeros(cell_count, device=device, requires_grad=True)
    raw_scales: torch.Tensor | None
    match request.mode:
        case "intercept":
            raw_scales = None
            parameters = [intercepts]
        case "affine":
            raw_scales = torch.full(
                (cell_count,),
                _UNIT_SOFTPLUS_RAW,
                device=device,
                requires_grad=True,
            )
            parameters = [intercepts, raw_scales]
        case unreachable:
            assert_never(unreachable)
    optimizer = torch.optim.LBFGS(
        parameters,
        max_iter=request.max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = response_nll(
            _calibrated_logits(request.train.logits, intercepts, raw_scales),
            request.train.targets,
            request.train.valid_mask,
            request.target_kind,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scales = (
            torch.ones_like(intercepts)
            if raw_scales is None
            else F.softplus(raw_scales)
        )
        train_logits = _calibrated_logits(
            request.train.logits,
            intercepts,
            raw_scales,
        )
        validation_logits = _calibrated_logits(
            request.validation.logits,
            intercepts,
            raw_scales,
        )
        return LogitCalibrationResult(
            request.mode,
            tuple(float(value) for value in scales.cpu()),
            tuple(float(value) for value in intercepts.cpu()),
            _metrics(request.train, train_logits, request),
            _metrics(request.validation, validation_logits, request),
        )


def fit_threshold_calibration(
    request: ThresholdCalibrationRequest,
) -> ThresholdCalibrationResult:
    _validate_request(request.target_kind, request.max_iterations)
    _validate_predictions(request.train, request.baseline_rates)
    _validate_predictions(request.validation, request.baseline_rates)
    model = copy.deepcopy(request.model)
    model.train(False)
    initial_thresholds = model.rgc.threshold().detach().clone()
    before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    threshold_parameters = list(model.rgc.threshold.parameters())
    for parameter in threshold_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        threshold_parameters,
        max_iter=request.max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = response_nll(
            model.rgc.logits_from_generator(request.train.generator_potential),
            request.train.targets,
            request.train.valid_mask,
            request.target_kind,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        train_logits = model.rgc.logits_from_generator(
            request.train.generator_potential
        )
        validation_logits = model.rgc.logits_from_generator(
            request.validation.generator_potential
        )
        train_metrics = _metrics(request.train, train_logits, request)
        validation_metrics = _metrics(
            request.validation,
            validation_logits,
            request,
        )
        gap = validation_metrics.nll - validation_metrics.constant_rate_nll
        changed = tuple(
            name
            for name, value in model.named_parameters()
            if not torch.equal(value, before[name])
        )
        return ThresholdCalibrationResult(
            tuple(float(value) for value in initial_thresholds.cpu()),
            tuple(float(value) for value in model.rgc.threshold().cpu()),
            changed,
            train_metrics,
            validation_metrics,
            gap,
            gap <= request.tolerance,
        )


def _calibrated_logits(
    logits: torch.Tensor,
    intercepts: torch.Tensor,
    raw_scales: torch.Tensor | None,
) -> torch.Tensor:
    if raw_scales is None:
        return logits + intercepts
    return logits * F.softplus(raw_scales) + intercepts


def _metrics(
    predictions: ResponsePredictionTensors,
    logits: torch.Tensor,
    request: LogitCalibrationRequest | ThresholdCalibrationRequest,
) -> ResponseMetrics:
    return compute_response_metrics(
        logits,
        predictions.targets,
        predictions.valid_mask,
        request.target_kind,
        request.baseline_rates,
    )


def _validate_request(
    target_kind: ResponseTargetKind,
    max_iterations: int,
) -> None:
    if target_kind is not ResponseTargetKind.BERNOULLI:
        raise CalibrationError("Calibration audit requires Bernoulli targets")
    if max_iterations <= 0:
        raise CalibrationError("max_iterations must be positive")


def _validate_predictions(
    predictions: ResponsePredictionTensors,
    baseline_rates: torch.Tensor,
) -> None:
    if (
        predictions.logits.shape != predictions.targets.shape
        or predictions.logits.shape != predictions.valid_mask.shape
        or predictions.logits.shape != predictions.generator_potential.shape
    ):
        raise CalibrationError("Calibration prediction tensors must share a shape")
    if predictions.logits.ndim != 4:
        raise CalibrationError("Calibration predictions must be four-dimensional")
    if baseline_rates.shape != (predictions.logits.shape[-1],):
        raise CalibrationError("Calibration needs one baseline rate per cell")


__all__ = [
    "CalibrationError",
    "CalibrationMode",
    "LogitCalibrationRequest",
    "LogitCalibrationResult",
    "ThresholdCalibrationRequest",
    "ThresholdCalibrationResult",
    "fit_logit_calibration",
    "fit_threshold_calibration",
]
