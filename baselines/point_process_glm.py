from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Literal, TypeAlias, assert_never

import torch
from torch import nn

from data.rgc_response import ResponseTargetKind
from evaluation.response_metrics import ResponseMetrics, compute_response_metrics
from loss.rgc_response import response_nll
from training.response_data import (
    PreparedResponseData,
    ResponseSplit,
    masked_history_counts,
)


class GLMError(ValueError):
    pass


GLMMode: TypeAlias = Literal["bias_only", "bias_plus_history", "full_glm"]


@dataclass(frozen=True, slots=True)
class GLMFitResult:
    model: PointProcessGLM
    validation_metrics: ResponseMetrics
    test_metrics: ResponseMetrics | None
    best_step: int

    @property
    def evaluation_metrics(self) -> ResponseMetrics:
        return self.validation_metrics if self.test_metrics is None else self.test_metrics


class PointProcessGLM(nn.Module):
    def __init__(
        self,
        cone_count: int,
        cell_count: int,
        temporal_lags: int,
        *,
        mode: str = "full_glm",
    ) -> None:
        super().__init__()
        self.mode = _parse_glm_mode(mode)
        self.kernel = nn.Parameter(
            torch.zeros(cell_count, temporal_lags, cone_count)
        )
        self.history = nn.Parameter(torch.zeros(cell_count, 4))
        self.bias = nn.Parameter(torch.zeros(cell_count))
        self.kernel.requires_grad_(self.mode == "full_glm")
        self.history.requires_grad_(self.mode != "bias_only")

    def forward(
        self,
        cones: torch.Tensor,
        observed_counts: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.bias.view(1, 1, -1).expand(
            cones.shape[0], cones.shape[1], -1
        ).clone()
        match self.mode:
            case "bias_only":
                return logits
            case "bias_plus_history":
                pass
            case "full_glm":
                for lag in range(self.kernel.shape[1]):
                    if lag >= cones.shape[1]:
                        break
                    logits[:, lag:] += torch.einsum(
                        "btc,rc->btr",
                        cones[:, : cones.shape[1] - lag],
                        self.kernel[:, lag],
                    )
            case unreachable:
                assert_never(unreachable)
        for lag in range(1, self.history.shape[1] + 1):
            if lag >= observed_counts.shape[1]:
                break
            logits[:, lag:] += (
                observed_counts[:, : observed_counts.shape[1] - lag]
                * self.history[:, lag - 1]
            )
        return logits


def fit_point_process_glm(
    data: PreparedResponseData,
    *,
    device: torch.device,
    steps: int = 100,
    temporal_lags: int = 16,
    burn_in_steps: int = 0,
    evaluate_test: bool = False,
    mode: str = "full_glm",
) -> GLMFitResult:
    model = PointProcessGLM(
        data.train.cone_response.shape[-1],
        data.train.spike_counts.shape[-1],
        temporal_lags,
        mode=mode,
    ).to(device)
    baseline_rates = _baseline_rates(data.train, device, burn_in_steps)
    with torch.no_grad():
        model.bias.copy_(_baseline_bias(baseline_rates, data.target_kind))
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.03,
    )
    cones, counts, mask = _all_trials(data.train, device)
    history_counts = masked_history_counts(counts, mask)
    mask = _supervised_mask(mask, burn_in_steps)
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    best_nll = float("inf")
    fitted_steps = 0 if model.mode == "bias_only" else steps
    for step in range(1, fitted_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = response_nll(
            model(cones, history_counts),
            counts,
            mask,
            data.target_kind,
        )
        loss.backward()
        optimizer.step()
        validation = _evaluate_split(
            model,
            data.validation,
            data.target_kind,
            baseline_rates,
            device,
            burn_in_steps,
        )
        if validation.nll < best_nll:
            best_nll = validation.nll
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    validation = _evaluate_split(
        model,
        data.validation,
        data.target_kind,
        baseline_rates,
        device,
        burn_in_steps,
    )
    test = (
        _evaluate_split(
            model,
            data.test,
            data.target_kind,
            baseline_rates,
            device,
            burn_in_steps,
        )
        if evaluate_test
        else None
    )
    return GLMFitResult(model, validation, test, best_step)


def _evaluate_split(
    model: PointProcessGLM,
    split: ResponseSplit,
    target_kind: ResponseTargetKind,
    baseline_rates: torch.Tensor,
    device: torch.device,
    burn_in_steps: int,
) -> ResponseMetrics:
    cones, counts, mask = _all_trials(split, device)
    history_counts = masked_history_counts(counts, mask)
    mask = _supervised_mask(mask, burn_in_steps)
    with torch.no_grad():
        logits = model(cones, history_counts)
    shape = split.spike_counts.shape
    return compute_response_metrics(
        logits.reshape(shape),
        counts.reshape(shape),
        mask.reshape(shape),
        target_kind,
        baseline_rates,
    )


def _all_trials(
    split: ResponseSplit,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stimulus_count, trial_count = split.spike_counts.shape[:2]
    cones = (
        split.cone_response[:, None]
        .expand(-1, trial_count, -1, -1)
        .reshape(
            stimulus_count * trial_count,
            split.cone_response.shape[1],
            split.cone_response.shape[2],
        )
    )
    return (
        cones.to(device),
        split.spike_counts.flatten(0, 1).to(device),
        split.valid_mask.flatten(0, 1).to(device),
    )


def _baseline_rates(
    split: ResponseSplit,
    device: torch.device,
    burn_in_steps: int,
) -> torch.Tensor:
    mask = _supervised_mask(split.valid_mask, burn_in_steps).to(torch.float32)
    rates = (split.spike_counts * mask).sum(dim=(0, 1, 2))
    rates /= mask.sum(dim=(0, 1, 2)).clamp_min(1)
    if rates.numel() == 0:
        raise GLMError(f"No valid {ResponseTargetKind} responses")
    return rates.to(device)


def _supervised_mask(mask: torch.Tensor, burn_in_steps: int) -> torch.Tensor:
    if burn_in_steps < 0 or burn_in_steps >= mask.shape[-2]:
        raise GLMError("burn_in_steps must leave at least one supervised bin")
    supervised = mask.clone()
    supervised[..., :burn_in_steps, :] = False
    return supervised


def _baseline_bias(
    rates: torch.Tensor,
    target_kind: ResponseTargetKind,
) -> torch.Tensor:
    match target_kind:
        case ResponseTargetKind.BERNOULLI:
            return torch.logit(rates.clamp(1e-5, 1 - 1e-5))
        case ResponseTargetKind.POISSON:
            return torch.log(torch.expm1(rates.clamp_min(1e-5)))
        case unreachable:
            assert_never(unreachable)


def _parse_glm_mode(value: str) -> GLMMode:
    match value:
        case "bias_only" | "bias_plus_history" | "full_glm" as mode:
            return mode
        case _:
            raise GLMError(f"Unknown GLM mode: {value!r}")


__all__ = [
    "GLMError",
    "GLMFitResult",
    "GLMMode",
    "PointProcessGLM",
    "fit_point_process_glm",
]
