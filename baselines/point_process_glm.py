from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from data.rgc_response import ResponseTargetKind
from evaluation.response_metrics import ResponseMetrics, compute_response_metrics
from loss.rgc_response import response_nll
from training.response_data import PreparedResponseData, ResponseSplit


class GLMError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GLMFitResult:
    model: PointProcessGLM
    metrics: ResponseMetrics


class PointProcessGLM(nn.Module):
    def __init__(
        self,
        cone_count: int,
        cell_count: int,
        temporal_lags: int,
    ) -> None:
        super().__init__()
        self.kernel = nn.Parameter(
            torch.zeros(cell_count, temporal_lags, cone_count)
        )
        self.history = nn.Parameter(torch.zeros(cell_count, 4))
        self.bias = nn.Parameter(torch.zeros(cell_count))

    def forward(
        self,
        cones: torch.Tensor,
        observed_counts: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.bias.view(1, 1, -1).expand(
            cones.shape[0], cones.shape[1], -1
        ).clone()
        for lag in range(self.kernel.shape[1]):
            if lag >= cones.shape[1]:
                break
            logits[:, lag:] += torch.einsum(
                "btc,rc->btr",
                cones[:, : cones.shape[1] - lag],
                self.kernel[:, lag],
            )
        for lag in range(1, self.history.shape[1] + 1):
            logits[:, lag:] += (
                observed_counts[:, : observed_counts.shape[1] - lag]
                * self.history[:, lag - 1]
            )
        return logits


def fit_point_process_glm(
    data: PreparedResponseData,
    *,
    device: torch.device,
    steps: int = 25,
    temporal_lags: int = 8,
) -> GLMFitResult:
    model = PointProcessGLM(
        data.train.cone_response.shape[-1],
        data.train.spike_counts.shape[-1],
        temporal_lags,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    cones, counts, mask = _all_trials(data.train, device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = response_nll(
            model(cones, counts),
            counts,
            mask,
            data.target_kind,
        )
        loss.backward()
        optimizer.step()
    validation_cones, validation_counts, validation_mask = _all_trials(
        data.validation, device
    )
    with torch.no_grad():
        logits = model(validation_cones, validation_counts)
    baseline_rates = _baseline_rates(data.train, device)
    metrics = compute_response_metrics(
        logits,
        validation_counts,
        validation_mask,
        data.target_kind,
        baseline_rates,
    )
    return GLMFitResult(model, metrics)


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
) -> torch.Tensor:
    mask = split.valid_mask.to(torch.float32)
    rates = (split.spike_counts * mask).sum(dim=(0, 1, 2))
    rates /= mask.sum(dim=(0, 1, 2)).clamp_min(1)
    if rates.numel() == 0:
        raise GLMError(f"No valid {ResponseTargetKind} responses")
    return rates.to(device)


__all__ = ["GLMError", "GLMFitResult", "PointProcessGLM", "fit_point_process_glm"]
