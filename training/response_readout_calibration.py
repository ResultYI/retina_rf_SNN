from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from data.rgc_response import ResponseTargetKind
from loss.rgc_response import response_nll
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit, masked_history_counts


STAGE05_L2: Final = 1e-4
STAGE05_MAX_ITERATIONS: Final = 200


class Stage05ReadoutCalibrationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Stage05ReadoutCalibrationRequest:
    model: ResponseRetinaModel
    train: ResponseSplit
    target_kind: ResponseTargetKind
    burn_in_steps: int
    device: torch.device
    expected_targets: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class Stage05ReadoutCalibrationResult:
    pre_train_nll: float
    post_train_nll: float
    fitted_bipolar_gain: tuple[tuple[float, ...], ...]
    fitted_amacrine_gain: tuple[tuple[float, ...], ...]
    bias_delta: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ReadoutTraces:
    features: torch.Tensor
    base_logits: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor


def fit_stage05_readout_calibration(
    request: Stage05ReadoutCalibrationRequest,
) -> Stage05ReadoutCalibrationResult:
    if request.target_kind is not ResponseTargetKind.BERNOULLI:
        raise Stage05ReadoutCalibrationError(
            "Stage0.5 requires Bernoulli response targets"
        )
    rgc = request.model.rgc
    for name in (
        "response_bias",
        "bipolar_readout_gain",
        "amacrine_readout_gain",
    ):
        if not hasattr(rgc, name):
            raise Stage05ReadoutCalibrationError(
                "Stage0.5 requires response bias and direct readout"
            )
    traces = _trace_readout_features(request)
    mean = traces.features.mean(dim=(0, 1, 2))
    std = traces.features.std(dim=(0, 1, 2)).clamp_min(1e-6)
    standardized = (traces.features - mean) / std
    weight = nn.Parameter(
        torch.zeros(rgc.cell_count, 4, device=request.device)
    )
    bias = nn.Parameter(torch.zeros(rgc.cell_count, device=request.device))
    optimizer = torch.optim.LBFGS(
        (weight, bias),
        max_iter=STAGE05_MAX_ITERATIONS,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = traces.base_logits + (standardized * weight).sum(dim=-1) + bias
        elements = nn.functional.binary_cross_entropy_with_logits(
            logits,
            traces.targets,
            reduction="none",
        )
        loss = elements[traces.valid_mask].mean()
        regularized = loss + STAGE05_L2 * weight.square().mean()
        regularized.backward()
        return regularized

    optimizer.step(closure)
    raw_weight = weight.detach() / std
    bias_delta = bias.detach() - (raw_weight * mean).sum(dim=-1)
    bipolar = raw_weight[:, :2].transpose(0, 1)
    amacrine = raw_weight[:, 2:].transpose(0, 1)
    with torch.no_grad():
        rgc.bipolar_readout_gain.copy_(bipolar)
        rgc.amacrine_readout_gain.copy_(amacrine)
        rgc.response_bias.add_(bias_delta)
        post_logits = traces.base_logits + (traces.features * raw_weight).sum(
            dim=-1
        ) + bias_delta
    return Stage05ReadoutCalibrationResult(
        pre_train_nll=float(
            response_nll(
                traces.base_logits,
                traces.targets,
                traces.valid_mask,
                request.target_kind,
            )
        ),
        post_train_nll=float(
            response_nll(
                post_logits,
                traces.targets,
                traces.valid_mask,
                request.target_kind,
            )
        ),
        fitted_bipolar_gain=_matrix_values(bipolar),
        fitted_amacrine_gain=_matrix_values(amacrine),
        bias_delta=tuple(float(value) for value in bias_delta),
    )


@torch.no_grad()
def _trace_readout_features(
    request: Stage05ReadoutCalibrationRequest,
) -> _ReadoutTraces:
    model = request.model
    model.eval()
    rgc = model.rgc
    polarities = rgc.cell_polarities
    if not isinstance(polarities, torch.Tensor):
        raise Stage05ReadoutCalibrationError("RGC polarities must be a tensor")
    spatial_weights = rgc.compute_spatial_weights().detach()
    feature_rows: list[torch.Tensor] = []
    generator_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    split = request.train
    targets = (
        split.spike_counts
        if request.expected_targets is None
        else request.expected_targets
    )
    if targets.shape != split.spike_counts.shape:
        raise Stage05ReadoutCalibrationError(
            "Expected targets must match the sampled response shape"
        )
    trial_count = split.spike_counts.shape[1]
    for stimulus in range(split.cone_response.shape[0]):
        cones = split.cone_response[stimulus : stimulus + 1].expand(
            trial_count, -1, -1
        ).to(request.device)
        counts = split.spike_counts[stimulus].to(request.device)
        mask = split.valid_mask[stimulus].to(request.device)
        observed = masked_history_counts(counts, mask)
        state = model.initial_state(trial_count, request.device, cones.dtype)
        stimulus_features: list[torch.Tensor] = []
        stimulus_generator: list[torch.Tensor] = []
        for index, cone_t in enumerate(cones.unbind(dim=1)):
            output, state = model.step(
                cone_t,
                state,
                spatial_weights,
                observed[:, index],
            )
            if index >= request.burn_in_steps:
                bipolar = _pool_selected(
                    state.bipolar.output,
                    spatial_weights,
                    polarities,
                )
                amacrine = _pool_selected(
                    state.amacrine,
                    spatial_weights,
                    polarities,
                )
                stimulus_features.append(torch.cat((bipolar, amacrine), dim=-1))
                stimulus_generator.append(output.generator_potential)
        feature_rows.append(torch.stack(stimulus_features, dim=1))
        generator_rows.append(torch.stack(stimulus_generator, dim=1))
        target_rows.append(
            targets[stimulus, :, request.burn_in_steps :].to(request.device)
        )
        mask_rows.append(mask[:, request.burn_in_steps :])
    generator = torch.stack(generator_rows)
    return _ReadoutTraces(
        features=torch.stack(feature_rows),
        base_logits=rgc.logits_from_generator(generator).detach(),
        targets=torch.stack(target_rows),
        valid_mask=torch.stack(mask_rows),
    )


def _pool_selected(
    features: torch.Tensor,
    spatial_weights: torch.Tensor,
    polarities: torch.Tensor,
) -> torch.Tensor:
    pooled = torch.einsum("uc,bpkc->bpku", spatial_weights, features)
    indices = polarities.view(1, 1, 1, -1).expand(features.shape[0], 1, 2, -1)
    return pooled.gather(1, indices).squeeze(1).transpose(1, 2)


def _matrix_values(values: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


__all__ = [
    "STAGE05_L2",
    "STAGE05_MAX_ITERATIONS",
    "Stage05ReadoutCalibrationError",
    "Stage05ReadoutCalibrationRequest",
    "Stage05ReadoutCalibrationResult",
    "fit_stage05_readout_calibration",
]
