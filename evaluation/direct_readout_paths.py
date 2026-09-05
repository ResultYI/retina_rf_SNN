from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from models.response_snn import ResponseRetinaModel, ResponseRetinaState


class DirectReadoutPathError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DirectReadoutPathOutput:
    total: torch.Tensor
    core: torch.Tensor
    bipolar_direct: torch.Tensor
    amacrine_direct: torch.Tensor
    bipolar_features: torch.Tensor
    amacrine_features: torch.Tensor
    generator_potential: torch.Tensor


def forward_sequence_readout_paths(
    model: ResponseRetinaModel,
    sequence: torch.Tensor,
    state: ResponseRetinaState | None = None,
    *,
    observed_counts: torch.Tensor | None = None,
    spatial_weights: torch.Tensor | None = None,
    identity_atol: float = 2e-6,
) -> tuple[DirectReadoutPathOutput, ResponseRetinaState]:
    if sequence.ndim != 3:
        raise DirectReadoutPathError("sequence must have shape [batch,time,cone]")
    if observed_counts is not None and observed_counts.shape[:2] != sequence.shape[:2]:
        raise DirectReadoutPathError("observed counts must match batch and time")
    if state is None:
        state = model.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
    if spatial_weights is None:
        spatial_weights = model.rgc.compute_spatial_weights()

    histories: list[list[torch.Tensor]] = [[] for _ in range(7)]
    for time, cone_t in enumerate(sequence.unbind(dim=1)):
        cone_modulated, h1_state = model.h1(cone_t, state.h1)
        bipolar_state = model.bipolar(
            cone_modulated,
            state.bipolar,
            amacrine_prev=state.amacrine,
        )
        amacrine_state = model.amacrine(bipolar_state.output, state.amacrine)
        bipolar_features = _selected_features(
            model,
            bipolar_state.output,
            spatial_weights,
        )
        amacrine_features = _selected_features(
            model,
            amacrine_state,
            spatial_weights,
        )
        counts_t = None if observed_counts is None else observed_counts[:, time]
        rgc_output, rgc_state = model.rgc(
            bipolar_state.output,
            amacrine_state,
            state.rgc,
            spatial_weights,
            counts_t,
        )
        core = model.rgc.logits_from_generator(rgc_output.generator_potential)
        bipolar_direct, amacrine_direct = _direct_logits(
            model,
            bipolar_features,
            amacrine_features,
        )
        reconstructed = core + bipolar_direct + amacrine_direct
        if not torch.allclose(
            reconstructed,
            rgc_output.spike_logits,
            atol=identity_atol,
            rtol=identity_atol,
        ):
            maximum = float((reconstructed - rgc_output.spike_logits).abs().max())
            raise DirectReadoutPathError(
                f"direct-readout path identity failed: max_abs={maximum:.9g}"
            )
        for history, value in zip(
            histories,
            (
                rgc_output.spike_logits,
                core,
                bipolar_direct,
                amacrine_direct,
                bipolar_features,
                amacrine_features,
                rgc_output.generator_potential,
            ),
            strict=True,
        ):
            history.append(value)
        state = ResponseRetinaState(h1_state, bipolar_state, amacrine_state, rgc_state)

    return (
        DirectReadoutPathOutput(*(torch.stack(values, dim=1) for values in histories)),
        state,
    )


@contextmanager
def direct_readout_intervention(
    model: ResponseRetinaModel,
    *,
    disable_bipolar: bool,
    disable_amacrine: bool,
) -> Iterator[None]:
    bipolar = model.rgc.bipolar_readout_gain
    amacrine = model.rgc.amacrine_readout_gain
    saved_bipolar = bipolar.detach().clone()
    saved_amacrine = amacrine.detach().clone()
    try:
        with torch.no_grad():
            if disable_bipolar:
                bipolar.zero_()
            if disable_amacrine:
                amacrine.zero_()
        yield
    finally:
        with torch.no_grad():
            bipolar.copy_(saved_bipolar)
            amacrine.copy_(saved_amacrine)


def _selected_features(
    model: ResponseRetinaModel,
    values: torch.Tensor,
    spatial_weights: torch.Tensor,
) -> torch.Tensor:
    pooled = torch.einsum("uc,bpkc->bpku", spatial_weights, values)
    polarity = model.rgc.cell_polarities.view(1, 1, 1, -1).expand(
        values.shape[0], 1, 2, -1
    )
    return pooled.gather(1, polarity).squeeze(1)


def _direct_logits(
    model: ResponseRetinaModel,
    bipolar_features: torch.Tensor,
    amacrine_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not model.rgc._enable_direct_readout:  # noqa: SLF001 - exact audit contract
        zero = bipolar_features[:, 0] * 0
        return zero, zero.clone()
    return (
        (model.rgc.bipolar_readout_gain.unsqueeze(0) * bipolar_features).sum(dim=1),
        (model.rgc.amacrine_readout_gain.unsqueeze(0) * amacrine_features).sum(dim=1),
    )


__all__ = [
    "DirectReadoutPathError",
    "DirectReadoutPathOutput",
    "direct_readout_intervention",
    "forward_sequence_readout_paths",
]
