from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from configs.rgc_type_priors import RGCTypePriors
from data.rgc_response import CellMetadata
from models.cells.amacrine import LocalAmacrineLayer
from models.cells.bipolar import BipolarLayer, BipolarState
from models.cells.horizontal import H1HorizontalNetwork
from models.cells.typed_rgc import (
    ParameterSharingMode,
    ReadoutMode,
    TypedRGCOutput,
    TypedRGCPopulation,
    TypedRGCState,
    TypedRGCStepOutput,
)

if TYPE_CHECKING:
    from configs.physiology_profiles import PhysiologyProfile


class ResponseSNNError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponseRetinaState:
    h1: torch.Tensor
    bipolar: BipolarState
    amacrine: torch.Tensor
    rgc: TypedRGCState


class ResponseRetinaModel(nn.Module):
    def __init__(
        self,
        h1: H1HorizontalNetwork,
        bipolar: BipolarLayer,
        amacrine: LocalAmacrineLayer,
        rgc: TypedRGCPopulation,
    ) -> None:
        super().__init__()
        self.h1 = h1
        self.bipolar = bipolar
        self.amacrine = amacrine
        self.rgc = rgc

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> ResponseRetinaState:
        return ResponseRetinaState(
            h1=self.h1.initial_state(batch_size, device, dtype),
            bipolar=self.bipolar.initial_state(batch_size, device, dtype),
            amacrine=self.amacrine.initial_state(batch_size, device, dtype),
            rgc=self.rgc.initial_state(batch_size, device, dtype),
        )

    def step(
        self,
        cone_t: torch.Tensor,
        state: ResponseRetinaState,
        spatial_weights: torch.Tensor,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[TypedRGCStepOutput, ResponseRetinaState]:
        cone_modulated, h1_state = self.h1(cone_t, state.h1)
        bipolar_state = self.bipolar(
            cone_modulated,
            state.bipolar,
            amacrine_prev=state.amacrine,
        )
        amacrine_state = self.amacrine(bipolar_state.output, state.amacrine)
        rgc_output, rgc_state = self.rgc(
            bipolar_state.output,
            amacrine_state,
            state.rgc,
            spatial_weights,
            observed_counts,
        )
        return rgc_output, ResponseRetinaState(
            h1_state, bipolar_state, amacrine_state, rgc_state
        )

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        state: ResponseRetinaState | None = None,
        *,
        observed_counts: torch.Tensor | None = None,
        spatial_weights: torch.Tensor | None = None,
    ) -> tuple[TypedRGCOutput, ResponseRetinaState]:
        if sequence.ndim != 3:
            raise ResponseSNNError("sequence must have shape [batch,time,cone]")
        if observed_counts is not None and observed_counts.shape[:2] != sequence.shape[:2]:
            raise ResponseSNNError("observed counts must match batch and time")
        if state is None:
            state = self.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
        if spatial_weights is None:
            spatial_weights = self.rgc.compute_spatial_weights()
        histories: list[list[torch.Tensor]] = [[], [], [], [], []]
        for index, cone_t in enumerate(sequence.unbind(dim=1)):
            counts_t = None if observed_counts is None else observed_counts[:, index]
            output, state = self.step(cone_t, state, spatial_weights, counts_t)
            for history, value in zip(
                histories,
                (
                    output.spike_logits,
                    output.spike_probability,
                    output.hard_spikes,
                    output.filtered_rate,
                    output.generator_potential,
                ),
                strict=True,
            ):
                history.append(value)
        return TypedRGCOutput(
            *(torch.stack(history, dim=1) for history in histories)
        ), state


def build_response_retina_model(
    cone_positions_degs: torch.Tensor,
    cells: CellMetadata,
    profile: PhysiologyProfile,
    priors: RGCTypePriors,
    *,
    support_radius_degs: float,
    readout_rate_tau_ms: float,
    surrogate_slope: float,
    parameter_sharing_mode: ParameterSharingMode = "type_aware",
    parameter_sharing_seed: int = 0,
    matched_initialization: bool = False,
    enable_response_bias: bool = False,
    enable_synaptic_gain: bool = False,
    enable_direct_readout: bool = False,
    synaptic_gain_min: float = 0.1,
    synaptic_gain_max: float = 4.0,
    synaptic_gain_init: float = 1.0,
    readout_mode: ReadoutMode = "v2_direct_logit",
) -> ResponseRetinaModel:
    return ResponseRetinaModel(
        H1HorizontalNetwork(cone_positions_degs, profile.h1),
        BipolarLayer(cone_positions_degs, profile.bipolar),
        LocalAmacrineLayer(cone_positions_degs, profile.amacrine),
        TypedRGCPopulation(
            torch.as_tensor(cone_positions_degs),
            cells,
            priors,
            dt_ms=profile.h1.dt_ms,
            support_radius_degs=support_radius_degs,
            readout_rate_tau_ms=readout_rate_tau_ms,
            surrogate_slope=surrogate_slope,
            parameter_sharing_mode=parameter_sharing_mode,
            parameter_sharing_seed=parameter_sharing_seed,
            matched_initialization=matched_initialization,
            enable_response_bias=enable_response_bias,
            enable_synaptic_gain=enable_synaptic_gain,
            enable_direct_readout=enable_direct_readout,
            synaptic_gain_min=synaptic_gain_min,
            synaptic_gain_max=synaptic_gain_max,
            synaptic_gain_init=synaptic_gain_init,
            readout_mode=readout_mode,
        ),
    )


def detach_response_state(state: ResponseRetinaState) -> ResponseRetinaState:
    return ResponseRetinaState(
        state.h1.detach(),
        BipolarState(
            state.bipolar.output.detach(),
            state.bipolar.transient_baseline.detach(),
        ),
        state.amacrine.detach(),
        TypedRGCState(
            state.rgc.membrane.detach(),
            state.rgc.adaptation.detach(),
            state.rgc.rate.detach(),
            state.rgc.subunit_energy.detach(),
        ),
    )


def response_state_to_tensors(
    state: ResponseRetinaState,
) -> tuple[torch.Tensor, ...]:
    return (
        state.h1,
        state.bipolar.output,
        state.bipolar.transient_baseline,
        state.amacrine,
        state.rgc.membrane,
        state.rgc.adaptation,
        state.rgc.rate,
        state.rgc.subunit_energy,
    )


def response_state_from_tensors(
    tensors: tuple[torch.Tensor, ...],
) -> ResponseRetinaState:
    if len(tensors) != 8:
        raise ResponseSNNError("Response retina state requires eight tensors")
    return ResponseRetinaState(
        h1=tensors[0],
        bipolar=BipolarState(tensors[1], tensors[2]),
        amacrine=tensors[3],
        rgc=TypedRGCState(tensors[4], tensors[5], tensors[6], tensors[7]),
    )


__all__ = [
    "ResponseRetinaModel",
    "ResponseRetinaState",
    "ResponseSNNError",
    "build_response_retina_model",
    "detach_response_state",
    "response_state_from_tensors",
    "response_state_to_tensors",
]
