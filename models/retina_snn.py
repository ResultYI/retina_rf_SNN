from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from data.geometry import PositionArray
from models.cells.amacrine import LocalAmacrineLayer
from models.cells.bipolar import BipolarLayer, BipolarState
from models.cells.horizontal import H1HorizontalNetwork
from models.cells.rgc import HeterogeneousRGCPool
from models.cells.rgc_runtime import detach_rgc_state
from models.cells.rgc_types import RGCConfig, RGCOutput, RGCState, RGCStepOutput

if TYPE_CHECKING:
    from configs.physiology_profiles import PhysiologyProfile


class RetinaModelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetinaState:
    h1: torch.Tensor
    bipolar: BipolarState
    amacrine: torch.Tensor
    rgc: RGCState


class RetinaModel(nn.Module):
    def __init__(
        self,
        h1: H1HorizontalNetwork,
        bipolar: BipolarLayer,
        amacrine: LocalAmacrineLayer,
        rgc: HeterogeneousRGCPool,
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
    ) -> RetinaState:
        return RetinaState(
            h1=self.h1.initial_state(batch_size, device, dtype),
            bipolar=self.bipolar.initial_state(batch_size, device, dtype),
            amacrine=self.amacrine.initial_state(batch_size, device, dtype),
            rgc=self.rgc.initial_state(batch_size, device, dtype),
        )

    def step(
        self,
        cone_t: torch.Tensor,
        state: RetinaState,
        spatial_weights: torch.Tensor,
        *,
        probe_continuous_output: bool = False,
    ) -> tuple[RGCStepOutput, RetinaState]:
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
            probe_continuous_output=probe_continuous_output,
        )
        return rgc_output, RetinaState(
            h1=h1_state,
            bipolar=bipolar_state,
            amacrine=amacrine_state,
            rgc=rgc_state,
        )

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        state: RetinaState | None = None,
        *,
        spatial_weights: torch.Tensor | None = None,
        probe_continuous_output: bool = False,
    ) -> tuple[RGCOutput, RetinaState]:
        cone_count = self.rgc.cone_positions_degs.shape[0]
        if sequence.ndim != 3 or sequence.shape[1] < 1 or sequence.shape[2] != cone_count:
            raise RetinaModelError("sequence must have shape [batch,time,Ncone]")
        if state is None:
            state = self.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
        if spatial_weights is None:
            spatial_weights = self.rgc.compute_spatial_weights()

        hard_spikes: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        rates: list[torch.Tensor] = []
        generators: list[torch.Tensor] = []
        for cone_t in sequence.unbind(dim=1):
            output, state = self.step(
                cone_t,
                state,
                spatial_weights,
                probe_continuous_output=probe_continuous_output,
            )
            hard_spikes.append(output.hard_spikes)
            probabilities.append(output.spike_probability)
            rates.append(output.rates)
            generators.append(output.generator_potential)
        return RGCOutput(
            hard_spikes=torch.stack(hard_spikes, dim=1),
            spike_probability=torch.stack(probabilities, dim=1),
            rates=torch.stack(rates, dim=1),
            generator_potential=torch.stack(generators, dim=1),
        ), state

    def forward(
        self,
        sequence: torch.Tensor,
        state: RetinaState | None = None,
        *,
        spatial_weights: torch.Tensor | None = None,
        probe_continuous_output: bool = False,
    ) -> tuple[RGCOutput, RetinaState]:
        return self.forward_sequence(
            sequence,
            state,
            spatial_weights=spatial_weights,
            probe_continuous_output=probe_continuous_output,
        )


def build_retina_model(
    cone_positions_degs: PositionArray,
    profile: PhysiologyProfile,
    rgc_config: RGCConfig,
) -> RetinaModel:
    return RetinaModel(
        H1HorizontalNetwork(cone_positions_degs, profile.h1),
        BipolarLayer(cone_positions_degs, profile.bipolar),
        LocalAmacrineLayer(cone_positions_degs, profile.amacrine),
        HeterogeneousRGCPool(cone_positions_degs, rgc_config),
    )


def detach_state(state: RetinaState) -> RetinaState:
    return RetinaState(
        h1=state.h1.detach(),
        bipolar=BipolarState(
            output=state.bipolar.output.detach(),
            transient_baseline=state.bipolar.transient_baseline.detach(),
        ),
        amacrine=state.amacrine.detach(),
        rgc=detach_rgc_state(state.rgc),
    )


def state_to_tensors(state: RetinaState) -> tuple[torch.Tensor, ...]:
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


def state_from_tensors(tensors: tuple[torch.Tensor, ...]) -> RetinaState:
    if len(tensors) != 8:
        raise RetinaModelError("Retina state requires eight tensors")
    return RetinaState(
        h1=tensors[0],
        bipolar=BipolarState(tensors[1], tensors[2]),
        amacrine=tensors[3],
        rgc=RGCState(tensors[4], tensors[5], tensors[6], tensors[7]),
    )


__all__ = [
    "RetinaModel",
    "RetinaModelError",
    "RetinaState",
    "build_retina_model",
    "detach_state",
    "state_from_tensors",
    "state_to_tensors",
]
