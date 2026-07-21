from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn

from models.cells.amacrine import LocalAmacrineDiagnostics, LocalAmacrineLayer
from models.cells.bipolar import (
    BipolarDiagnostics,
    BipolarLayer,
    BipolarState,
)
from models.cells.horizontal import (
    H1Diagnostics,
    H1HorizontalNetwork,
)
from models.cells.rgc import (
    RGCDiagnostics,
    RGCOutput,
    RGCPopulationLayer,
    RGCPopulationTensors,
    RGCState,
)


class RetinaSNNError(ValueError):
    pass


class RetinaStepDiagnostics(TypedDict):
    h1: H1Diagnostics
    bipolar: BipolarDiagnostics
    amacrine: LocalAmacrineDiagnostics
    rgc: RGCDiagnostics


@dataclass(frozen=True, slots=True)
class RetinaSNNState:
    h1: torch.Tensor
    bipolar: BipolarState
    amacrine: torch.Tensor
    rgc: RGCState


class RetinaSNNCore(nn.Module):
    def __init__(
        self,
        h1: H1HorizontalNetwork,
        bipolar: BipolarLayer,
        amacrine: LocalAmacrineLayer,
        rgc: RGCPopulationLayer,
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
    ) -> RetinaSNNState:
        return RetinaSNNState(
            h1=self.h1.initial_state(batch_size, device, dtype),
            bipolar=self.bipolar.initial_state(batch_size, device, dtype),
            amacrine=self.amacrine.initial_state(batch_size, device, dtype),
            rgc=self.rgc.initial_state(batch_size, device, dtype),
        )

    def step(
        self,
        cone_t: torch.Tensor,
        state: RetinaSNNState,
        return_diagnostics: bool = False,
    ) -> (
        tuple[RGCOutput, RetinaSNNState]
        | tuple[RGCOutput, RetinaSNNState, RetinaStepDiagnostics]
    ):
        if not return_diagnostics:
            cone_mod, h1_state = self.h1(cone_t, state.h1)
            bipolar_state = self.bipolar(
                cone_mod,
                state.bipolar,
                amacrine_prev=state.amacrine,
            )
            amacrine_state = self.amacrine(
                bipolar_state.output,
                state.amacrine,
            )
            rgc_output, rgc_state = self.rgc(
                bipolar_state.output,
                amacrine_state,
                state.rgc,
            )
            return rgc_output, RetinaSNNState(
                h1_state,
                bipolar_state,
                amacrine_state,
                rgc_state,
            )

        cone_mod, h1_state, h1_diagnostics = self.h1(
            cone_t,
            state.h1,
            return_diagnostics=True,
        )
        bipolar_state, bipolar_diagnostics = self.bipolar(
            cone_mod,
            state.bipolar,
            amacrine_prev=state.amacrine,
            return_diagnostics=True,
        )
        amacrine_state, amacrine_diagnostics = self.amacrine(
            bipolar_state.output,
            state.amacrine,
            return_diagnostics=True,
        )
        rgc_output, rgc_state, rgc_diagnostics = self.rgc(
            bipolar_state.output,
            amacrine_state,
            state.rgc,
            return_diagnostics=True,
        )
        next_state = RetinaSNNState(
            h1_state,
            bipolar_state,
            amacrine_state,
            rgc_state,
        )
        diagnostics = RetinaStepDiagnostics(
            h1=h1_diagnostics,
            bipolar=bipolar_diagnostics,
            amacrine=amacrine_diagnostics,
            rgc=rgc_diagnostics,
        )
        return rgc_output, next_state, diagnostics

    def forward_sequence(
        self,
        x_cone: torch.Tensor,
        state: RetinaSNNState | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[RGCOutput, RetinaSNNState]
        | tuple[
            RGCOutput,
            RetinaSNNState,
            tuple[RetinaStepDiagnostics, ...],
        ]
    ):
        cone_count = self.h1.cone_to_h1.shape[1]
        if (
            x_cone.ndim != 3
            or x_cone.shape[1] < 1
            or x_cone.shape[2] != cone_count
        ):
            raise RetinaSNNError("x_cone must have shape [batch,time,Ncone]")
        if state is None:
            state = self.initial_state(
                x_cone.shape[0],
                x_cone.device,
                x_cone.dtype,
            )

        spikes: list[RGCPopulationTensors] = []
        rates: list[RGCPopulationTensors] = []
        diagnostics: list[RetinaStepDiagnostics] = []
        for cone_t in x_cone.unbind(dim=1):
            if return_diagnostics:
                output, state, step_diagnostics = self.step(
                    cone_t,
                    state,
                    return_diagnostics=True,
                )
                diagnostics.append(step_diagnostics)
            else:
                output, state = self.step(cone_t, state)
            spikes.append(output.spikes)
            rates.append(output.rates)

        sequence_output = RGCOutput(
            spikes=_stack_populations(spikes),
            rates=_stack_populations(rates),
        )
        if return_diagnostics:
            return sequence_output, state, tuple(diagnostics)
        return sequence_output, state

    def forward(
        self,
        x_cone: torch.Tensor,
        state: RetinaSNNState | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[RGCOutput, RetinaSNNState]
        | tuple[
            RGCOutput,
            RetinaSNNState,
            tuple[RetinaStepDiagnostics, ...],
        ]
    ):
        return self.forward_sequence(x_cone, state, return_diagnostics)


def detach_state(state: RetinaSNNState) -> RetinaSNNState:
    rgc = state.rgc
    return RetinaSNNState(
        h1=state.h1.detach(),
        bipolar=BipolarState(
            state.bipolar.output.detach(),
            state.bipolar.transient_baseline.detach(),
        ),
        amacrine=state.amacrine.detach(),
        rgc=RGCState(
            membrane=_detach_populations(rgc.membrane),
            adaptation=_detach_populations(rgc.adaptation),
            rate=_detach_populations(rgc.rate),
            subunit_energy=rgc.subunit_energy.detach(),
        ),
    )


def _detach_populations(
    populations: RGCPopulationTensors,
) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        populations.midget.detach(),
        populations.parasol.detach(),
    )


def _stack_populations(
    history: list[RGCPopulationTensors],
) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        midget=torch.stack([step.midget for step in history], dim=1),
        parasol=torch.stack([step.parasol for step in history], dim=1),
    )
