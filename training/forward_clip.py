from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from loss.retina import RetinaLosses, RetinaObjective
from models.cells.rgc_types import RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel, RetinaState, detach_state
from training.bootstrap import (
    MultiViewBootstrapContext,
    MultiViewBootstrapRuntime,
    MultiViewReadouts,
    apply_multiview_bootstrap,
)
from training.config import ExperimentConfig
from training.schedule import objective_weights
from training.state import BootstrapState, EnergyBudgetState
from training.unroll import ForwardRegionRequest, forward_region


class ForwardClipError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ForwardClipRequest:
    model: RetinaModel
    decoder: TiedLocalDecoder
    objective: RetinaObjective
    config: ExperimentConfig
    reconstruction_scale: float
    optimizer_step: int
    energy_state: EnergyBudgetState
    bootstrap_state: BootstrapState
    noisy_input: torch.Tensor
    clean_target: torch.Tensor
    checkpointed: bool
    full_bptt: bool


def forward_training_clip(
    request: ForwardClipRequest,
) -> tuple[RetinaLosses, RGCOutput, RetinaState]:
    if (
        request.noisy_input.shape != request.clean_target.shape
        or request.noisy_input.ndim != 3
    ):
        raise ForwardClipError("Input and target must match [batch,time,cone]")
    training = request.config.training
    if request.noisy_input.shape[1] != request.config.data.sequence_steps:
        raise ForwardClipError(
            "Clip length does not match data.sequence_steps"
        )
    state = request.model.initial_state(
        request.noisy_input.shape[0],
        request.noisy_input.device,
        torch.float32,
    )
    spatial_weights = request.model.rgc.compute_spatial_weights()
    if request.full_bptt:
        history, state = request.model.forward_sequence(
            request.noisy_input.float(),
            state,
            spatial_weights=spatial_weights,
        )
    else:
        with torch.no_grad():
            _, state = request.model.forward_sequence(
                request.noisy_input[:, : training.burn_in_steps].float(),
                state,
                spatial_weights=spatial_weights,
            )
        state = detach_state(state)
        region = request.noisy_input[:, training.burn_in_steps :].float()
        if region.shape[1] != training.differentiable_steps:
            raise ForwardClipError(
                "Differentiable region length is inconsistent"
            )
        if (
            training.context_only_steps + training.supervised_steps
            != region.shape[1]
        ):
            raise ForwardClipError(
                "Context and supervised regions are inconsistent"
            )
        history, state = forward_region(
            ForwardRegionRequest(
                model=request.model,
                region=region,
                state=state,
                spatial_weights=spatial_weights,
                checkpointed=request.checkpointed,
                block_steps=training.checkpoint_block_steps,
            )
        )
    target_region = (
        request.clean_target.float()
        if request.full_bptt
        else request.clean_target[:, training.burn_in_steps :].float()
    )
    persistent_prediction = request.decoder(history.rates, spatial_weights)
    weights = objective_weights(request.optimizer_step, request.config)
    prediction = persistent_prediction
    bootstrap_auxiliary = persistent_prediction.new_zeros(())
    bootstrap_active = (
        request.model.training
        and torch.is_grad_enabled()
        and not request.full_bptt
        and request.optimizer_step < training.reconstruction_bootstrap_steps
    )
    if bootstrap_active:
        supervised = training.supervised_steps
        application = apply_multiview_bootstrap(
            MultiViewReadouts(
                generator_readout=history.generator_potential[
                    :, -supervised:
                ],
                target=target_region[:, -supervised:],
                persistent_prediction=persistent_prediction[
                    :, -supervised:
                ],
            ),
            MultiViewBootstrapContext(
                reconstruction_scale=request.reconstruction_scale,
                view_consistency_scale=weights.view_consistency_scale,
                generator_variance_weight=weights.variance,
            ),
            MultiViewBootstrapRuntime(
                state=request.bootstrap_state,
                parameters=tuple(request.model.parameters()),
                optimizer_step=request.optimizer_step,
            ),
        )
        prediction = torch.cat(
            (
                persistent_prediction[:, :-supervised].detach(),
                application.prediction,
            ),
            dim=1,
        )
        bootstrap_auxiliary = application.auxiliary_loss
    losses = request.objective(
        prediction,
        target_region,
        history,
        request.model.rgc,
        spatial_weights,
        reconstruction_scale=request.reconstruction_scale,
        energy_budget=request.energy_state.current_budget,
        energy_dual=request.energy_state.dual,
        energy_weight=weights.energy,
        wiring_weight=weights.wiring,
        variance_weight=weights.variance,
        phenotype_repulsion_weight=weights.phenotype_repulsion,
        homeostasis_weight=weights.homeostasis,
        supervised_steps=training.supervised_steps,
    )
    if bootstrap_active:
        losses = replace(
            losses,
            total=losses.total + bootstrap_auxiliary,
        )
    return losses, history, state


__all__ = [
    "ForwardClipError",
    "ForwardClipRequest",
    "forward_training_clip",
]
