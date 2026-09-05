from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from evaluation.mechanistic_retina.clean_sampled_data import (
    CleanBenchmarkError,
    CleanBenchmarkState,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import (
    JsonValue,
    bounded_parameter_update_records,
    explicit_delay_bounds,
    explicit_delay_gradients,
    explicit_delay_order_valid,
    explicit_delay_values,
    parameter_update_record,
    pathway_parameter_refs,
    pathway_parameters,
    tau_bounds,
    tau_gradients,
    tau_values,
    validation_nll,
)
from evaluation.model_comparison.parameters import (
    parameter_inventory,
    parameter_snapshot,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import (
    build_phase1_optimizer,
    phase1_parameters,
)


@dataclass(frozen=True, slots=True)
class CleanTrainingEvidence:
    validation_nll: tuple[dict[str, float | int], ...]
    pathway_updates: dict[str, dict[str, float | int | bool]]
    cell_gain_updates: dict[str, float | int | bool]
    tau_updates: dict[str, dict[str, JsonValue]]
    explicit_delay_updates: dict[str, dict[str, JsonValue]]
    optimizer_parameters: tuple[str, ...]
    parameter_inventory: dict[str, int | None]


def train_clean_student(state: CleanBenchmarkState) -> CleanTrainingEvidence:
    model = state.student
    config = state.config
    optimizer = build_phase1_optimizer(model, learning_rate=config.learning_rate)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in optimizer_ids
    )
    expected_taus = {"h1.raw_tau", "feature_bank.raw_tau", "amacrine.raw_tau"}
    if not expected_taus.issubset(optimizer_names):
        raise CleanBenchmarkError(
            "learnable tau parameters are missing from the optimizer"
        )
    expected_delays = {"h1.raw_delay", "feature_bank.raw_delay", "amacrine.raw_delay"}
    if not expected_delays.issubset(optimizer_names):
        raise CleanBenchmarkError(
            "learnable delay parameters are missing from the optimizer"
        )
    initial_parameters = parameter_snapshot(model)
    pathway_initial = pathway_parameters(model)
    tau_initial = tau_values(model)
    delay_initial = explicit_delay_values(model)
    pathway_gradient_seen = {name: False for name in pathway_initial}
    if model.cell_gains is None:
        raise CleanBenchmarkError("Canonical V1 training requires cell-specific gains")
    cell_gain_initial = model.cell_gains.audit_values.detach().flatten().clone()
    cell_gain_gradient_seen = False
    tau_gradient_seen = {name: False for name in tau_initial}
    delay_gradient_seen = {name: False for name in delay_initial}
    generator = torch.Generator().manual_seed(config.training_seed)
    checkpoints = set(config.checkpoint_steps)
    validation = [{"step": 0, "nll": validation_nll(state)}]
    for step in range(1, config.steps + 1):
        stimulus = torch.randint(
            state.train_spikes.shape[0], (config.batch_size,), generator=generator
        )
        trial = torch.randint(
            state.train_spikes.shape[1], (config.batch_size,), generator=generator
        )
        cones = state.train_cones[stimulus]
        spikes = state.train_spikes[stimulus, trial]
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_sequence(cones, observed_counts=spikes).logits
        loss = expected_bernoulli_nll(logits, spikes, torch.ones_like(spikes))
        if not bool(torch.isfinite(loss)):
            raise CleanBenchmarkError("training NLL became non-finite")
        loss.backward()
        for name, parameters in pathway_parameter_refs(model).items():
            finite_nonzero = any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and bool(torch.count_nonzero(parameter.grad))
                for parameter in parameters
            )
            pathway_gradient_seen[name] |= finite_nonzero
        cell_gain_gradient_seen |= any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.cell_gains.raw_parameters
        )
        for name, gradient in tau_gradients(model).items():
            finite_nonzero = (
                gradient is not None
                and bool(torch.isfinite(gradient).all())
                and bool(torch.count_nonzero(gradient))
            )
            tau_gradient_seen[name] |= finite_nonzero
        for name, gradient in explicit_delay_gradients(model).items():
            finite_nonzero = (
                gradient is not None
                and bool(torch.isfinite(gradient).all())
                and bool(torch.count_nonzero(gradient))
            )
            delay_gradient_seen[name] |= finite_nonzero
        optimizer.step()
        model.project_mechanism_parameters()
        if not explicit_delay_order_valid(model):
            raise CleanBenchmarkError("explicit pathway delay ordering became invalid")
        bounded_values = (
            *tau_values(model).values(),
            *explicit_delay_values(model).values(),
        )
        if not all(bool(torch.isfinite(value).all()) for value in bounded_values):
            raise CleanBenchmarkError(
                "learned tau or explicit pathway delay became non-finite"
            )
        if step in checkpoints:
            nll = validation_nll(state)
            if not torch.isfinite(torch.tensor(nll)):
                raise CleanBenchmarkError("validation NLL became non-finite")
            validation.append({"step": step, "nll": nll})
    pathway_final = pathway_parameters(model)
    cell_gain_final = model.cell_gains.audit_values.detach().flatten().clone()
    tau_final = tau_values(model)
    delay_final = explicit_delay_values(model)
    pathway_updates = {
        name: parameter_update_record(
            pathway_initial[name], pathway_final[name], pathway_gradient_seen[name]
        )
        for name in pathway_initial
    }
    cell_gain_updates = parameter_update_record(
        cell_gain_initial, cell_gain_final, cell_gain_gradient_seen
    )
    tau_updates = bounded_parameter_update_records(
        tau_initial, tau_final, tau_bounds(model), tau_gradient_seen
    )
    delay_updates = bounded_parameter_update_records(
        delay_initial,
        delay_final,
        explicit_delay_bounds(model),
        delay_gradient_seen,
    )
    inventory = asdict(
        parameter_inventory(
            model,
            phase1_parameters(model),
            initial_parameters=initial_parameters,
        )
    )
    return CleanTrainingEvidence(
        tuple(validation),
        pathway_updates,
        cell_gain_updates,
        tau_updates,
        delay_updates,
        optimizer_names,
        inventory,
    )


__all__ = ["CleanTrainingEvidence", "train_clean_student"]
