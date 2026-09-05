from __future__ import annotations

import torch

from evaluation.mechanistic_retina.clean_sampled_data import CleanBenchmarkState
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll

type JsonScalar = None | bool | int | float | str
type JsonValue = (
    JsonScalar | list["JsonValue"] | tuple["JsonValue", ...] | dict[str, "JsonValue"]
)


def pathway_parameter_refs(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    gain_parameters = () if model.cell_gains is None else model.cell_gains.raw_parameters
    return {
        "H1": (model.gates.raw_h1_amplitude,),
        "BC": (model.bipolar.raw_weights,),
        "AC": (
            model.amacrine.raw_tau,
            model.amacrine.raw_delay,
            model.gates.ac_local,
            model.gates.ac_transient,
            *gain_parameters[len(gain_parameters) // 2 :],
        ),
    }


def pathway_parameters(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor]:
    gains = (
        model.bipolar.raw_weights.new_empty(0)
        if model.cell_gains is None
        else model.cell_gains.audit_values[:, model.cell_gains.audit_values.shape[1] // 2 :].flatten()
    )
    return {
        "H1": model.gates.values(frozenset()).h1.detach().reshape(1).clone(),
        "BC": model.bipolar.positive_weights().detach().flatten().clone(),
        "AC": torch.cat(
            (
                torch.stack((
                    model.gates.values(frozenset()).ac_local,
                    model.gates.values(frozenset()).ac_transient,
                )).flatten(),
                model.amacrine.tau_ms,
                model.amacrine.delay_ms,
                gains,
            )
        ).detach().clone(),
    }


def effective_parameter_values(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor]:
    if model.cell_gains is None:
        raise ValueError("Canonical V1 effective recovery requires cell-specific gains")
    gates = model.gates.values(frozenset())
    gains = model.cell_gains.audit_values.detach()
    return {
        "H1_effective_amplitude": model.gates.h1.detach().reshape(1).clone(),
        "BC_effective_weights": model.bipolar.positive_weights().detach().clone(),
        "AC_effective_gates": torch.stack(
            (gates.ac_local.detach(), gates.ac_transient.detach())
        ),
        "cell_BC_gains": gains[:, 0].clone(),
        "cell_AC_gains": gains[:, 1].clone(),
    }


def effective_parameter_recovery_summary(
    teacher: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
) -> dict[str, JsonValue]:
    return {
        "comparison_space": "effective_normalized_parameters_only",
        "parameters": {
            name: _rf_record(teacher[name], raw[name], trained[name])
            for name in teacher
        },
    }


def parameter_update_record(
    initial: torch.Tensor, final: torch.Tensor, gradient_seen: bool
) -> dict[str, float | int | bool]:
    delta = (final - initial).abs()
    return {
        "gradient_seen": gradient_seen,
        "actually_updated": bool(torch.count_nonzero(delta)),
        "updated_elements": int(torch.count_nonzero(delta)),
        "total_elements": delta.numel(),
        "max_abs_delta": float(delta.max()),
    }


def tau_values(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor]:
    return {
        "H1": model.h1.tau_ms.detach().reshape(1).clone(),
        "BC_sustained_basis": model.feature_bank.tau_ms[0].detach().clone(),
        "BC_transient_basis": model.feature_bank.tau_ms[1].detach().clone(),
        "AC_local_state": model.amacrine.tau_ms[0].detach().reshape(1).clone(),
        "AC_transient_state": model.amacrine.tau_ms[1].detach().reshape(1).clone(),
    }


def explicit_delay_values(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor]:
    return {
        "H1": model.h1.delay_ms.detach().reshape(1).clone(),
        "BC_sustained": model.feature_bank.delay_ms[0].detach().reshape(1).clone(),
        "BC_transient": model.feature_bank.delay_ms[1].detach().reshape(1).clone(),
        "AC_local_downstream": model.amacrine.delay_ms[0].detach().reshape(1).clone(),
        "AC_transient_downstream": model.amacrine.delay_ms[1].detach().reshape(1).clone(),
    }


def tau_bounds(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor]:
    return {
        "H1": model.h1.tau_bounds_ms.detach().reshape(1, 2),
        "BC_sustained_basis": model.feature_bank.tau_bounds_ms[0].detach(),
        "BC_transient_basis": model.feature_bank.tau_bounds_ms[1].detach(),
        "AC_local_state": model.amacrine.tau_bounds_ms[0].detach().reshape(1, 2),
        "AC_transient_state": model.amacrine.tau_bounds_ms[1].detach().reshape(1, 2),
    }


def explicit_delay_bounds(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor]:
    bounds = model.feature_bank.delay_bounds_ms.detach()
    ac_bounds = model.amacrine.delay_bounds_ms.detach()
    return {
        "H1": model.h1.delay_bounds_ms.detach().reshape(1, 2),
        "BC_sustained": bounds[0].reshape(1, 2),
        "BC_transient": bounds[1].reshape(1, 2),
        "AC_local_downstream": ac_bounds[0].reshape(1, 2),
        "AC_transient_downstream": ac_bounds[1].reshape(1, 2),
    }


def tau_gradients(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor | None]:
    path_gradient = model.feature_bank.raw_tau.grad
    amacrine_gradient = model.amacrine.raw_tau.grad
    return {
        "H1": model.h1.raw_tau.grad,
        "BC_sustained_basis": None if path_gradient is None else path_gradient[0],
        "BC_transient_basis": None if path_gradient is None else path_gradient[1],
        "AC_local_state": None if amacrine_gradient is None else amacrine_gradient[0],
        "AC_transient_state": (
            None if amacrine_gradient is None else amacrine_gradient[1]
        ),
    }


def explicit_delay_gradients(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor | None]:
    path_gradient = model.feature_bank.raw_delay.grad
    ac_gradient = model.amacrine.raw_delay.grad
    return {
        "H1": model.h1.raw_delay.grad,
        "BC_sustained": None if path_gradient is None else path_gradient[0:1],
        "BC_transient": None if path_gradient is None else path_gradient[1:2],
        "AC_local_downstream": None if ac_gradient is None else ac_gradient[0:1],
        "AC_transient_downstream": None if ac_gradient is None else ac_gradient[1:2],
    }


def explicit_delay_order_valid(model: MechanisticGraphTemporalRetina) -> bool:
    delays = model.feature_bank.delay_ms.detach()
    ac_delays = model.amacrine.delay_ms.detach()
    return bool(delays[0] > delays[1] and ac_delays[0] > ac_delays[1])


def bounded_parameter_update_records(
    initial: dict[str, torch.Tensor],
    final: dict[str, torch.Tensor],
    bounds: dict[str, torch.Tensor],
    gradient_seen: dict[str, bool],
) -> dict[str, dict[str, JsonValue]]:
    records: dict[str, dict[str, JsonValue]] = {}
    for name in initial:
        lower = bounds[name][..., 0]
        upper = bounds[name][..., 1]
        delta = final[name] - initial[name]
        margin = torch.minimum(final[name] - lower, upper - final[name])
        hit_boundary = bool(
            torch.isclose(final[name], lower, atol=1e-4, rtol=0).any()
            or torch.isclose(final[name], upper, atol=1e-4, rtol=0).any()
        )
        records[name] = {
            "initial_ms": initial[name].tolist(),
            "trained_ms": final[name].tolist(),
            "lower_ms": lower.tolist(),
            "upper_ms": upper.tolist(),
            "gradient_seen": gradient_seen[name],
            "actually_updated": bool(torch.count_nonzero(delta)),
            "max_abs_delta_ms": float(delta.abs().max()),
            "minimum_boundary_margin_ms": float(margin.min()),
            "hit_boundary": hit_boundary,
            "finite": bool(torch.isfinite(final[name]).all()),
        }
    return records


def bounded_parameter_learning_confirmed(
    records: dict[str, dict[str, JsonValue]],
) -> bool:
    return bool(records) and all(
        record["gradient_seen"] is True
        and record["actually_updated"] is True
        and record["finite"] is True
        for record in records.values()
    )


def validation_nll(state: CleanBenchmarkState) -> float:
    model = state.student
    trials = state.validation_spikes.shape[1]
    cones = state.validation_cones[:, None].expand(-1, trials, -1, -1).flatten(0, 1)
    spikes = state.validation_spikes.flatten(0, 1)
    with torch.no_grad():
        logits = model.forward_sequence(cones, observed_counts=spikes).logits
        return float(expected_bernoulli_nll(logits, spikes, torch.ones_like(spikes)))


def rf_bundle(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    spikes: torch.Tensor,
    *,
    include_pathways: bool = True,
) -> dict[str, torch.Tensor]:
    global_rf = effective_rf(model, cones, spikes)
    bundle = {"global": global_rf, "temporal": global_rf.sum(dim=-1)}
    if include_pathways:
        h1_off = effective_rf(
            model, cones, spikes, clamps=frozenset({PathwayClamp.H1})
        )
        bc = effective_rf(
            model,
            cones,
            spikes,
            clamps=frozenset(
                {
                    PathwayClamp.H1,
                    PathwayClamp.AMACRINE_LOCAL,
                    PathwayClamp.AMACRINE_TRANSIENT,
                }
            ),
        )
        bundle.update({"BC": bc, "AC": h1_off - bc, "H1": global_rf - h1_off})
    return bundle


def rf_summary(
    teacher: dict[str, torch.Tensor],
    raw: dict[str, torch.Tensor],
    trained: dict[str, torch.Tensor],
) -> dict[str, JsonValue]:
    summary: dict[str, JsonValue] = {
        "global": _rf_record(teacher["global"], raw["global"], trained["global"]),
        "temporal": _rf_record(
            teacher["temporal"], raw["temporal"], trained["temporal"]
        ),
    }
    if all(
        name in teacher and name in raw and name in trained
        for name in ("H1", "BC", "AC")
    ):
        summary["pathways"] = {
            name: _rf_record(teacher[name], raw[name], trained[name])
            for name in ("H1", "BC", "AC")
        }
    return summary


def tensor_change_record(
    initial: torch.Tensor, final: torch.Tensor
) -> dict[str, float]:
    difference = final - initial
    return {
        "initial_norm": float(initial.norm()),
        "trained_norm": float(final.norm()),
        "cosine": _cosine(initial, final),
        "difference_norm": float(difference.norm()),
        "mean_absolute_change": float(difference.abs().mean()),
    }


def _rf_record(
    teacher: torch.Tensor, raw: torch.Tensor, trained: torch.Tensor
) -> dict[str, float]:
    return {
        "teacher_norm": float(teacher.norm()),
        "raw_norm": float(raw.norm()),
        "trained_norm": float(trained.norm()),
        "raw_teacher_cosine": _cosine(raw, teacher),
        "trained_teacher_cosine": _cosine(trained, teacher),
        "raw_to_trained_change_norm": float((trained - raw).norm()),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0:
        return 0.0
    return float((left.flatten() @ right.flatten()) / denominator)


__all__ = [
    "JsonValue",
    "bounded_parameter_learning_confirmed",
    "bounded_parameter_update_records",
    "explicit_delay_bounds",
    "explicit_delay_gradients",
    "explicit_delay_order_valid",
    "explicit_delay_values",
    "effective_parameter_recovery_summary",
    "effective_parameter_values",
    "parameter_update_record",
    "pathway_parameter_refs",
    "pathway_parameters",
    "rf_bundle",
    "rf_summary",
    "tau_bounds",
    "tau_gradients",
    "tau_values",
    "tensor_change_record",
    "validation_nll",
]
