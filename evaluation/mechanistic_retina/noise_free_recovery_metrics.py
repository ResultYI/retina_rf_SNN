from __future__ import annotations

import torch

from evaluation.mechanistic_retina.clean_sampled_reporting import (
    explicit_delay_values,
    tau_values,
)
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return 0.0 if float(denominator) == 0.0 else float(
        torch.dot(left.flatten(), right.flatten()) / denominator
    )


def comparison(reference: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    difference = value - reference
    return {
        "teacher_norm": float(torch.linalg.vector_norm(reference)),
        "student_norm": float(torch.linalg.vector_norm(value)),
        "cosine": cosine(reference, value),
        "difference_norm": float(torch.linalg.vector_norm(difference)),
        "mean_absolute_error": float(difference.abs().mean()),
        "maximum_absolute_error": float(difference.abs().max()),
    }


def probability(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    cells = model.rgc.response_bias.numel()
    observed = (
        cones.new_zeros((*cones.shape[:2], cells))
        if history is None
        else history
    )
    with torch.no_grad():
        return model.forward_sequence(
            cones, observed_counts=observed
        ).spike_probability.detach()


def expected_metrics(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    history = torch.zeros_like(target)
    with torch.no_grad():
        output = model.forward_sequence(cones, observed_counts=history)
        mask = torch.ones_like(target)
        ce = expected_bernoulli_nll(output.logits, target, mask)
        teacher_logits = torch.logit(target.clamp(1e-7, 1.0 - 1e-7))
        entropy = expected_bernoulli_nll(teacher_logits, target, mask)
    return {"expected_ce": float(ce), "kl": float(ce - entropy)}


def temporal_values(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, dict[str, list[float]]]:
    return {
        "tau_ms": {
            name: value.tolist() for name, value in tau_values(model).items()
        },
        "explicit_delay_ms": {
            name: value.tolist()
            for name, value in explicit_delay_values(model).items()
        },
    }


def counterfactuals(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    normal_rf: torch.Tensor,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, float]]]:
    cells = model.rgc.response_bias.numel()
    history = cones.new_zeros((*cones.shape[:2], cells))
    clamps = {
        "H1_off": frozenset({PathwayClamp.H1}),
        "direct_BC_off": frozenset(
            {PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}
        ),
        "AC_off": frozenset(
            {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
        ),
    }
    with torch.no_grad():
        normal = model.forward_sequence(cones, observed_counts=history)
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name, pathway_clamps in clamps.items():
        with torch.no_grad():
            clamped = model.forward_sequence(
                cones, observed_counts=history, clamps=pathway_clamps
            )
        if PathwayClamp.DIRECT_BC_SUSTAINED in pathway_clamps:
            assert torch.count_nonzero(clamped.bc_sustained_current) == 0
            assert torch.count_nonzero(clamped.bc_transient_current) == 0
            assert torch.equal(clamped.bc_broad_presynaptic, normal.bc_broad_presynaptic)
            assert torch.equal(clamped.amacrine_local_current, normal.amacrine_local_current)
            assert torch.equal(clamped.amacrine_transient_current, normal.amacrine_transient_current)
        clamped_rf = effective_rf(
            model, cones[:2], history[:2], clamps=pathway_clamps
        )
        logit_delta = clamped.logits - normal.logits
        probability_delta = clamped.spike_probability - normal.spike_probability
        rf_delta = clamped_rf - normal_rf
        tensors[name] = {
            "logit_delta": logit_delta.detach(),
            "probability_delta": probability_delta.detach(),
            "rf_delta": rf_delta.detach(),
        }
        summary[name] = {
            "mean_absolute_logit_change": float(logit_delta.abs().mean()),
            "mean_absolute_probability_change": float(
                probability_delta.abs().mean()
            ),
            "rf_change_norm": float(torch.linalg.vector_norm(rf_delta)),
            "rf_cosine_to_normal": cosine(clamped_rf, normal_rf),
        }
    return tensors, summary


def counterfactual_comparison(
    teacher: dict[str, dict[str, torch.Tensor]],
    student: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        pathway: {
            name: comparison(teacher[pathway][name], student[pathway][name])
            for name in ("logit_delta", "probability_delta", "rf_delta")
        }
        for pathway in teacher
    }


__all__ = [
    "comparison",
    "counterfactual_comparison",
    "counterfactuals",
    "expected_metrics",
    "probability",
    "temporal_values",
]
