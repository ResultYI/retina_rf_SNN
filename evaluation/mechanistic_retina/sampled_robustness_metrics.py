from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.mechanistic_retina.clean_sampled_data import CleanBenchmarkState
from evaluation.mechanistic_retina.noise_free_recovery_metrics import cosine
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll


@dataclass(frozen=True, slots=True)
class SampledSplit:
    cones: torch.Tensor
    spikes: torch.Tensor


def split_data(state: CleanBenchmarkState, *, validation: bool) -> SampledSplit:
    cones = state.validation_cones if validation else state.train_cones
    spikes = state.validation_spikes if validation else state.train_spikes
    stimuli, trials, time_steps, cells = spikes.shape
    repeated_cones = cones[:, None].expand(-1, trials, -1, -1).reshape(
        stimuli * trials, time_steps, -1
    )
    return SampledSplit(
        repeated_cones,
        spikes.reshape(stimuli * trials, time_steps, cells),
    )


def sampled_nll(model: MechanisticGraphTemporalRetina, split: SampledSplit) -> float:
    with torch.no_grad():
        logits = model.forward_sequence(
            split.cones, observed_counts=split.spikes
        ).logits
        return float(
            expected_bernoulli_nll(logits, split.spikes, torch.ones_like(split.spikes))
        )


def teacher_probability_metrics(
    student: MechanisticGraphTemporalRetina,
    teacher: MechanisticGraphTemporalRetina,
    split: SampledSplit,
) -> dict[str, float]:
    with torch.no_grad():
        teacher_probability = teacher.forward_sequence(
            split.cones, observed_counts=split.spikes
        ).spike_probability
        student_logits = student.forward_sequence(
            split.cones, observed_counts=split.spikes
        ).logits
        mask = torch.ones_like(teacher_probability)
        ce = expected_bernoulli_nll(student_logits, teacher_probability, mask)
        entropy = expected_bernoulli_nll(
            torch.logit(teacher_probability.clamp(1e-7, 1.0 - 1e-7)),
            teacher_probability,
            mask,
        )
    return {"expected_ce": float(ce), "kl": float(ce - entropy)}


def counterfactuals(
    model: MechanisticGraphTemporalRetina,
    validation: SampledSplit,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, float]]]:
    probe_cones = validation.cones[:2]
    probe_history = validation.spikes[:2]
    normal_rf = effective_rf(model, probe_cones, probe_history)
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
        normal = model.forward_sequence(
            validation.cones, observed_counts=validation.spikes
        )
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name, pathway_clamps in clamps.items():
        with torch.no_grad():
            clamped = model.forward_sequence(
                validation.cones,
                observed_counts=validation.spikes,
                clamps=pathway_clamps,
            )
        if PathwayClamp.DIRECT_BC_SUSTAINED in pathway_clamps:
            assert torch.count_nonzero(clamped.bc_sustained_current) == 0
            assert torch.count_nonzero(clamped.bc_transient_current) == 0
            assert torch.equal(clamped.bc_broad_presynaptic, normal.bc_broad_presynaptic)
            assert torch.equal(clamped.amacrine_local_current, normal.amacrine_local_current)
            assert torch.equal(clamped.amacrine_transient_current, normal.amacrine_transient_current)
        clamped_rf = effective_rf(
            model, probe_cones, probe_history, clamps=pathway_clamps
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


__all__ = [
    "SampledSplit",
    "counterfactuals",
    "sampled_nll",
    "split_data",
    "teacher_probability_metrics",
]
