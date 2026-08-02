from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.gradient_optimizer import (
    NamedNorm,
    NamedParameters,
    TensorValues,
    TypeVector,
    differential,
    group_norms,
    tensor_norm,
    type_vectors,
)
from evaluation.parameter_audit import TypeDifferential
from loss.rgc_response import response_nll
from training.response_data import masked_history_counts
from training.response_trainer import ResponseTrainer
from training.response_unroll import ResponseUnrollRequest, unroll_response


class TargetGradientComparisonError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TargetGradientSummary:
    likelihood: float
    raw_gradient_norm: float
    group_norms: tuple[NamedNorm, ...]
    type_vectors: tuple[TypeVector, ...]
    type_differential: TypeDifferential


@dataclass(frozen=True, slots=True)
class FullBatchTargetGradientComparison:
    source_count: int
    trial_count: int
    sequence_count: int
    probe_steps: int
    hard: TargetGradientSummary
    soft: TargetGradientSummary
    soft_to_hard_norm_ratio: float
    soft_to_hard_type_separation_ratio: float
    hard_soft_type_vector_cosines: tuple[NamedNorm, ...]


def compare_full_batch_target_gradients(
    trainer: ResponseTrainer,
    soft_targets: torch.Tensor,
    *,
    probe_steps: int,
) -> FullBatchTargetGradientComparison:
    split = trainer.data.validation
    if soft_targets.shape != split.spike_counts.shape:
        raise TargetGradientComparisonError(
            "Soft targets must match validation spike tensor shape"
        )
    differentiable_steps = trainer.config.training.differentiable_steps
    if not 1 <= probe_steps <= differentiable_steps:
        raise TargetGradientComparisonError("probe_steps is outside the likelihood window")
    stimulus_count, trial_count, _, _ = split.spike_counts.shape
    cones = split.cone_response[:, None].expand(
        -1,
        trial_count,
        -1,
        -1,
    ).reshape(stimulus_count * trial_count, *split.cone_response.shape[1:])
    counts = split.spike_counts.reshape(
        stimulus_count * trial_count,
        *split.spike_counts.shape[2:],
    )
    mask = split.valid_mask.reshape(
        stimulus_count * trial_count,
        *split.valid_mask.shape[2:],
    )
    soft = soft_targets.reshape_as(split.spike_counts).reshape_as(counts)
    cones = cones.to(trainer.device)
    counts = counts.to(trainer.device)
    mask = mask.to(trainer.device)
    soft = soft.to(trainer.device)
    trainer.model.eval()
    trainer.optimizer.zero_grad(set_to_none=True)
    output, _ = unroll_response(
        ResponseUnrollRequest(
            trainer.model,
            cones,
            masked_history_counts(counts, mask),
            trainer.config.training.burn_in_steps,
            differentiable_steps,
            trainer.config.training.checkpoint_block_steps,
            False,
        )
    )
    burn_in = trainer.config.training.burn_in_steps
    window = slice(-probe_steps, None)
    logits = output.spike_logits[:, window]
    likelihood_mask = mask[:, burn_in:][:, window]
    losses = (
        response_nll(
            logits,
            counts[:, burn_in:][:, window],
            likelihood_mask,
            trainer.data.target_kind,
        ),
        response_nll(
            logits,
            soft[:, burn_in:][:, window],
            likelihood_mask,
            trainer.data.target_kind,
        ),
    )
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    )
    type_ids = tuple(sorted(set(trainer.data.cells.type_ids)))
    summaries = []
    for index, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            tuple(parameter for _, parameter in named_parameters),
            retain_graph=index == 0,
            allow_unused=True,
        )
        filled = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for (_, parameter), gradient in zip(
                named_parameters,
                gradients,
                strict=True,
            )
        )
        summaries.append(
            summarize_target_gradients(loss, named_parameters, filled, type_ids)
        )
    hard, soft_summary = summaries
    return FullBatchTargetGradientComparison(
        source_count=stimulus_count // 2,
        trial_count=trial_count,
        sequence_count=stimulus_count * trial_count,
        probe_steps=probe_steps,
        hard=hard,
        soft=soft_summary,
        soft_to_hard_norm_ratio=soft_summary.raw_gradient_norm
        / max(hard.raw_gradient_norm, 1e-12),
        soft_to_hard_type_separation_ratio=soft_summary.type_differential.separation_ratio
        / max(hard.type_differential.separation_ratio, 1e-12),
        hard_soft_type_vector_cosines=_type_cosines(
            hard.type_vectors,
            soft_summary.type_vectors,
        ),
    )


def summarize_target_gradients(
    loss: torch.Tensor,
    named_parameters: NamedParameters,
    gradients: TensorValues,
    type_ids: tuple[str, ...],
) -> TargetGradientSummary:
    vectors = type_vectors(named_parameters, gradients, type_ids)
    return TargetGradientSummary(
        float(loss.detach()),
        tensor_norm(gradients),
        group_norms(named_parameters, gradients),
        vectors,
        differential(vectors),
    )


def _type_cosines(
    hard: tuple[TypeVector, ...],
    soft: tuple[TypeVector, ...],
) -> tuple[NamedNorm, ...]:
    return tuple(
        NamedNorm(
            hard_vector.type_id,
            float(
                torch.nn.functional.cosine_similarity(
                    torch.tensor(hard_vector.values).unsqueeze(0),
                    torch.tensor(soft_vector.values).unsqueeze(0),
                )[0]
            ),
        )
        for hard_vector, soft_vector in zip(hard, soft, strict=True)
    )


__all__ = [
    "FullBatchTargetGradientComparison",
    "TargetGradientComparisonError",
    "TargetGradientSummary",
    "compare_full_batch_target_gradients",
    "summarize_target_gradients",
]
