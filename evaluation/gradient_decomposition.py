from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from data.rgc_response import ResponseTargetKind
from evaluation.gradient_optimizer import (
    AdamStateAudit,
    GradientDecompositionError,
    GradientSegmentAudit,
    NamedNorm,
    NamedParameters,
    TensorValues,
    TypeVector,
    adam_moments,
    differential,
    effective_updates,
    group_norms,
    tensor_norm,
    type_vectors,
)
from loss.rgc_response import response_nll
from training.response_data import ResponseSplit, masked_history_counts
from training.response_trainer import ResponseTrainer
from training.response_unroll import ResponseUnrollRequest, unroll_response


@dataclass(frozen=True, slots=True)
class GradientDecomposition:
    source_id: str
    context_ids: tuple[str, str]
    probe_steps: int
    optimizer_step: int
    context: GradientSegmentAudit
    probe: GradientSegmentAudit
    total: GradientSegmentAudit
    adam_state: AdamStateAudit


@dataclass(frozen=True, slots=True)
class _MatchedBatch:
    cones: torch.Tensor
    counts: torch.Tensor
    valid_mask: torch.Tensor
    source_id: str
    context_ids: tuple[str, str]


@dataclass(frozen=True, slots=True)
class _LikelihoodInputs:
    logits: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor
    target_kind: ResponseTargetKind

    def loss(self, window: slice) -> torch.Tensor:
        return response_nll(
            self.logits[:, window],
            self.targets[:, window],
            self.valid_mask[:, window],
            self.target_kind,
        )


def audit_gradient_decomposition(
    trainer: ResponseTrainer,
    probe_steps: int,
) -> GradientDecomposition:
    differentiable_steps = trainer.config.training.differentiable_steps
    if not 1 <= probe_steps < differentiable_steps:
        raise GradientDecompositionError(
            "probe_steps must leave non-empty context and probe windows"
        )
    batch = _matched_context_batch(trainer.data.train, trainer.device)
    trainer.model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    output, _ = unroll_response(
        ResponseUnrollRequest(
            trainer.model,
            batch.cones,
            masked_history_counts(batch.counts, batch.valid_mask),
            trainer.config.training.burn_in_steps,
            differentiable_steps,
            trainer.config.training.checkpoint_block_steps,
            False,
        )
    )
    burn_in = trainer.config.training.burn_in_steps
    likelihood = _LikelihoodInputs(
        output.spike_logits,
        batch.counts[:, burn_in:],
        batch.valid_mask[:, burn_in:],
        trainer.data.target_kind,
    )
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    )
    losses = (
        likelihood.loss(slice(None, -probe_steps)),
        likelihood.loss(slice(-probe_steps, None)),
        likelihood.loss(slice(None)),
    )
    segments = []
    for index, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            tuple(parameter for _, parameter in named_parameters),
            retain_graph=index < len(losses) - 1,
            allow_unused=True,
        )
        filled_gradients = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for (_, parameter), gradient in zip(
                named_parameters,
                gradients,
                strict=True,
            )
        )
        updates = effective_updates(
            trainer,
            named_parameters,
            filled_gradients,
        )
        segments.append(
            _segment_audit(
                loss,
                named_parameters,
                filled_gradients,
                updates,
                tuple(sorted(set(trainer.data.cells.type_ids))),
            )
        )
    exp_avg, exp_avg_sq = adam_moments(trainer, named_parameters)
    type_ids = tuple(sorted(set(trainer.data.cells.type_ids)))
    return GradientDecomposition(
        source_id=batch.source_id,
        context_ids=batch.context_ids,
        probe_steps=probe_steps,
        optimizer_step=trainer.optimizer_step,
        context=segments[0],
        probe=segments[1],
        total=segments[2],
        adam_state=AdamStateAudit(
            exp_avg_norm=tensor_norm(exp_avg),
            exp_avg_sq_norm=tensor_norm(exp_avg_sq),
            exp_avg_group_norms=group_norms(named_parameters, exp_avg),
            exp_avg_sq_group_norms=group_norms(named_parameters, exp_avg_sq),
            exp_avg_type_vectors=type_vectors(named_parameters, exp_avg, type_ids),
            exp_avg_sq_type_vectors=type_vectors(
                named_parameters,
                exp_avg_sq,
                type_ids,
            ),
        ),
    )


def write_gradient_decomposition(
    path: str | Path,
    result: GradientDecomposition,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _matched_context_batch(split: ResponseSplit, device: torch.device) -> _MatchedBatch:
    grouped: dict[str, dict[str, int]] = {}
    for index, (source_id, context_id) in enumerate(
        zip(split.source_ids, split.context_ids, strict=True)
    ):
        grouped.setdefault(source_id, {}).setdefault(context_id, index)
    for source_id, contexts in grouped.items():
        if len(contexts) < 2:
            continue
        context_ids = tuple(sorted(contexts)[:2])
        first, second = (contexts[context_id] for context_id in context_ids)
        indices = torch.tensor((first, second), dtype=torch.long)
        return _MatchedBatch(
            split.cone_response.index_select(0, indices).to(device),
            split.spike_counts.index_select(0, indices)[:, 0].to(device),
            split.valid_mask.index_select(0, indices)[:, 0].to(device),
            source_id,
            context_ids,
        )
    raise GradientDecompositionError("No matched low/high context pair was found")


def _segment_audit(
    loss: torch.Tensor,
    named_parameters: NamedParameters,
    gradients: TensorValues,
    updates: TensorValues,
    type_ids: tuple[str, ...],
) -> GradientSegmentAudit:
    raw_types = type_vectors(named_parameters, gradients, type_ids)
    update_types = type_vectors(named_parameters, updates, type_ids)
    return GradientSegmentAudit(
        likelihood=float(loss.detach()),
        raw_gradient_norm=tensor_norm(gradients),
        raw_group_norms=group_norms(named_parameters, gradients),
        raw_type_vectors=raw_types,
        raw_type_differential=differential(raw_types),
        effective_update_norm=tensor_norm(updates),
        effective_group_norms=group_norms(named_parameters, updates),
        effective_type_vectors=update_types,
        effective_type_differential=differential(update_types),
    )


__all__ = [
    "AdamStateAudit",
    "GradientDecomposition",
    "GradientDecompositionError",
    "GradientSegmentAudit",
    "NamedNorm",
    "TypeVector",
    "audit_gradient_decomposition",
    "write_gradient_decomposition",
]
