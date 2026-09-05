from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch.optim import Optimizer

from models.mechanistic_retina.contracts import MECHANISTIC_MODEL_REVISION
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.optimizer import build_phase1_optimizer, phase1_parameters


CHECKPOINT_SCHEMA: Final = "mechanistic_graph_temporal_retina"
CHECKPOINT_REVISION: Final = MECHANISTIC_MODEL_REVISION
_CHECKPOINT_KEYS: Final = {
    "schema",
    "revision",
    "seed",
    "step",
    "model",
    "optimizer",
}


@dataclass(frozen=True, slots=True)
class CheckpointState:
    seed: int
    step: int


@dataclass(frozen=True, slots=True)
class MatchedControlFit:
    coefficients: torch.Tensor
    bias: torch.Tensor
    logits: torch.Tensor
    ranks: tuple[int, ...]
    converged: bool


@dataclass(frozen=True, slots=True)
class Phase1TrainingRequest:
    model: MechanisticGraphTemporalRetina
    train_cones: torch.Tensor
    train_probability: torch.Tensor
    train_mask: torch.Tensor
    validation_cones: torch.Tensor
    validation_probability: torch.Tensor
    validation_mask: torch.Tensor
    steps: int
    checkpoint_steps: tuple[int, ...]
    learning_rate: float
    batch_size: int
    seed: int


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    step: int
    train_ce: float
    validation_ce: float


@dataclass(frozen=True, slots=True)
class Phase1TrainingResult:
    checkpoints: tuple[TrainingPoint, ...]
    gradients_finite: bool


@dataclass(frozen=True, slots=True)
class MechanisticCheckpointError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def fit_signed_control(
    design: torch.Tensor,
    teacher_probability: torch.Tensor,
    valid_mask: torch.Tensor,
) -> MatchedControlFit:
    if design.shape[:-1] != teacher_probability.shape or valid_mask.shape != teacher_probability.shape:
        raise MechanisticCheckpointError("matched-control tensors have incompatible shapes")
    coefficients = []
    biases = []
    ranks = []
    for cell in range(design.shape[-2]):
        active = valid_mask[..., cell].reshape(-1)
        matrix = design[..., cell, :].reshape(-1, design.shape[-1])[active].double()
        ones = torch.ones((matrix.shape[0], 1), dtype=matrix.dtype, device=matrix.device)
        augmented = torch.cat((matrix, ones), dim=1)
        targets = torch.logit(
            teacher_probability[..., cell].reshape(-1)[active].double().clamp(1e-7, 1 - 1e-7)
        )
        solution = torch.linalg.lstsq(augmented, targets, driver="gelsd")
        coefficients.append(solution.solution[:-1])
        biases.append(solution.solution[-1])
        ranks.append(int(solution.rank))
    stacked = torch.stack(coefficients)
    bias = torch.stack(biases)
    logits = torch.einsum("...nf,nf->...n", design.double(), stacked) + bias
    converged = bool(torch.isfinite(stacked).all() and torch.isfinite(logits).all())
    return MatchedControlFit(stacked, bias, logits, tuple(ranks), converged)


def signed_path_design(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
) -> torch.Tensor:
    features = model.pathway_basis_features(cones)
    return features.reshape(*features.shape[:3], 24)


def matched_control_rf(
    model: MechanisticGraphTemporalRetina,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    basis = model.pathway_basis_rfs()
    flattened = basis.reshape(basis.shape[0], 24, *basis.shape[-2:])
    return (flattened * coefficients[:, :, None, None]).sum(dim=1)


def train_phase1(request: Phase1TrainingRequest) -> Phase1TrainingResult:
    torch.manual_seed(request.seed)
    optimizer = build_phase1_optimizer(
        request.model, learning_rate=request.learning_rate
    )
    generator = torch.Generator().manual_seed(request.seed + 1000)
    checkpoints = [_training_point(request, 0)]
    gradients_finite = True
    for step in range(1, request.steps + 1):
        indices = torch.randint(
            request.train_cones.shape[0],
            (request.batch_size,),
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        output = request.model.forward_sequence(
            request.train_cones[indices],
            observed_counts=request.train_probability[indices],
        )
        loss = expected_bernoulli_nll(
            output.logits,
            request.train_probability[indices],
            request.train_mask[indices],
        )
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in phase1_parameters(request.model)
        )
        optimizer.step()
        request.model.project_mechanism_parameters()
        if step in request.checkpoint_steps:
            checkpoints.append(_training_point(request, step))
    return Phase1TrainingResult(tuple(checkpoints), gradients_finite)


def _training_point(request: Phase1TrainingRequest, step: int) -> TrainingPoint:
    with torch.no_grad():
        train = request.model.forward_sequence(
            request.train_cones,
            observed_counts=request.train_probability,
        )
        validation = request.model.forward_sequence(
            request.validation_cones,
            observed_counts=request.validation_probability,
        )
        train_ce = expected_bernoulli_nll(
            train.logits, request.train_probability, request.train_mask
        )
        validation_ce = expected_bernoulli_nll(
            validation.logits,
            request.validation_probability,
            request.validation_mask,
        )
    return TrainingPoint(step, float(train_ce), float(validation_ce))


def save_checkpoint(
    path: Path,
    model: MechanisticGraphTemporalRetina,
    optimizer: Optimizer,
    *,
    seed: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "revision": CHECKPOINT_REVISION,
            "seed": seed,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MechanisticGraphTemporalRetina,
    optimizer: Optimizer,
) -> CheckpointState:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != _CHECKPOINT_KEYS
        or payload["schema"] != CHECKPOINT_SCHEMA
        or payload["revision"] != CHECKPOINT_REVISION
    ):
        raise MechanisticCheckpointError("mechanistic checkpoint schema is invalid")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return CheckpointState(int(payload["seed"]), int(payload["step"]))


__all__ = [
    "CHECKPOINT_REVISION",
    "CHECKPOINT_SCHEMA",
    "CheckpointState",
    "MatchedControlFit",
    "Phase1TrainingRequest",
    "Phase1TrainingResult",
    "TrainingPoint",
    "MechanisticCheckpointError",
    "load_checkpoint",
    "fit_signed_control",
    "matched_control_rf",
    "save_checkpoint",
    "signed_path_design",
    "train_phase1",
]
