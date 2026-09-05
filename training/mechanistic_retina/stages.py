from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

import torch

from evaluation.candidate0_likelihood_math import (
    StaticTargetRequest,
    build_static_teacher_targets,
    causal_static_drive,
)
from evaluation.mechanistic_retina.rf_base import Candidate0Reference
from evaluation.mechanistic_retina.pathway_decomposition import effective_pathway_rf
from evaluation.mechanistic_retina.rf_base import base_rf
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.mechanistic_retina.metrics import RFMetric, evaluate_rf
from evaluation.phase1_benchmark import BudgetCondition
from evaluation.phase1_data import (
    AssetRequest,
    FINAL_TEST_BOUNDARY,
    build_condition_data,
    prepare_phase1_assets,
)
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.trainer import (
    Phase1TrainingRequest,
    Phase1TrainingResult,
    train_phase1,
)

@unique
class Phase(StrEnum):
    ARCHITECTURE = "P0"
    PROJECTION = "P1"
    MATCHED_CONTROL = "P2"
    PHYSIOLOGY_LIKELIHOOD = "P3"
    OPERATOR_SMOKE = "P4"
    CLEANUP = "P5"


@dataclass(frozen=True, slots=True)
class Phase1Schedule:
    smoke_steps: tuple[int, ...] = (0, 10, 25, 50)
    final_steps: tuple[int, ...] = (0, 50, 100, 200, 400)
    final_step: int = 400


@dataclass(frozen=True, slots=True)
class MechanisticSeedData:
    seed: int
    train_cones: torch.Tensor
    train_probability: torch.Tensor
    train_mask: torch.Tensor
    validation_cones: torch.Tensor
    validation_probability: torch.Tensor
    validation_mask: torch.Tensor
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    cell_ids: tuple[str, ...]
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    final_test_boundary: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SeedDataError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class SeedStageRequest:
    candidate: Candidate0Reference
    seed: int
    steps: int
    checkpoint_steps: tuple[int, ...]
    learning_rate: float = 0.03
    batch_size: int = 8
    analyze_effective: bool = True


@dataclass(frozen=True, slots=True)
class SeedStageResult:
    model: MechanisticGraphTemporalRetina
    data: MechanisticSeedData
    training: Phase1TrainingResult
    raw_metric: RFMetric
    base_metric: RFMetric
    effective_metric: RFMetric | None
    prediction_ce: float
    logit_rmse: float
    pathway_sum_error: float
    current_contributions: tuple[float, float, float, float]
    pathway_rf_names: tuple[str, ...]
    pathway_rf_norms: tuple[float, ...]
    pathway_rf_cosines: tuple[float, ...]


def build_seed_data(seed: int, candidate: Candidate0Reference) -> MechanisticSeedData:
    assets = prepare_phase1_assets(AssetRequest(240819, 56, seed, 91001, 16))
    condition = build_condition_data(
        assets,
        BudgetCondition(112, 8),
        bank_seed=seed,
        teacher_seed=seed,
    )
    if assets.canonical_data.cells.ids != tuple(value.cell_id for value in candidate.metadata):
        raise SeedDataError("Candidate0 and phase-one cell ordering differ")
    train_drive = causal_static_drive(condition.data.train.cone_response, candidate.rf)
    validation_drive = causal_static_drive(assets.validation_split.cone_response, candidate.rf)
    targets = build_static_teacher_targets(
        StaticTargetRequest(
            train_drive,
            validation_drive,
            condition.data.train.valid_mask,
            assets.validation_split.spike_counts.shape[1],
            16,
            -2.0,
        )
    )
    return MechanisticSeedData(
        seed,
        condition.data.train.cone_response,
        targets.train_probabilities,
        condition.data.train.valid_mask,
        assets.validation_split.cone_response,
        targets.validation_probabilities,
        assets.validation_split.valid_mask,
        torch.from_numpy(assets.canonical_data.cone_positions_degs.copy()),
        torch.from_numpy(assets.canonical_data.cells.positions_degs.copy()),
        assets.canonical_data.cells.ids,
        assets.canonical_data.cells.type_ids,
        tuple("ON" if int(value) == 0 else "OFF" for value in assets.canonical_data.cells.polarities),
        FINAL_TEST_BOUNDARY,
    )


def run_seed_stage(request: SeedStageRequest) -> SeedStageResult:
    data = build_seed_data(request.seed, request.candidate)
    torch.manual_seed(request.seed)
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(),
        data.cone_positions,
        data.cell_positions,
        data.cell_types,
        data.polarities,
    )
    raw = evaluate_rf(
        base_rf(model).detach(),
        request.candidate.rf,
        data.cone_positions,
        data.cell_positions,
        request.candidate.metadata,
    )
    training = train_phase1(
        Phase1TrainingRequest(
            model,
            data.train_cones,
            data.train_probability[:, 0],
            data.train_mask[:, 0],
            data.validation_cones,
            data.validation_probability[:, 0],
            data.validation_mask[:, 0],
            request.steps,
            request.checkpoint_steps,
            request.learning_rate,
            request.batch_size,
            request.seed,
        )
    )
    learned_rf = base_rf(model).detach()
    learned = evaluate_rf(
        learned_rf,
        request.candidate.rf,
        data.cone_positions,
        data.cell_positions,
        request.candidate.metadata,
    )
    output = model.forward_sequence(
        data.validation_cones,
        observed_counts=data.validation_probability[:, 0],
    )
    probability = data.validation_probability[:, 0]
    mask = data.validation_mask[:, 0]
    ce = expected_bernoulli_nll(output.logits, probability, mask)
    target_logit = torch.logit(probability.clamp(1e-7, 1 - 1e-7))
    rmse = torch.sqrt(((output.logits - target_logit).square() * mask).sum() / mask.sum())
    contributions = (
        float(output.bc_sustained_current.abs().mean()),
        float(output.bc_transient_current.abs().mean()),
        float(output.amacrine_local_current.abs().mean()),
        float(output.amacrine_transient_current.abs().mean()),
    )
    effective_metric = None
    pathway_error = 0.0
    pathway_names: tuple[str, ...] = ()
    pathway_norms: tuple[float, ...] = ()
    pathway_cosines: tuple[float, ...] = ()
    if request.analyze_effective:
        effective = effective_rf(
            model, data.validation_cones, data.validation_probability[:, 0]
        )
        expanded_teacher = request.candidate.rf.unsqueeze(0).unsqueeze(0).expand(
            6, 2, -1, -1, -1
        )
        effective_metric = evaluate_rf(
            effective.reshape(6, 2, 16, 16, 29),
            expanded_teacher,
            data.cone_positions,
            data.cell_positions,
            request.candidate.metadata,
        )
        pathways = effective_pathway_rf(
            model, data.validation_cones, data.validation_probability[:, 0]
        )
        pathway_sum = sum(
            pathways.values(), torch.zeros_like(next(iter(pathways.values())))
        )
        pathway_error = float((pathway_sum - effective).abs().max())
        pathway_names = tuple(pathways)
        pathway_norms = tuple(float(value.norm()) for value in pathways.values())
        pathway_cosines = tuple(
            float(
                torch.nn.functional.cosine_similarity(
                    value.flatten().double(), expanded_teacher.flatten().double(), dim=0
                )
            )
            for value in pathways.values()
        )
    return SeedStageResult(
        model,
        data,
        training,
        raw,
        learned,
        effective_metric,
        float(ce),
        float(rmse),
        pathway_error,
        contributions,
        pathway_names,
        pathway_norms,
        pathway_cosines,
    )


__all__ = [
    "MechanisticSeedData",
    "Phase",
    "Phase1Schedule",
    "SeedDataError",
    "SeedStageRequest",
    "SeedStageResult",
    "build_seed_data",
    "run_seed_stage",
]
