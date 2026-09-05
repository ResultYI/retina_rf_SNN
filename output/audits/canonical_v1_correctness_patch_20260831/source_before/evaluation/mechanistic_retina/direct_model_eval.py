from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping

import torch

from evaluation.mechanistic_retina.direct_metrics import (
    DirectRFSummary,
    PredictionSummary,
    prediction_payload,
    prediction_summary,
    rf_payload,
    rf_summary,
)
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.pathway_decomposition import effective_pathway_rf
from evaluation.mechanistic_retina.rf_base import Candidate0Reference, base_rf
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina
from training.mechanistic_retina.sampled import (
    SampledTrainingRequest,
    predict_sampled_model,
    train_sampled_model,
)
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.stages import MechanisticSeedData


@dataclass(frozen=True, slots=True)
class ModelEvidence:
    name: str
    prediction: PredictionSummary
    rf: DirectRFSummary
    base: DirectRFSummary
    rf_tensor: torch.Tensor
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class MechanisticRunEvidence:
    trained: ModelEvidence
    raw: ModelEvidence
    pathway_sum_error: float
    current_sum_error: float
    gradients_finite: bool
    checkpoints: tuple[Mapping[str, JsonValue], ...]
    parameter_count: int


def build_model(data: MechanisticSeedData, seed: int) -> MechanisticGraphTemporalRetina:
    torch.manual_seed(seed)
    return build_mechanistic_retina(
        MechanisticRetinaConfig(),
        data.cone_positions,
        data.cell_positions,
        data.cell_types,
        data.polarities,
    )


def run_mechanistic_sampled(
    data: MechanisticSeedData,
    candidate: Candidate0Reference,
    train_spikes: torch.Tensor,
    validation_spikes: torch.Tensor,
    *,
    model_seed: int,
    steps: int,
    checkpoints: tuple[int, ...],
    learning_rate: float,
    batch_size: int,
) -> MechanisticRunEvidence:
    model = build_model(data, model_seed)
    train_mask = _expanded_mask(data.train_mask[:, 0], train_spikes.shape[1])
    validation_mask = _expanded_mask(
        data.validation_mask[:, 0], validation_spikes.shape[1]
    )
    raw = evaluate_model(
        "raw-physiological-prior",
        model,
        data,
        candidate,
        validation_spikes,
        validation_mask,
    )
    training = train_sampled_model(
        SampledTrainingRequest(
            model,
            data.train_cones,
            train_spikes,
            train_mask,
            data.validation_cones,
            validation_spikes,
            validation_mask,
            data.validation_probability[:, 0],
            steps,
            checkpoints,
            learning_rate,
            batch_size,
            model_seed,
        )
    )
    trained = evaluate_model(
        "mechanistic-physiology",
        model,
        data,
        candidate,
        validation_spikes,
        validation_mask,
    )
    cones = data.validation_cones[:2]
    history = validation_spikes[:2, 0]
    total_rf = effective_rf(model, cones, history)
    pathways = effective_pathway_rf(model, cones, history)
    summed_rf = sum(pathways.values(), torch.zeros_like(total_rf))
    pathway_error = float((summed_rf - total_rf).abs().max())
    with torch.no_grad():
        output = model.forward_sequence(cones, observed_counts=history)
        current_sum = (
            output.bc_sustained_current
            + output.bc_transient_current
            + output.amacrine_local_current
            + output.amacrine_transient_current
        )
        current_error = float((current_sum - output.total_current).abs().max())
    checkpoint_payload = tuple(
        {key: value for key, value in asdict(point).items()}
        for point in training.checkpoints
    )
    return MechanisticRunEvidence(
        trained,
        raw,
        pathway_error,
        current_error,
        training.gradients_finite,
        checkpoint_payload,
        sum(parameter.numel() for parameter in phase1_parameters(model)),
    )


def evaluate_model(
    name: str,
    model: MechanisticGraphTemporalRetina,
    data: MechanisticSeedData,
    candidate: Candidate0Reference,
    validation_spikes: torch.Tensor,
    validation_mask: torch.Tensor,
) -> ModelEvidence:
    logits = predict_sampled_model(model, data.validation_cones, validation_spikes)
    prediction = prediction_summary(
        logits,
        validation_spikes,
        validation_mask,
        data.validation_probability[:, 0],
    )
    effective = effective_rf(model, data.validation_cones, validation_spikes[:, 0])
    effective_summary = rf_summary(
        effective,
        candidate.rf,
        data.cone_positions,
        data.cell_positions,
        candidate.metadata,
    )
    base_summary = rf_summary(
        base_rf(model).detach(),
        candidate.rf,
        data.cone_positions,
        data.cell_positions,
        candidate.metadata,
    )
    return ModelEvidence(
        name,
        prediction,
        effective_summary,
        base_summary,
        effective,
        {
            "name": name,
            "prediction": dict(prediction_payload(prediction)),
            "effective_rf": dict(rf_payload(effective_summary)),
            "base_rf": dict(rf_payload(base_summary)),
        },
    )


def _expanded_mask(mask: torch.Tensor, trials: int) -> torch.Tensor:
    return mask[:, None].expand(-1, trials, -1, -1).clone()


__all__ = [
    "MechanisticRunEvidence",
    "ModelEvidence",
    "build_model",
    "evaluate_model",
    "run_mechanistic_sampled",
]
