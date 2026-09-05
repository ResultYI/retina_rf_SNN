from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TypeAlias

import torch

from data.karamanlis_2024 import KaramanlisMarmosetData
from data.karamanlis_rf_population import RFPopulationMarmosetData
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    PopulationGLMTrainingResult,
    grouped_nll,
    winner_counts,
)
from training.mechanistic_retina.real_sampled import SpikePredictionMetrics


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class BaselineReportRequest:
    source_checkpoint: Path
    checkpoint_stage: str
    dataset_label: str
    data: KaramanlisMarmosetData | RFPopulationMarmosetData
    training_steps: int
    seed: int
    support_radius_deg: float | None
    glm_training: PopulationGLMTrainingResult
    constant_metrics: SpikePredictionMetrics
    glm_metrics: SpikePredictionMetrics
    retinal_metrics: SpikePredictionMetrics
    retinal_total_parameters: int
    retinal_requires_grad_parameters: int
    retinal_optimizer_listed_parameters: int
    glm_support_definition: str = "canonical graph-radius local cones"
    lineage: Mapping[str, JsonValue] | None = None


def build_baseline_payload(
    request: BaselineReportRequest,
) -> Mapping[str, JsonValue]:
    per_model = {
        "constant_rate": request.constant_metrics.per_cell_nll,
        "glm": request.glm_metrics.per_cell_nll,
        "retinal": request.retinal_metrics.per_cell_nll,
    }
    winners = winner_counts(per_model)
    grouped = grouped_nll(
        per_model,
        request.data.polarities,
        request.data.cell_types,
    )
    glm_parameters = sum(
        parameter.numel()
        for parameter in request.glm_training.model.parameters()
    )
    per_cell: list[JsonValue] = [
        {
            "id": cell_id,
            "type": cell_type,
            "polarity": polarity,
            "constant_rate_nll": request.constant_metrics.per_cell_nll[index],
            "glm_nll": request.glm_metrics.per_cell_nll[index],
            "retinal_nll": request.retinal_metrics.per_cell_nll[index],
            "winner": min(
                per_model,
                key=lambda name: per_model[name][index],
            ),
        }
        for index, (cell_id, cell_type, polarity) in enumerate(
            zip(
                request.data.cell_ids,
                request.data.cell_types,
                request.data.polarities,
                strict=True,
            )
        )
    ]
    payload: dict[str, JsonValue] = {
        "schema": "karamanlis_2024_population_prediction_baselines_v1",
        "source_checkpoint": str(request.source_checkpoint),
        "source_checkpoint_stage": request.checkpoint_stage,
        "dataset": request.dataset_label,
        "session_id": request.data.session_id,
        "evaluation_contract": {
            "target": "Bernoulli spike event per native 85 Hz bin",
            "nll": "mean binary cross-entropy over identical valid bins",
            "train_source_images": len(set(request.data.train.source_image_ids)),
            "validation_source_images": len(
                set(request.data.validation.source_image_ids)
            ),
            "train_sequences": request.data.train.cone_drive.shape[0],
            "validation_sequences": request.data.validation.cone_drive.shape[0],
            "train_valid_bins": int(request.data.train.valid_mask.sum()),
            "validation_valid_bins": int(
                request.data.validation.valid_mask.sum()
            ),
            "source_image_disjoint": set(
                request.data.train.source_image_ids
            ).isdisjoint(request.data.validation.source_image_ids),
        },
        "glm_contract": {
            "stimulus_features": (
                f"{request.glm_support_definition} x "
                f"{request.glm_training.model.temporal_lags} current/past causal lags"
            ),
            "history_features": "4 strictly-past same-cell spike-event lags",
            "cross_cell_coupling": False,
            "validation_used_for_fit_or_selection": False,
            "optimizer": "full-batch L-BFGS strong-Wolfe, train split only",
            "maximum_iterations": request.training_steps,
            "solver_iterations": request.glm_training.solver_iterations,
            "solver_converged": request.glm_training.converged,
            "seed": request.seed,
            "support_radius_deg": request.support_radius_deg,
            "support_count_min": min(request.glm_training.model.support_counts),
            "support_count_mean": sum(request.glm_training.model.support_counts)
            / len(request.glm_training.model.support_counts),
            "support_count_max": max(request.glm_training.model.support_counts),
            "train_nll_initial": request.glm_training.train_nll_initial,
            "train_nll_trained": request.glm_training.train_nll_trained,
            "gradients_finite": request.glm_training.gradients_finite,
            "actually_updated": list(request.glm_training.actually_updated),
        },
        "population_validation_nll": {
            "constant_rate": request.constant_metrics.population_nll,
            "glm": request.glm_metrics.population_nll,
            "retinal": request.retinal_metrics.population_nll,
        },
        "retinal_delta_nll": {
            "versus_constant_rate": request.retinal_metrics.population_nll
            - request.constant_metrics.population_nll,
            "versus_glm": request.retinal_metrics.population_nll
            - request.glm_metrics.population_nll,
        },
        "winner_cell_counts": dict(winners),
        "exact_tie_cell_count": _tie_count(per_model),
        "by_cell_class": {
            label: dict(values) for label, values in grouped.items()
        },
        "per_cell": per_cell,
        "parameter_counts": {
            "constant_rate": {"fitted": len(request.data.cell_ids)},
            "glm": {
                "total": glm_parameters,
                "requires_grad": glm_parameters,
                "optimizer_listed": glm_parameters,
            },
            "retinal": {
                "total": request.retinal_total_parameters,
                "requires_grad": request.retinal_requires_grad_parameters,
                "optimizer_listed_in_source_fit": request.retinal_optimizer_listed_parameters,
            },
        },
        "comparison_scope": {
            "claim": "prediction/capacity comparison only",
            "matched_capacity": False,
            "retinal_model_retrained": False,
            "validation_used_for_glm_fit_or_selection": False,
        },
    }
    if request.lineage is not None:
        payload["lineage"] = dict(request.lineage)
    return payload


def write_baseline_artifacts(
    output_dir: Path,
    payload: Mapping[str, JsonValue],
    request: BaselineReportRequest,
    constant_logits: torch.Tensor,
    glm_logits: torch.Tensor,
    retinal_logits: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    torch.save(
        {
            "schema": payload["schema"],
            "model": request.glm_training.model.state_dict(),
            "contract": payload["glm_contract"],
        },
        output_dir / "glm-trained.pt",
    )
    torch.save(
        {
            "constant_rate_logits": constant_logits,
            "glm_logits": glm_logits,
            "retinal_logits": retinal_logits,
            "spike_events": request.data.validation.spike_events,
            "valid_mask": request.data.validation.valid_mask,
            "source_image_ids": request.data.validation.source_image_ids,
            "cell_ids": request.data.cell_ids,
        },
        output_dir / "validation-predictions.pt",
    )


def _tie_count(per_model) -> int:
    cell_count = len(next(iter(per_model.values())))
    return sum(
        sorted(per_model[name][cell] for name in per_model)[1]
        - sorted(per_model[name][cell] for name in per_model)[0]
        <= 1e-12
        for cell in range(cell_count)
    )


__all__ = [
    "BaselineReportRequest",
    "build_baseline_payload",
    "write_baseline_artifacts",
]
