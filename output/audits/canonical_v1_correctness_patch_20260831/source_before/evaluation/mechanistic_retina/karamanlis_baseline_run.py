from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from data.karamanlis_cells import CellSelection
from data.karamanlis_rf_population import (
    RFPopulationAdapterConfig,
    RFPopulationMarmosetData,
    load_rf_population_geometry,
    load_rf_population_imagesequence,
)
from evaluation.mechanistic_retina.karamanlis_baseline_reporting import (
    BaselineReportRequest,
    build_baseline_payload,
    write_baseline_artifacts,
)
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    PopulationGLMTrainingRequest,
    constant_rate_logits,
    evaluate_population_glm,
    evaluate_retinal_model,
    fit_population_glm,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.real_sampled import spike_prediction_metrics


@dataclass(frozen=True, slots=True)
class KaramanlisBaselineRunConfig:
    session_dir: Path
    graph_dir: Path
    checkpoint_path: Path
    output_dir: Path
    glm_max_iterations: int = 500


@dataclass(frozen=True, slots=True)
class KaramanlisBaselineRunResult:
    artifact_dir: Path
    constant_rate_nll: float
    glm_nll: float
    retinal_nll: float


@dataclass(frozen=True, slots=True)
class KaramanlisBaselineRunError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def run_karamanlis_prediction_baselines(
    config: KaramanlisBaselineRunConfig,
) -> KaramanlisBaselineRunResult:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise KaramanlisBaselineRunError("baseline output directory must be empty")
    checkpoint = torch.load(
        config.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    source_results = json.loads(
        (config.checkpoint_path.parent / "results.json").read_text(encoding="utf-8")
    )
    graph_sha256 = str(source_results["source_sha256"]["locality_graph.npz"])
    geometry = load_rf_population_geometry(
        config.graph_dir,
        grid_size=51,
        expected_graph_sha256=graph_sha256,
    )
    data = load_rf_population_imagesequence(
        config.session_dir,
        geometry,
        RFPopulationAdapterConfig(
            train_image_count=176,
            validation_image_count=44,
            cell_selection=CellSelection.ALL_QUALITY_1_TARGETS,
        ),
    )
    _verify_lineage(checkpoint, source_results, data, graph_sha256)
    model_payload = checkpoint["model_config"]
    model_config = MechanisticRetinaConfig(
        **model_payload
        | {"architecture_mode": ArchitectureMode(model_payload["architecture_mode"])}
    )
    retinal = build_mechanistic_retina(
        model_config,
        data.model_cone_positions,
        data.model_cell_positions,
        data.cell_types,
        data.polarities,
        shared_subunit_edge_index=data.edge_index,
        pathway_spatial_geometry=data.pathway_spatial_geometry,
    )
    retinal.load_state_dict(checkpoint["model"])
    retinal_metrics, retinal_logits = evaluate_retinal_model(
        retinal, data.validation
    )
    constant_logits = constant_rate_logits(
        data.train.spike_events,
        data.train.valid_mask,
        data.validation.spike_events,
        data.validation.valid_mask,
    )
    constant_metrics = spike_prediction_metrics(
        constant_logits,
        data.validation.spike_events,
        data.validation.valid_mask,
    )
    support_mask = (
        data.pathway_spatial_geometry.bc_support
        + data.pathway_spatial_geometry.ac_support
    ) > 0
    seed = int(checkpoint["training_seed"])
    glm_training = fit_population_glm(
        PopulationGLMTrainingRequest(
            train=data.train,
            cone_positions=data.model_cone_positions,
            cell_positions=data.model_cell_positions,
            graph_radius_deg=None,
            temporal_lags=model_config.lag_steps,
            steps=config.glm_max_iterations,
            seed=seed,
            support_mask=support_mask,
        )
    )
    glm_metrics, glm_logits = evaluate_population_glm(
        glm_training.model, data.validation
    )
    retinal_total = sum(parameter.numel() for parameter in retinal.parameters())
    retinal_requires_grad = sum(
        parameter.numel()
        for parameter in retinal.parameters()
        if parameter.requires_grad
    )
    retinal_optimizer_listed = sum(
        parameter.numel() for parameter in phase1_parameters(retinal)
    )
    report = BaselineReportRequest(
        source_checkpoint=config.checkpoint_path,
        checkpoint_stage=checkpoint["stage"],
        dataset_label="Karamanlis/Gollisch marmoset 60-cell RF-QC population",
        data=data,
        training_steps=config.glm_max_iterations,
        seed=seed,
        support_radius_deg=None,
        glm_training=glm_training,
        constant_metrics=constant_metrics,
        glm_metrics=glm_metrics,
        retinal_metrics=retinal_metrics,
        retinal_total_parameters=retinal_total,
        retinal_requires_grad_parameters=retinal_requires_grad,
        retinal_optimizer_listed_parameters=retinal_optimizer_listed,
        glm_support_definition=(
            "measured white-noise RF contour union over the identical canonical cone drive"
        ),
        lineage={
            "cell_count": len(data.cell_ids),
            "edge_count": int(data.edge_index.shape[1]),
            "rf_graph_sha256": graph_sha256,
            "source_retinal_nll": retinal_metrics.population_nll,
        },
    )
    payload = build_baseline_payload(report)
    write_baseline_artifacts(
        config.output_dir,
        payload,
        report,
        constant_logits,
        glm_logits,
        retinal_logits,
    )
    return KaramanlisBaselineRunResult(
        config.output_dir,
        constant_metrics.population_nll,
        glm_metrics.population_nll,
        retinal_metrics.population_nll,
    )


def _verify_lineage(
    checkpoint,
    source_results,
    data: RFPopulationMarmosetData,
    graph_sha256: str,
) -> None:
    spatial = source_results["spatial_geometry"]
    split = source_results["split"]
    population = source_results["population"]
    valid = (
        checkpoint["schema"]
        == "karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1"
        and checkpoint["stage"] == "best_trained"
        and int(checkpoint["revision"]) == MECHANISTIC_MODEL_REVISION
        and checkpoint["session_id"] == data.session_id
        and tuple(checkpoint["cell_ids"]) == data.cell_ids
        and tuple(checkpoint["cell_types"]) == data.cell_types
        and tuple(checkpoint["polarities"]) == data.polarities
        and torch.equal(checkpoint["model_cone_positions"], data.model_cone_positions)
        and torch.equal(checkpoint["model_cell_positions"], data.model_cell_positions)
        and torch.equal(checkpoint["edge_index"], data.edge_index)
        and bool(checkpoint["model_config"]["cell_specific_gains"])
        and source_results["likelihood"]
        == "Bernoulli event per native projector frame"
        and spatial["electrode_proxy_used"] is False
        and population["cell_count"] == 60
        and population["edge_count"] == 268
        and split["train_source_images"] == 176
        and split["validation_source_images"] == 44
        and source_results["source_sha256"]["locality_graph.npz"]
        == graph_sha256
        and data.train.cone_drive.shape[0] == 1290
        and data.validation.cone_drive.shape[0] == 327
        and data.train.spike_events.shape[-1] == 60
        and set(data.train.source_image_ids).isdisjoint(data.validation.source_image_ids)
    )
    if not valid:
        raise KaramanlisBaselineRunError("checkpoint/data lineage contract mismatch")
__all__ = [
    "KaramanlisBaselineRunConfig",
    "KaramanlisBaselineRunError",
    "KaramanlisBaselineRunResult",
    "run_karamanlis_prediction_baselines",
]
