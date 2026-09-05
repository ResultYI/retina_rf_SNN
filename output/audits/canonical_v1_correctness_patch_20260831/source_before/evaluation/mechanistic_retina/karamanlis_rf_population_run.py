from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.atomic_artifacts import (
    atomic_torch_save,
    atomic_write_text,
)
from evaluation.mechanistic_retina.karamanlis_rf_checkpoint import (
    rf_population_checkpoint_base,
)
from evaluation.json_types import JsonValue
from data.karamanlis_cells import CellSelection
from data.karamanlis_rf_population import (
    RFPopulationAdapterConfig,
    RFPopulationMarmosetData,
    load_rf_population_geometry,
    load_rf_population_imagesequence,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.real_early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingTrainingResult,
    EarlyStoppingTrainingRequest,
    fit_real_spike_model_early_stopping,
)


_CONSTANT_RATE_NLL = 0.15232950448989868
_LOCALITY_GRAPH_SHA256 = "daf98676d14f829e079ebd5b8f1666f30892bc3ed12a654ef3f9033768edf743"


@dataclass(frozen=True, slots=True)
class RFPopulationRunConfig:
    session_dir: Path
    graph_dir: Path
    output_dir: Path
    learning_rate: float = 0.03
    batch_size: int = 4
    seed: int = 202_603_02
    adapter: RFPopulationAdapterConfig = RFPopulationAdapterConfig()
    stopping: EarlyStoppingConfig = EarlyStoppingConfig(
        max_steps=500,
        evaluation_interval=10,
        patience=8,
        min_delta=1e-5,
    )


@dataclass(frozen=True, slots=True)
class RFPopulationRunResult:
    artifact_dir: Path
    validation_nll_raw: float
    validation_nll_best: float
    best_step: int


def run_rf_population_training(
    config: RFPopulationRunConfig,
) -> RFPopulationRunResult:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError("RF population output directory must be empty")
    geometry = load_rf_population_geometry(
        config.graph_dir,
        grid_size=51,
        expected_graph_sha256=_LOCALITY_GRAPH_SHA256,
    )
    data = load_rf_population_imagesequence(
        config.session_dir,
        geometry,
        config.adapter,
    )
    model_config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        cell_specific_gains=True,
        dt_ms=data.dt_ms,
    )
    torch.manual_seed(config.seed)
    model = build_mechanistic_retina(
        model_config,
        data.model_cone_positions,
        data.model_cell_positions,
        data.cell_types,
        data.polarities,
        shared_subunit_edge_index=data.edge_index,
        pathway_spatial_geometry=data.pathway_spatial_geometry,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = rf_population_checkpoint_base(
        data,
        model_config,
        training_seed=config.seed,
    )
    atomic_torch_save(
        checkpoint | {"stage": "raw", "model": model.state_dict()},
        config.output_dir / "model-raw.pt",
    )
    training = fit_real_spike_model_early_stopping(
        EarlyStoppingTrainingRequest(
            model=model,
            train=data.train,
            validation=data.validation,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            seed=config.seed,
            stopping=config.stopping,
        )
    )
    atomic_torch_save(
        checkpoint
        | {
            "stage": "best_trained",
            "best_step": training.best_step,
            "model": model.state_dict(),
        },
        config.output_dir / "model-best.pt",
    )
    payload = _results_payload(data, config, model_config, training)
    atomic_write_text(
        config.output_dir / "results.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )
    return RFPopulationRunResult(
        config.output_dir,
        training.raw_metrics.population_nll,
        training.best_metrics.population_nll,
        training.best_step,
    )


def _results_payload(
    data: RFPopulationMarmosetData,
    config: RFPopulationRunConfig,
    model_config: MechanisticRetinaConfig,
    training: EarlyStoppingTrainingResult,
) -> dict[str, JsonValue]:
    per_cell = [
        {
            "id": cell_id,
            "type": cell_type,
            "polarity": polarity,
            "nll_raw": raw,
            "nll_best": best,
            "nll_improvement": raw - best,
        }
        for cell_id, cell_type, polarity, raw, best in zip(
            data.cell_ids,
            data.cell_types,
            data.polarities,
            training.raw_metrics.per_cell_nll,
            training.best_metrics.per_cell_nll,
            strict=True,
        )
    ]
    classes = {}
    for polarity, cell_type in tuple(
        dict.fromkeys(zip(data.polarities, data.cell_types, strict=True))
    ):
        rows = [
            row
            for row in per_cell
            if row["polarity"] == polarity and row["type"] == cell_type
        ]
        classes[f"{polarity} {cell_type}"] = {
            "cell_count": len(rows),
            "mean_validation_nll_raw": sum(row["nll_raw"] for row in rows)
            / len(rows),
            "mean_validation_nll_best": sum(row["nll_best"] for row in rows)
            / len(rows),
        }
    edge_index = data.edge_index
    return {
        "schema": "karamanlis_marmoset_rf_geometry_population_training_v1",
        "model_class": "physiology-constrained recurrent/state-space retinal point-process model",
        "model_revision": MECHANISTIC_MODEL_REVISION,
        "causal_contract": model_config.causal_contract,
        "topology": "Cone->H1->shared BC encoder; narrow BC->RGC; broad BC->AC->RGC; RGC->spike likelihood",
        "session_id": data.session_id,
        "fresh_initialization": True,
        "fresh_optimizer": True,
        "native_dt_ms": data.dt_ms,
        "likelihood": "Bernoulli event per native projector frame",
        "spatial_geometry": {
            "cell_and_cone_unit": data.cone_drive_coordinate_unit,
            "axis_convention": "x-right/y-down source-array encoding used in this completed run; a global y reflection leaves distances, supports, graph mixing, and forward values unchanged",
            "lineage": "white-noise RF center/contour/extent in retinal micrometers and RF-derived explicit locality graph; post-run audit identified the source-array y-down encoding",
            "electrode_proxy_used": False,
            "crop_pixels": data.crop_pixels,
            "pool_factor": data.pool_factor,
            "pooled_grid": [data.pooled_grid_size, data.pooled_grid_size],
            "bc_ac_support": "BC central support at the unchanged extent ratio; AC full measured 25-percent RF contour including BC",
        },
        "population": {
            "cell_count": len(data.cell_ids),
            "edge_count": int(edge_index.shape[1]),
            "self_edge_count": int((edge_index[0] == edge_index[1]).sum()),
            "nonself_edge_count": int((edge_index[0] != edge_index[1]).sum()),
        },
        "split": {
            "train_source_images": len(set(data.train.source_image_ids)),
            "validation_source_images": len(set(data.validation.source_image_ids)),
            "train_sequences": data.train.cone_drive.shape[0],
            "validation_sequences": data.validation.cone_drive.shape[0],
            "source_image_disjoint": set(data.train.source_image_ids).isdisjoint(
                data.validation.source_image_ids
            ),
        },
        "validation_prediction": {
            "nll_raw": training.raw_metrics.population_nll,
            "nll_best": training.best_metrics.population_nll,
            "best_minus_constant_rate": training.best_metrics.population_nll
            - _CONSTANT_RATE_NLL,
            "per_cell": per_cell,
            "by_cell_class": classes,
        },
        "training": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "early_stopping": asdict(config.stopping),
            "best_step": training.best_step,
            "completed_steps": training.completed_steps,
            "stopped_early": training.stopped_early,
            "validation_trace": [asdict(point) for point in training.validation_trace],
            "gradients_finite": training.gradients_finite,
            "best_checkpoint_parameters_different_from_initial": training.actually_updated,
            "nonself_connections": {
                "nonzero_gradient": training.nonself_connection_gradient_nonzero,
                "optimizer_step_changed_parameter": training.nonself_connection_optimizer_updated,
                "best_checkpoint_differs_from_initial": training.nonself_connection_updated,
                "max_abs_gradient": training.nonself_connection_max_abs_gradient,
                "best_checkpoint_update_norm": training.nonself_connection_update_norm,
            },
            "cell_specific_gain_audit": None if training.cell_gain_audit is None else asdict(training.cell_gain_audit),
        },
        "model_config": asdict(model_config)
        | {"architecture_mode": model_config.architecture_mode.value},
        "source_sha256": {
            "expdata.mat": _sha256(config.session_dir / "expdata.mat"),
            "imagesequence_data.mat": _sha256(
                config.session_dir / "imagesequence_data.mat"
            ),
            "locality_graph.npz": _sha256(config.graph_dir / "locality_graph.npz"),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["RFPopulationRunConfig", "RFPopulationRunResult", "run_rf_population_training"]
