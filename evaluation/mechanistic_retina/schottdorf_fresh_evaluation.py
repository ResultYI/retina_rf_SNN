from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
import statistics
from typing import Final

import torch

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import SchottdorfRecording, mc_pc_recordings
from data.schottdorf_lee_multirecording import (
    load_schottdorf_cell,
    load_schottdorf_movie_drive,
)
from evaluation.mechanistic_retina.clean_sampled_reporting import (
    effective_parameter_values,
    explicit_delay_values,
    rf_bundle,
    tau_values,
)
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    constant_rate_logits,
    evaluate_retinal_model,
)
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    sha256_file,
    tensor_summary,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_types import SourceLineageError
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import (
    MechanisticGraphTemporalRetina,
    build_mechanistic_retina,
)
from training.mechanistic_retina.real_sampled import spike_prediction_metrics


_CLAMPS: Final = {
    "H1_off": frozenset({PathwayClamp.H1}),
    "direct_BC_off": frozenset(
        {PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}
    ),
    "AC_off": frozenset(
        {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
    ),
}


def learned_parameter_values(
    model: MechanisticGraphTemporalRetina,
) -> Mapping[str, torch.Tensor]:
    gates = model.gates.values(frozenset())
    return (
        effective_parameter_values(model)
        | {
            "H1_effective_amplitude": gates.h1.detach().reshape(1).clone(),
            "history_gate": gates.history.detach().reshape(1).clone(),
        }
        | {f"tau_{name}": value for name, value in tau_values(model).items()}
        | {
            f"delay_{name}": value
            for name, value in explicit_delay_values(model).items()
        }
    )


def evaluate_fresh_artifact(
    repository_dir: Path,
    movie_path: Path,
    training_dir: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("fresh evaluation output directory must be empty")
    source_path = training_dir / "results.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    adapter = SchottdorfAdapterConfig(**source["adapter_config"])
    movie = load_schottdorf_movie_drive(movie_path, adapter)
    grouped = _group_recordings(mc_pc_recordings(repository_dir / "data"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    parameter_tensors = {}
    rf_tensors = {}
    perturbation_tensors = {}
    source_cells = {cell["cell_id"]: cell for cell in source["cells"]}
    for cell_id, recordings in grouped.items():
        data = load_schottdorf_cell(recordings, movie, adapter)
        source_cell = source_cells[cell_id]
        checkpoint_path = (
            training_dir / "cells" / cell_id.replace("#", "_") / "model-trained.pt"
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config_payload = dict(checkpoint["model_config"])
        config_payload["architecture_mode"] = ArchitectureMode(
            config_payload["architecture_mode"]
        )
        model_config = MechanisticRetinaConfig(**config_payload)
        model = build_mechanistic_retina(
            model_config,
            data.cone_positions_degs,
            data.cell_positions_degs,
            data.cell_types,
            data.polarities,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        retinal_metrics, _ = evaluate_retinal_model(model, data.validation)
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
        parameters = learned_parameter_values(model)
        bundle = rf_bundle(
            model,
            data.validation.cone_drive,
            data.validation.spike_events,
        )
        mean_rf = {name: value.mean(dim=0) for name, value in bundle.items()}
        normal_rf = mean_rf["global"]
        with torch.no_grad():
            normal = model.forward_sequence(
                data.validation.cone_drive,
                observed_counts=data.validation.spike_events,
            )
        perturbations = {}
        cell_perturbation_tensors = {}
        for name, clamps in _CLAMPS.items():
            with torch.no_grad():
                clamped = model.forward_sequence(
                    data.validation.cone_drive,
                    observed_counts=data.validation.spike_events,
                    clamps=clamps,
                )
            if PathwayClamp.DIRECT_BC_SUSTAINED in clamps:
                assert torch.count_nonzero(clamped.bc_sustained_current) == 0
                assert torch.count_nonzero(clamped.bc_transient_current) == 0
                assert torch.equal(clamped.bc_broad_presynaptic, normal.bc_broad_presynaptic)
                assert torch.equal(clamped.amacrine_local_current, normal.amacrine_local_current)
                assert torch.equal(clamped.amacrine_transient_current, normal.amacrine_transient_current)
            clamped_rf = effective_rf(
                model,
                data.validation.cone_drive,
                data.validation.spike_events,
                clamps=clamps,
            ).mean(dim=0)
            logit_delta = clamped.logits - normal.logits
            probability_delta = clamped.spike_probability - normal.spike_probability
            rf_delta = clamped_rf - normal_rf
            perturbations[name] = {
                "mean_signed_logit_delta": float(logit_delta.mean()),
                "mean_absolute_logit_delta": float(logit_delta.abs().mean()),
                "mean_signed_probability_delta": float(probability_delta.mean()),
                "mean_absolute_probability_delta": float(
                    probability_delta.abs().mean()
                ),
                "normal_rf_norm": float(torch.linalg.vector_norm(normal_rf)),
                "clamped_rf_norm": float(torch.linalg.vector_norm(clamped_rf)),
                "rf_change_norm": float(torch.linalg.vector_norm(rf_delta)),
                "rf_cosine": _cosine(normal_rf, clamped_rf),
            }
            cell_perturbation_tensors[name] = {
                "logit_delta": logit_delta,
                "probability_delta": probability_delta,
                "clamped_rf": clamped_rf,
                "rf_delta": rf_delta,
            }
        cell_record = {
            "cell_id": cell_id,
            "retinal_class": source_cell["retinal_class"],
            "polarity": source_cell["polarity"],
            "validation_nll_raw": source_cell["validation_nll_raw"],
            "validation_nll_trained": retinal_metrics.population_nll,
            "constant_validation_nll": constant_metrics.population_nll,
            "effective_parameters": {
                name: tensor_summary(value) for name, value in parameters.items()
            },
            "rf": {name: tensor_summary(value) for name, value in mean_rf.items()},
            "perturbation": perturbations,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_replay_error": abs(
                retinal_metrics.population_nll
                - float(source_cell["validation_nll_trained"])
            ),
        }
        cells.append(cell_record)
        parameter_tensors[cell_id] = parameters
        rf_tensors[cell_id] = mean_rf
        perturbation_tensors[cell_id] = cell_perturbation_tensors
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise SourceLineageError("evaluation changed a trained checkpoint")
    payload = {
        "schema": "schottdorf_lee_2021_22cell_fresh_canonical_v1_evaluation",
        "source_training_artifact": str(training_dir.resolve()),
        "source_results_sha256": sha256_file(source_path),
        "cell_count": len(cells),
        "overall": _summary(cells),
        "groups": {
            f"{retinal_class}_{polarity}": _summary(
                [
                    cell
                    for cell in cells
                    if cell["retinal_class"] == retinal_class
                    and cell["polarity"] == polarity
                ]
            )
            for retinal_class in ("MC", "PC")
            for polarity in ("ON", "OFF")
        },
        "cells": cells,
        "rf_contract": {
            "contexts": "all held-out validation sequences",
            "global": "mean forward-autograd logit RF",
            "temporal": "signed cone-sum of global RF",
            "pathways": "H1/BC/AC structural decomposition",
        },
        "training_contract": source["training_contract"],
        "adapter_config": source["adapter_config"],
    }
    torch.save(parameter_tensors, output_dir / "effective-parameters.pt")
    torch.save(rf_tensors, output_dir / "rf-tensors.pt")
    torch.save(perturbation_tensors, output_dir / "perturbation-tensors.pt")
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _group_recordings(
    recordings: tuple[SchottdorfRecording, ...],
) -> dict[str, tuple[SchottdorfRecording, ...]]:
    grouped: dict[str, list[SchottdorfRecording]] = {}
    for recording in recordings:
        grouped.setdefault(recording.cell_id, []).append(recording)
    return {cell_id: tuple(values) for cell_id, values in grouped.items()}


def _summary(cells: list[dict]) -> dict[str, float | int]:
    return {
        "cells": len(cells),
        "validation_nll_raw": statistics.fmean(
            float(cell["validation_nll_raw"]) for cell in cells
        ),
        "validation_nll_trained": statistics.fmean(
            float(cell["validation_nll_trained"]) for cell in cells
        ),
        "constant_validation_nll": statistics.fmean(
            float(cell["constant_validation_nll"]) for cell in cells
        ),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return 0.0 if float(denominator) == 0.0 else float(
        torch.dot(left.flatten(), right.flatten()) / denominator
    )


__all__ = ["evaluate_fresh_artifact", "learned_parameter_values"]
