from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from evaluation.mechanistic_retina.schottdorf_ln_source import LNSourcePaths, load_ln_cell
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.r4_development import (
    BATCH_SIZE, LEARNING_RATE, MAX_STEPS, select_and_refit_r4,
)
from training.mechanistic_retina.real_sampled import RealSpikeTrainingError


@dataclass(frozen=True, slots=True)
class R4CellResult:
    cell_id: str
    best_step: int
    stop_step: int
    best_inner_dev_nll: float
    validation_nll: float
    artifact_dir: Path


def verify_fresh_state(current: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    if current.keys() != reference.keys():
        raise RealSpikeTrainingError("fresh initialization state keys differ")
    graph_roundoff = 0.0
    for name, value in current.items():
        expected = reference[name]
        if torch.equal(value, expected):
            continue
        if name != "h1.graph.edge_weight" or not torch.allclose(value, expected, atol=1e-6, rtol=0):
            raise RealSpikeTrainingError(f"fresh initialization differs from frozen R4 raw state: {name}")
        graph_roundoff = float((value - expected).abs().max())
    return graph_roundoff


def run_r4_development_cell(paths: LNSourcePaths, cell_id: str, output: Path) -> R4CellResult:
    resolved, source = output.resolve(), paths.retinal_artifact.resolve()
    if resolved == source or source in resolved.parents or resolved in source.parents:
        raise RealSpikeTrainingError("benchmark output must be outside frozen artifacts")
    if output.exists() and any(output.iterdir()):
        raise RealSpikeTrainingError("benchmark output must be empty")
    loaded = load_ln_cell(paths, cell_id)
    data = loaded.data
    raw_path = source / "cells" / cell_id.replace("#", "_") / "model-raw.pt"
    raw_hash = sha256_file(raw_path)
    reference = torch.load(raw_path, map_location="cpu", weights_only=True)
    if reference["revision"] != 4 or reference["cell_id"] != cell_id or reference["stage"] != "raw":
        raise RealSpikeTrainingError("frozen R4 initialization metadata mismatch")
    model_payload = dict(reference["model_config"])
    model_payload["architecture_mode"] = ArchitectureMode(model_payload["architecture_mode"])
    config = MechanisticRetinaConfig(**model_payload)
    seed = int(reference["seed"])
    graph_roundoff = []

    def factory() -> MechanisticGraphTemporalRetina:
        torch.manual_seed(seed)
        model = build_mechanistic_retina(config, data.cone_positions_degs,
                                        data.cell_positions_degs, data.cell_types, data.polarities)
        graph_roundoff.append(verify_fresh_state(model.state_dict(), reference["model"]))
        return model

    selection = select_and_refit_r4(data.train, factory, seed)
    raw_metrics, raw_logits = evaluate_retinal_model(factory(), data.validation)
    metrics, logits = evaluate_retinal_model(selection.refit.model, data.validation)
    hashes = loaded.source_hashes | {str(raw_path): raw_hash}
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise RealSpikeTrainingError("frozen source changed during benchmark")
    if selection.inner.best_dev_nll is None:
        raise RealSpikeTrainingError("inner development score missing")
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "maximum_inner_steps": MAX_STEPS, "patience": 200, "min_delta": 1e-7,
        "seed": seed, "minibatch_seed": seed + 1_000_003, "regularizer": None,
        "likelihood": "Bernoulli sampled measured spike events",
        "split": "same make_inner_dev as center-surround LN; 80/20 per training trial; 60-bin guard",
        "raw_step_zero_is_eligible": True,
        "best_checkpoint": "exact lowest inner-dev NLL; min_delta controls patience reset only",
        "original_validation_used_for_selection": False,
        "fresh_full_train_refit_steps": selection.inner.best_step,
        "fresh_optimizer_per_fit": True, "fresh_parameters_exactly_match_old_raw": True,
        "fixed_h1_graph_buffer_max_abs_difference": max(graph_roundoff),
        "fixed_h1_graph_buffer_absolute_tolerance": 1e-6,
        "all_other_state_matches_old_raw_exactly": True,
        "retinal_projection_after_each_update": True,
    }
    metadata = {
        "schema": "schottdorf_r4_development_stopping_v1", "revision": 4,
        "cell_id": cell_id, "recording_ids": data.recording_ids,
        "model_config": reference["model_config"], "seed": seed,
        "cell_types": data.cell_types, "polarities": data.polarities,
        "cone_positions_degs": data.cone_positions_degs, "cell_positions_degs": data.cell_positions_degs,
        "source_sha256": hashes, "training_contract": contract,
    }
    torch.save(metadata | {"model": selection.inner.model.state_dict(), "stage": "inner-best",
                           "best_step": selection.inner.best_step, "stop_step": selection.inner.stop_step},
               output / "model-inner-best.pt")
    torch.save(metadata | {"model": selection.refit.initial_state, "stage": "raw", "steps": 0},
               output / "model-raw.pt")
    torch.save(metadata | {"model": selection.refit.model.state_dict(), "stage": "trained",
                           "steps": selection.refit.stop_step, "optimizer_steps": selection.refit.optimizer_steps},
               output / "model-trained.pt")
    torch.save({"logits_raw": raw_logits, "logits_trained": logits,
                "probabilities_trained": logits.sigmoid(), "target": data.validation.spike_events,
                "valid_mask": data.validation.valid_mask, "source_image_ids": data.validation.source_image_ids,
                "trial_indices": data.validation.trial_indices}, output / "validation-predictions.pt")
    for name, fitted in (("inner", selection.inner), ("refit", selection.refit)):
        with (output / f"{name}-trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("step", "sampled_train_batch_nll", "inner_dev_nll"))
            writer.writeheader()
            writer.writerows(asdict(row) for row in fitted.trajectory)
    payload = {
        "cell_id": cell_id, "retinal_class": data.retinal_classes[0], "polarity": data.polarities[0],
        "recording_ids": data.recording_ids, "native_dt_ms": data.dt_ms,
        "input_representation": data.input_representation, "adapter_config": asdict(loaded.adapter),
        "training_contract": contract, "model_config": reference["model_config"],
        "best_step": selection.inner.best_step, "stopping_step": selection.inner.stop_step,
        "best_inner_dev_nll": selection.inner.best_dev_nll,
        "inner_trajectory": [asdict(row) for row in selection.inner.trajectory],
        "refit_steps": selection.refit.stop_step, "optimizer_steps": selection.refit.optimizer_steps,
        "inner_train_nll_raw": selection.inner.train_nll_raw,
        "inner_train_nll_best": selection.inner.train_nll_trained,
        "full_train_nll_raw": selection.refit.train_nll_raw,
        "full_train_nll_refit": selection.refit.train_nll_trained,
        "validation_nll_raw": raw_metrics.population_nll, "validation_nll_trained": metrics.population_nll,
        "gradients_finite": selection.inner.gradients_finite and selection.refit.gradients_finite,
        "actually_updated": selection.refit.actually_updated,
        "parameter_counts": {"total": sum(p.numel() for p in selection.refit.model.parameters()),
                             "requires_grad": sum(p.numel() for p in selection.refit.model.parameters() if p.requires_grad),
                             "optimizer_listed": sum(p.numel() for p in phase1_parameters(selection.refit.model))},
        "inner_boundaries": [asdict(row) for row in selection.split.boundaries],
        "train_sequences": data.train.cone_drive.shape[0], "validation_sequences": data.validation.cone_drive.shape[0],
        "train_valid_bins": int(data.train.valid_mask.sum()), "validation_valid_bins": int(data.validation.valid_mask.sum()),
        "inner_train_valid_bins": int(selection.split.train.valid_mask.sum()),
        "inner_dev_valid_bins": int(selection.split.development.valid_mask.sum()),
        "source_hashes": hashes, "source_hashes_unchanged": True,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return R4CellResult(cell_id, selection.inner.best_step, selection.inner.stop_step,
                        selection.inner.best_dev_nll, metrics.population_nll, output)
