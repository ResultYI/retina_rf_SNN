#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "scipy", "opencv-python"]
# ///
# Run with the frozen repository environment: D:/anaconda/python.exe -B run.py
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Final

import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from evaluation.mechanistic_retina.schottdorf_ln_source import FrozenSource
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.r4_development import select_and_refit_r4

OLD: Final = ROOT / "output/real_data/schottdorf_r4_development_22cell_20260830_verified"
SOURCE: Final = ROOT / "output/real_data/schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829"
REPOSITORY: Final = ROOT / "data/real/schottdorf_lee_2021_repository"
MOVIE: Final = ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"


def main() -> None:
    """Reuse frozen selection/refit, without reading any checkpoint for training."""
    assert not (OUT / "cells").exists(), "fresh output required"
    torch.set_num_threads(2)
    source = FrozenSource.model_validate_json((SOURCE / "results.json").read_text())
    recordings = mc_pc_recordings(REPOSITORY / "data")
    hashes = {str(path): sha256_file(path) for folder in ("models", "training", "evaluation", "data")
              for path in (ROOT / folder).rglob("*.py")}
    for path in (SOURCE / "results.json", OLD / "results.json", MOVIE, Path(__file__).resolve()):
        hashes[str(path)] = sha256_file(path)
    assert hashes[str(MOVIE)] == source.source_sha256[MOVIE.name]
    movie = load_schottdorf_movie_drive(MOVIE, source.adapter_config)
    manifest = {"source_sha256": hashes, "torch_version": torch.__version__, "torch_threads": 2,
                "model_name": "Canonical V1", "old_checkpoint_loaded_for_training": False,
                "protocol": "unmodified training.mechanistic_retina.r4_development.select_and_refit_r4",
                "cell_ids": [cell.cell_id for cell in source.cells]}
    (OUT / "run-manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for index, expected in enumerate(source.cells):
        cid = expected.cell_id
        cell_dir = OUT / "cells" / cid.replace("#", "_")
        cell_dir.mkdir(parents=True)
        old_json = OLD / "cells" / cid.replace("#", "_") / "results.json"
        hashes[str(old_json)] = sha256_file(old_json)
        previous = json.loads(old_json.read_text())
        selected_recordings = tuple(r for r in recordings if r.cell_id == cid)
        assert tuple(r.recording_id for r in selected_recordings) == expected.recording_ids
        for recording in selected_recordings:
            hashes[str(recording.path)] = sha256_file(recording.path)
            assert hashes[str(recording.path)] == expected.source_sha256[recording.path.name]
        data = load_schottdorf_cell(selected_recordings, movie, source.adapter_config)
        counts = (len(data.train.source_image_ids), len(data.validation.source_image_ids),
                  int(data.train.valid_mask.sum()), int(data.validation.valid_mask.sum()), data.dt_ms)
        assert counts == (expected.train_sequences, expected.validation_sequences,
                          expected.train_valid_bins, expected.validation_valid_bins, expected.native_dt_ms)
        assert data.input_representation == source.input_representation
        assert not set(data.train.source_image_ids) & set(data.validation.source_image_ids)
        config_payload = previous["model_config"] | {"architecture_mode": ArchitectureMode.MECHANISM_IDENTIFIABLE}
        config = MechanisticRetinaConfig(**config_payload)
        assert config.spatial_contract == "bc-central-disk_ac-overlapping-full-disk"
        serialized_config = json.loads(json.dumps(asdict(config)))
        assert {k: v for k, v in serialized_config.items() if k != "spatial_contract"} == previous["model_config"]
        seed = previous["training_contract"]["seed"]

        def factory() -> MechanisticGraphTemporalRetina:
            torch.manual_seed(seed)
            return build_mechanistic_retina(config, data.cone_positions_degs,
                                            data.cell_positions_degs, data.cell_types, data.polarities)

        print(f"START {index + 1}/22 {cid} seed={seed}", flush=True)
        selection = select_and_refit_r4(data.train, factory, seed)
        raw = factory()
        assert all(torch.equal(value, selection.refit.initial_state[name]) for name, value in raw.state_dict().items())
        assert all(torch.equal(value, selection.inner.initial_state[name]) for name, value in raw.state_dict().items())
        assert all(step == selection.inner.best_step for step in selection.refit.optimizer_steps)
        boundaries = [asdict(boundary) for boundary in selection.split.boundaries]
        assert boundaries == previous["inner_boundaries"]
        trained = selection.refit.model
        bc, ac = trained.feature_bank.bc_support.bool(), trained.feature_bank.ac_support.bool()
        assert bool((bc <= ac).all()) and bool((bc & ac).any()) and bool((ac & ~bc).any())
        raw_metrics, raw_logits = evaluate_retinal_model(raw, data.validation)
        trained_metrics, trained_logits = evaluate_retinal_model(trained, data.validation)
        contract = {key: value for key, value in previous["training_contract"].items()
                    if key not in {"fresh_parameters_exactly_match_old_raw", "fixed_h1_graph_buffer_max_abs_difference",
                                   "fixed_h1_graph_buffer_absolute_tolerance", "all_other_state_matches_old_raw_exactly"}}
        contract.update({"fresh_full_train_refit_steps": selection.inner.best_step,
                         "old_checkpoint_loaded_for_training": False, "fresh_initializations_identical": True})
        metadata = {"schema": "schottdorf_canonical_v1_overlapping_support_development",
                    "revision": 4, "model_name": "Canonical V1", "cell_id": cid,
                    "recording_ids": list(expected.recording_ids), "model_config": serialized_config,
                    "seed": seed, "cell_types": list(data.cell_types), "polarities": list(data.polarities),
                    "cone_positions_degs": data.cone_positions_degs, "cell_positions_degs": data.cell_positions_degs,
                    "training_contract": contract}
        for stage, state, steps in (("raw", selection.refit.initial_state, 0),
                                    ("inner-best", selection.inner.model.state_dict(), selection.inner.best_step),
                                    ("trained", trained.state_dict(), selection.refit.stop_step)):
            torch.save(metadata | {"model": state, "stage": stage, "steps": steps}, cell_dir / f"model-{stage}.pt")
        torch.save({"logits_raw": raw_logits, "logits_trained": trained_logits,
                    "probabilities_trained": trained_logits.sigmoid(), "target": data.validation.spike_events,
                    "valid_mask": data.validation.valid_mask, "source_image_ids": data.validation.source_image_ids,
                    "trial_indices": data.validation.trial_indices}, cell_dir / "validation-predictions.pt")
        for label, fit in (("inner", selection.inner), ("refit", selection.refit)):
            with (cell_dir / f"{label}-trajectory.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step", "sampled_train_batch_nll", "inner_dev_nll"])
                writer.writeheader()
                writer.writerows(asdict(step) for step in fit.trajectory)
        row = {"cell_id": cid, "retinal_class": previous["retinal_class"], "polarity": previous["polarity"],
               "recording_ids": list(expected.recording_ids), "native_dt_ms": data.dt_ms,
               "input_representation": data.input_representation, "model_config": serialized_config,
               "training_contract": contract, "best_step": selection.inner.best_step,
               "stopping_step": selection.inner.stop_step, "best_inner_dev_nll": selection.inner.best_dev_nll,
               "inner_train_nll_raw": selection.inner.train_nll_raw, "inner_train_nll_best": selection.inner.train_nll_trained,
               "full_train_nll_raw": selection.refit.train_nll_raw, "full_train_nll_refit": selection.refit.train_nll_trained,
               "validation_nll_raw": raw_metrics.population_nll, "validation_nll_trained": trained_metrics.population_nll,
               "gradients_finite": selection.inner.gradients_finite and selection.refit.gradients_finite,
               "actually_updated": list(selection.refit.actually_updated), "optimizer_steps": selection.refit.optimizer_steps,
               "parameter_counts": {"total": sum(p.numel() for p in trained.parameters()),
                                    "requires_grad": sum(p.numel() for p in trained.parameters() if p.requires_grad),
                                    "optimizer_listed": sum(p.numel() for p in phase1_parameters(trained))},
               "inner_boundaries": boundaries, "train_sequences": counts[0], "validation_sequences": counts[1],
               "train_valid_bins": counts[2], "validation_valid_bins": counts[3],
               "support": {"bc": int(bc.sum()), "ac": int(ac.sum()), "overlap": int((bc & ac).sum())}}
        (cell_dir / "results.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"DONE {index + 1}/22 {cid} best={row['best_step']} stop={row['stopping_step']} NLL={row['validation_nll_trained']:.9f}", flush=True)
    assert all(sha256_file(Path(path)) == digest for path, digest in hashes.items())
    manifest.update({"source_sha256": hashes, "source_hashes_unchanged": True})
    (OUT / "run-manifest.json").write_text(json.dumps(manifest, indent=2))
    payload = {"model_name": "Canonical V1", "spatial_contract": config.spatial_contract,
               "cell_count": 22, "recording_count": 37, "cells": rows,
               "adapter_config": asdict(source.adapter_config), "training_contract": rows[0]["training_contract"]}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
