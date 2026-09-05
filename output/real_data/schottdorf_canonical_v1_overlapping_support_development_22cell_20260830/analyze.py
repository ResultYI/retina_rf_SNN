#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "scipy", "opencv-python", "matplotlib"]
# ///
# Run after run.py with the same frozen repository Python environment.
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import statistics
from typing import Final

import torch

from run import ROOT, OUT, OLD, SOURCE, REPOSITORY, MOVIE
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from evaluation.mechanistic_retina.schottdorf_fresh_evaluation import learned_parameter_values
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.center_surround_ln import DevelopmentStop
from training.mechanistic_retina.losses import expected_bernoulli_nll

REPLAY: Final = ROOT / "output/real_data/schottdorf_r4_dev_circuit_replay_20260830"


def main() -> None:
    """Compare frozen inference outputs; old checkpoint tensors are comparison-only."""
    torch.set_num_threads(2)
    assert not (OUT / "comparison.json").exists()
    source = json.loads((OUT / "results.json").read_text())
    assert len(source["cells"]) == 22
    circuit = runpy.run_path(str(REPLAY / "replay.py"))["circuit"]
    old_rf = torch.load(REPLAY / "rf-tensors.pt", weights_only=True)
    old_perturbation = torch.load(REPLAY / "perturbation-tensors.pt", weights_only=True)
    reference_hashes = json.loads((REPLAY / "replay-results.json").read_text())["source_sha256"]
    old_nll = {r["cell_id"]: r for r in json.loads((OLD / "results.json").read_text())["cells"]}
    hashes = {str(p): sha256_file(p) for p in (OUT / "results.json", OLD / "results.json",
              REPLAY / "rf-tensors.pt", REPLAY / "perturbation-tensors.pt", REPLAY / "replay.py", Path(__file__).resolve())}
    adapter = SchottdorfAdapterConfig(**source["adapter_config"])
    movie = load_schottdorf_movie_drive(MOVIE, adapter)
    recordings = mc_pc_recordings(REPOSITORY / "data")
    rows, parameter_rows, checks = [], [], []
    rf_tensors, perturbation_tensors, parameter_tensors, normal_tensors = {}, {}, {}, {}
    for cell in source["cells"]:
        cid = cell["cell_id"]
        folder = OUT / "cells" / cid.replace("#", "_")
        old_folder = OLD / "cells" / cid.replace("#", "_")
        data = load_schottdorf_cell(tuple(r for r in recordings if r.cell_id == cid), movie, adapter)
        checkpoint_path = folder / "model-trained.pt"
        hashes[str(checkpoint_path)] = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        config = MechanisticRetinaConfig(**(checkpoint["model_config"] | {
            "architecture_mode": ArchitectureMode(checkpoint["model_config"]["architecture_mode"])}))
        model = build_mechanistic_retina(config, data.cone_positions_degs, data.cell_positions_degs,
                                         data.cell_types, data.polarities)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics, logits = evaluate_retinal_model(model, data.validation)
        predictions = torch.load(folder / "validation-predictions.pt", weights_only=True)
        old_predictions = torch.load(old_folder / "validation-predictions.pt", weights_only=True)
        for key in ("target", "valid_mask"):
            assert torch.equal(predictions[key], old_predictions[key])
        for key in ("source_image_ids", "trial_indices"):
            assert predictions[key] == old_predictions[key]
        assert torch.equal(predictions["target"], data.validation.spike_events)
        assert torch.equal(predictions["valid_mask"], data.validation.valid_mask)
        assert torch.equal(logits, predictions["logits_trained"])
        assert metrics.population_nll == cell["validation_nll_trained"]
        assert float(expected_bernoulli_nll(old_predictions["logits_trained"], predictions["target"],
                                            predictions["valid_mask"])) == old_nll[cid]["r4_development_nll"]
        with (folder / "inner-trajectory.csv").open(newline="") as handle:
            curve = list(csv.DictReader(handle))
        status = DevelopmentStop(float(curve[0]["inner_dev_nll"]), 0, float(curve[0]["inner_dev_nll"]), 0)
        for point in curve[1:]:
            assert not status.stopped
            status = status.observe(float(point["inner_dev_nll"]), int(point["step"]))
        assert len(curve) == cell["stopping_step"] + 1
        assert status.best_step == cell["best_step"] and status.best_nll == cell["best_inner_dev_nll"]
        assert status.stopped or cell["stopping_step"] == 1000
        inner_checkpoint = torch.load(folder / "model-inner-best.pt", weights_only=True)
        inner_model = build_mechanistic_retina(config, data.cone_positions_degs, data.cell_positions_degs,
                                               data.cell_types, data.polarities)
        inner_model.load_state_dict(inner_checkpoint["model"], strict=True)
        inner_data = make_inner_dev(data.train)
        assert [asdict(b) for b in inner_data.boundaries] == cell["inner_boundaries"]
        dev_metrics, _ = evaluate_retinal_model(inner_model, inner_data.development)
        assert dev_metrics.population_nll == cell["best_inner_dev_nll"]
        result = circuit(model, data.validation)
        assert all(torch.equal(value, checkpoint["model"][name]) for name, value in model.state_dict().items())
        assert all(parameter.grad is None for parameter in model.parameters())
        assert torch.allclose(sum(result.rf[p] for p in ("H1", "BC", "AC")), result.rf["global"], atol=1e-7)
        parameters = learned_parameter_values(model)
        old_parameter_path = old_folder / "model-trained.pt"
        hashes[str(old_parameter_path)] = sha256_file(old_parameter_path)
        assert hashes[str(old_parameter_path)] == reference_hashes[str(old_parameter_path)]
        old_state = torch.load(old_parameter_path, weights_only=True)["model"]
        assert not bool((old_state["feature_bank.bc_support"].bool() & old_state["feature_bank.ac_support"].bool()).any())
        old_bc_raw, old_ac_raw = old_state["bipolar.raw_weights"], old_state["amacrine.raw_weights"]
        old_bc = old_bc_raw.flatten(1).softmax(1).reshape_as(old_bc_raw)[old_state["bipolar.group_index"]]
        old_ac = old_ac_raw.flatten(2).softmax(2).reshape_as(old_ac_raw)[old_state["amacrine.group_index"]]
        lower, upper = old_state["gates.h1_amplitude_bounds"].unbind()
        old_h1 = lower + (upper - lower) * old_state["gates.raw_h1_amplitude"].sigmoid()
        row = {"cell_id": cid, "group": f"{cell['retinal_class']}_{cell['polarity']}",
               "new_validation_nll": metrics.population_nll, "old_validation_nll": old_nll[cid]["r4_development_nll"],
               "new_best_step": cell["best_step"], "old_best_step": old_nll[cid]["best_step"],
               "new_stopping_step": cell["stopping_step"], "old_stopping_step": old_nll[cid]["stopping_step"],
               "new_H1_amplitude": float(parameters["H1_effective_amplitude"].item()), "old_H1_amplitude": float(old_h1)}
        for label, rfs, perturbations in (("new", result.rf, result.perturbation),
                                         ("old", old_rf[cid], old_perturbation[cid])):
            for path in ("global", "H1", "BC", "AC"):
                row[f"{label}_{path}_rf_norm"] = float(rfs[path].double().norm())
            for pathway in ("H1", "BC", "AC"):
                for response in ("logit", "probability"):
                    delta = perturbations[f"{pathway}_off"][f"{response}_delta"].double()
                    row[f"{label}_{pathway}_off_mean_abs_{response}"] = float(delta.abs().mean())
        for key in tuple(row):
            if key.startswith("new_"):
                metric = key.removeprefix("new_")
                row[f"delta_{metric}"] = row[key] - row[f"old_{metric}"]
        for path, new, old in (("BC", parameters["BC_effective_weights"], old_bc),
                               ("AC", parameters["AC_effective_weights"], old_ac)):
            new_spatial, old_spatial = new.sum(-1), old.sum(-1)
            for pathway in range(2):
                for spatial in range(2):
                    new_value, old_value = float(new_spatial[0, pathway, spatial]), float(old_spatial[0, pathway, spatial])
                    parameter_rows.append({"cell_id": cid, "group": row["group"], "path": path,
                                           "pathway_index": pathway, "spatial_mode_index": spatial,
                                           "new_weight": new_value, "old_weight": old_value, "delta": new_value - old_value})
            assert torch.allclose(new.sum((-1, -2, -3)) if path == "BC" else new.sum((-1, -2)),
                                  torch.ones_like(new.sum((-1, -2, -3)) if path == "BC" else new.sum((-1, -2))))
        parameter_tensors[cid] = {"new": parameters, "old_BC_effective_weights": old_bc,
                                  "old_AC_effective_weights": old_ac, "old_H1_amplitude": old_h1}
        rf_tensors[cid], perturbation_tensors[cid], normal_tensors[cid] = result.rf, result.perturbation, result.normal
        rows.append(row)
        checks.append({"cell_id": cid, "target_mask_segment_match": True, "prediction_replay_exact": True,
                       "inner_dev_best_and_stopping_verified": True, "structural_clamps_exact_zero": True,
                       "inference_parameters_unchanged": True})
        print(f"ANALYZED {len(rows)}/22 {cid}", flush=True)
    groups = []
    for group in ("ALL", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        subset = rows if group == "ALL" else [row for row in rows if row["group"] == group]
        groups.append({"group": group, "cells": len(subset), **{
            key: statistics.fmean(row[key] for row in subset) for key in rows[0] if key not in ("cell_id", "group")}})
    for name, values in (("per-cell-comparison.csv", rows), ("group-comparison.csv", groups),
                         ("spatial-mode-weights.csv", parameter_rows)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for name, values in (("rf-tensors.pt", rf_tensors), ("perturbation-tensors.pt", perturbation_tensors),
                         ("effective-parameters.pt", parameter_tensors), ("validation-normal-responses.pt", normal_tensors)):
        torch.save(values, OUT / name)
    assert all(sha256_file(Path(path)) == digest for path, digest in hashes.items())
    payload = {"groups": groups, "cells": rows, "spatial_mode_weights": parameter_rows, "checks": checks,
               "source_sha256": hashes, "source_hashes_unchanged": True,
               "reference": str(OLD), "reference_circuit": str(REPLAY),
               "parameter_comparison": "old checkpoint read as tensor data only after fitting; never loaded into a model",
               "rf_definition": "same mean endpoint logit Jacobian over validation sequences; last 16 bins",
               "rf_lag_window_bins": 16, "native_dt_ms": config.dt_ms,
               "rf_lag_window_ms": 16 * config.dt_ms,
               "pathway_definition": "BC=RF(H1-off,AC-off); AC=RF(H1-off)-BC; H1=global-RF(H1-off)",
               "perturbation_aggregation": "same all validation sequence bins including warmup; off minus normal",
               "spatial_weight_definition": "sum effective normalized weights over temporal-mode axis; BC pathways share unit total, AC each pathway unit total",
               "spatial_mode_order": "sigma 0.05/0.14 deg for PC(midget); 0.09/0.20 deg for MC(parasol)",
               "pathway_order": "BC sustained/transient; AC local/transient", "delta_definition": "overlapping minus exclusive-annulus"}
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
