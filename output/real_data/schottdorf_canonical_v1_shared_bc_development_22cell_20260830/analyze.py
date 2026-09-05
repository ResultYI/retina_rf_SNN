#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "scipy", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u analyze.py after run.py finishes.
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch

from run import OUT, OLD, REPOSITORY, MOVIE
from causal_replay import replay
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
from evaluation.mechanistic_retina.clean_sampled_reporting import rf_bundle
from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from evaluation.mechanistic_retina.schottdorf_fresh_evaluation import learned_parameter_values
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.center_surround_ln import DevelopmentStop


def main() -> None:
    torch.set_num_threads(2)
    assert not (OUT / "comparison.json").exists()
    source = json.loads((OUT / "results.json").read_text())
    previous = {r["cell_id"]: r for r in json.loads((OLD / "results.json").read_text())["cells"]}
    old_rf = torch.load(OLD / "rf-tensors.pt", weights_only=True)
    old_perturbations = torch.load(OLD / "perturbation-tensors.pt", weights_only=True)
    hashes = json.loads((OUT / "run-manifest.json").read_text())["source_sha256"]
    for path in (OLD / "rf-tensors.pt", OLD / "perturbation-tensors.pt", OUT / "results.json",
                 Path(__file__).resolve(), OUT / "causal_replay.py"):
        hashes[str(path)] = sha256_file(path)
    adapter = SchottdorfAdapterConfig(**source["adapter_config"])
    movie = load_schottdorf_movie_drive(MOVIE, adapter)
    recordings = mc_pc_recordings(REPOSITORY / "data")
    rows, parameter_rows, checks = [], [], []
    rf_tensors, perturbations, parameter_tensors = {}, {}, {}
    for cell in source["cells"]:
        cid = cell["cell_id"]
        folder = OUT / "cells" / cid.replace("#", "_")
        data = load_schottdorf_cell(tuple(r for r in recordings if r.cell_id == cid), movie, adapter)
        checkpoint_path = folder / "model-trained.pt"
        hashes[str(checkpoint_path)] = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        config = MechanisticRetinaConfig(**(checkpoint["model_config"] | {
            "architecture_mode": ArchitectureMode.MECHANISM_IDENTIFIABLE}))
        model = build_mechanistic_retina(config, data.cone_positions_degs, data.cell_positions_degs,
                                         data.cell_types, data.polarities)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        metrics, logits = evaluate_retinal_model(model, data.validation)
        predictions = torch.load(folder / "validation-predictions.pt", weights_only=True)
        reference_path = OLD / "cells" / cid.replace("#", "_") / "validation-predictions.pt"
        hashes[str(reference_path)] = sha256_file(reference_path)
        reference = torch.load(reference_path, weights_only=True)
        assert all(torch.equal(predictions[k], reference[k]) for k in ("target", "valid_mask"))
        assert all(predictions[k] == reference[k] for k in ("source_image_ids", "trial_indices"))
        assert torch.equal(predictions["target"], data.validation.spike_events)
        assert torch.equal(predictions["valid_mask"], data.validation.valid_mask)
        assert torch.equal(logits, predictions["logits_trained"])
        assert metrics.population_nll == cell["validation_nll_trained"]
        with (folder / "inner-trajectory.csv").open(newline="") as handle:
            curve = list(csv.DictReader(handle))
        status = DevelopmentStop(float(curve[0]["inner_dev_nll"]), 0, float(curve[0]["inner_dev_nll"]), 0)
        for point in curve[1:]:
            assert not status.stopped
            status = status.observe(float(point["inner_dev_nll"]), int(point["step"]))
        assert len(curve) == cell["stopping_step"] + 1
        assert status.best_step == cell["best_step"] and status.best_nll == cell["best_inner_dev_nll"]
        assert status.stopped or cell["stopping_step"] == 1000
        assert [asdict(b) for b in make_inner_dev(data.train).boundaries] == cell["inner_boundaries"]
        bundle = rf_bundle(model, data.validation.cone_drive, data.validation.spike_events)
        rfs = {("direct_BC" if k == "BC" else k): v.mean(0) for k, v in bundle.items()}
        assert torch.allclose(rfs["H1"] + rfs["direct_BC"] + rfs["AC"], rfs["global"], atol=1e-7)
        verification, replay_tensors = replay(model, data.validation.cone_drive, data.validation.spike_events)
        parameters = learned_parameter_values(model)
        verification.update({
            "prediction_replay_exact": True, "targets_masks_segments_equal_previous": True,
            "best_stopping_step_replayed": True,
            "inference_state_unchanged_including_RF": all(torch.equal(v, checkpoint["model"][k]) for k, v in model.state_dict().items()),
            "inference_parameter_gradients_absent": all(p.grad is None for p in model.parameters()),
            "inference_training_mode_unchanged": not model.training,
            "RF_and_parameters_finite": all(bool(torch.isfinite(t).all()) for t in (*rfs.values(), *parameters.values())),
        })
        assert verification["all_passed"]
        assert verification["inference_state_unchanged_including_RF"]
        assert verification["inference_parameter_gradients_absent"]
        assert verification["inference_training_mode_unchanged"]
        assert verification["RF_and_parameters_finite"]
        row = {"cell_id": cid, "group": f"{cell['retinal_class']}_{cell['polarity']}",
               "raw_validation_nll": cell["validation_nll_raw"],
               "current_validation_nll": metrics.population_nll,
               "previous_validation_nll": previous[cid]["validation_nll_trained"],
               "current_best_step": cell["best_step"], "previous_best_step": previous[cid]["best_step"],
               "current_stopping_step": cell["stopping_step"], "previous_stopping_step": previous[cid]["stopping_step"]}
        effects = {}
        for path, old_path in (("H1", "H1"), ("direct_BC", "BC"), ("AC", "AC")):
            row[f"current_{path}_rf_norm"] = float(rfs[path].double().norm())
            row[f"previous_{path}_rf_norm"] = float(old_rf[cid][old_path].double().norm())
            effects[f"{path}_off"] = {}
            for response, field in (("logit", "logits"), ("probability", "spike_probability")):
                delta = replay_tensors[f"{path}_off"][field] - replay_tensors["normal"][field]
                effects[f"{path}_off"][f"{response}_delta"] = delta
                old_delta = old_perturbations[cid][f"{old_path}_off"][f"{response}_delta"]
                assert delta.shape == old_delta.shape
                row[f"current_{path}_off_mean_abs_{response}"] = float(delta.double().abs().mean())
                row[f"previous_{path}_off_mean_abs_{response}"] = float(old_delta.double().abs().mean())
        for key in tuple(row):
            if key.startswith("current_"):
                metric = key.removeprefix("current_")
                row[f"delta_{metric}"] = row[key] - row[f"previous_{metric}"]
        parameter_row = {"cell_id": cid, "group": row["group"],
                         "H1_amplitude": float(parameters["H1_effective_amplitude"][0]),
                         "direct_BC_gain": float(parameters["cell_BC_gains"][0]),
                         "AC_gain": float(parameters["cell_AC_gains"][0]),
                         "AC_local_mixture": float(parameters["AC_effective_gates"][0, 0]),
                         "AC_transient_mixture": float(parameters["AC_effective_gates"][1, 0])}
        spatial = parameters["BC_effective_weights"].sum(-1)[0]
        for p, pathway in enumerate(("sustained", "transient")):
            for s in range(2):
                parameter_row[f"BC_shared_{pathway}_spatial_mode_{s}"] = float(spatial[p, s])
        for name, tensor in parameters.items():
            if name.startswith(("tau_", "delay_")):
                for i, value in enumerate(tensor.flatten()):
                    parameter_row[f"{name}_{i}_ms"] = float(value)
        rf_tensors[cid], perturbations[cid], parameter_tensors[cid] = rfs, effects, parameters
        rows.append(row)
        parameter_rows.append(parameter_row)
        checks.append({"cell_id": cid, **verification})
        torch.save(replay_tensors, folder / "causal-replay-tensors.pt")
        (folder / "analysis.json").write_text(json.dumps({"metrics": row, "learned": parameter_row,
                                                        "verification": verification}, indent=2))
        print(f"ANALYZED {len(rows)}/22 {cid}", flush=True)
    groups = []
    for group in ("ALL", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        selected = rows if group == "ALL" else [r for r in rows if r["group"] == group]
        groups.append({"group": group, "cells": len(selected), **{
            key: statistics.fmean(r[key] for r in selected) for key in rows[0] if key not in ("cell_id", "group")}})
    for name, values in (("per-cell-comparison.csv", rows), ("group-comparison.csv", groups),
                         ("learned-pathway-quantities.csv", parameter_rows)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for name, values in (("rf-tensors.pt", rf_tensors), ("perturbation-tensors.pt", perturbations),
                         ("effective-parameters.pt", parameter_tensors)):
        torch.save(values, OUT / name)
    assert all(sha256_file(Path(p)) == digest for p, digest in hashes.items())
    payload = {"model_name": "Canonical V1", "causal_contract": config.causal_contract,
               "spatial_contract": config.spatial_contract, "groups": groups, "cells": rows,
               "learned_pathway_quantities": parameter_rows, "verification": checks,
               "source_sha256": hashes, "source_hashes_unchanged": True, "reference": str(OLD),
               "old_checkpoints_read": False, "delta_definition": "current shared-BC minus previous overlapping-support",
               "rf_definition": "same mean endpoint logit Jacobian over validation sequences; last 16 bins",
               "pathway_RF_definition": "direct_BC=RF(H1-off,AC-off); AC=RF(H1-off)-direct_BC; H1=global-RF(H1-off)",
               "previous_BC_label_mapping": "previous BC_off final direct contribution clamp maps to direct_BC_off",
               "perturbation_aggregation": "same all validation sequence bins including warmup; off minus normal; then mean absolute per cell",
               "population_aggregation": "unweighted mean over 22 cells",
               "time_contract": {"native_dt_ms": config.dt_ms, "rf_lag_window_bins": 16,
                                 "rf_lag_window_ms": 16 * config.dt_ms, "tau_units": "ms",
                                 "explicit_pathway_delay_units": "ms", "RGC_history_shift_bins": 1},
               "BC_spatial_mode_weights_definition": "shared normalized BC weights summed over temporal basis; sustained/transient by spatial mode",
               "BC_spatial_mode_scales_deg": {"PC": [0.05, 0.14], "MC": [0.09, 0.20]}}
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
