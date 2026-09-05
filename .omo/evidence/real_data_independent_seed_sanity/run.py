#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u run.py
# Run prepare.py first; never resume, retry a fit, or change its training contract.
from __future__ import annotations

from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import time

import torch

from source import APPLICATION, OUT, ROOT, SOURCE, Snapshot, fresh, load_data, save_json, sha
from replay import FitIdentity, illusion, pathway_effects
from metrics import write_csv
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from training.mechanistic_retina.r4_development import select_and_refit_r4


def main() -> None:
    assert not (OUT / "fits").exists(), "STOP: fresh output required, no resumes"
    selection = json.loads((OUT / "selection.json").read_text())
    preflight = json.loads((OUT / "preflight.json").read_text())
    assert preflight["status"] == "PASS"
    hashes = preflight["source_sha256"]
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    for path in (OUT / "selection.json", OUT / "preflight.json", OUT / "provenance.json", *OUT.glob("*.py")):
        hashes[str(path)] = sha(path)
    save_json(OUT / "run-manifest.json", {"status": "STARTED", "source_sha256": hashes,
        "seed_selection_saved_before_training": True, "training_function": "select_and_refit_r4",
        "new_fit_count": 8, "primary_refitted": False, "training_device": "cpu", "torch_threads": 2})
    torch.set_num_threads(2)
    snapshot = Snapshot.model_validate_json((SOURCE / "results.json").read_text())
    movie = load_schottdorf_movie_drive(ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg", snapshot.adapter_config)
    saved_illusions = torch.load(APPLICATION / "illusion/responses.pt", weights_only=True)["cells"]
    results, signatures, checks = [], [], []
    training_count = 0
    for selected in selection["cells"]:
        cell = next(c for c in snapshot.cells if c.cell_id == selected["cell_id"])
        data = load_data(cell, movie, snapshot.adapter_config)
        starts = []
        for label, seed in zip(("primary", "fresh_1", "fresh_2"), (cell.primary_seed, *selected["fresh_seeds"]), strict=True):
            folder = OUT / "fits" / cell.cell_id.replace("#", "_") / label
            folder.mkdir(parents=True)
            factory = partial(fresh, cell, data, seed)
            start = time.perf_counter()
            raw = factory()
            raw_state = {k: v.detach().clone() for k, v in raw.state_dict().items()}
            assert all(any(not torch.equal(v, old[k]) for k, v in raw_state.items()) for old in starts)
            starts.append(raw_state)
            match label:
                case "primary":
                    raw.load_state_dict(torch.load(cell.directory / "model-trained.pt", weights_only=True)["model"], strict=True)
                    model = raw
                    best_step, stop_step = cell.best_step, cell.stopping_step
                    checkpoint_path = cell.directory / "model-trained.pt"
                case "fresh_1" | "fresh_2":
                    training_count += 1
                    assert training_count <= 8
                    print(f"START {training_count}/8 {cell.cell_id} {label} seed={seed}", flush=True)
                    fit = select_and_refit_r4(data.train, factory, seed)
                    assert tuple(asdict(b) for b in fit.split.boundaries) == cell.inner_boundaries
                    assert all(torch.equal(v, fit.inner.initial_state[k]) and torch.equal(v, fit.refit.initial_state[k]) for k, v in raw_state.items())
                    assert all(step == fit.inner.best_step for step in fit.refit.optimizer_steps)
                    model, best_step, stop_step = fit.refit.model, fit.inner.best_step, fit.inner.stop_step
                    contract = cell.training_contract | {"seed": seed, "minibatch_seed": seed + 1000003,
                        "fresh_full_train_refit_steps": best_step}
                    metadata = {"cell_id": cell.cell_id, "model_config": asdict(cell.configuration),
                        "cell_types": data.cell_types, "polarities": data.polarities,
                        "cone_positions_degs": data.cone_positions_degs, "cell_positions_degs": data.cell_positions_degs,
                        "seed": seed, "training_contract": contract, "recording_ids": data.recording_ids}
                    for stage, state in (("raw", raw_state), ("inner-best", fit.inner.model.state_dict()), ("trained", model.state_dict())):
                        torch.save(metadata | {"model": state, "stage": stage}, folder / f"model-{stage}.pt")
                    checkpoint_path = folder / "model-trained.pt"
                    for stage, result in (("inner", fit.inner), ("refit", fit.refit)):
                        write_csv(folder / f"{stage}-trajectory.csv", [asdict(step) for step in result.trajectory])
                    save_json(folder / "training.json", {"training_contract": contract,
                        "best_step": best_step, "stopping_step": stop_step, "best_inner_dev_nll": fit.inner.best_dev_nll,
                        "full_train_raw_nll": fit.refit.train_nll_raw, "full_train_trained_nll": fit.refit.train_nll_trained,
                        "gradients_finite": fit.inner.gradients_finite and fit.refit.gradients_finite,
                        "initialization_identical_inner_refit": True, "optimizer_steps": fit.refit.optimizer_steps})
                case _:
                    raise AssertionError(label)
            model.eval()
            model.zero_grad(set_to_none=True)
            prediction, logits = evaluate_retinal_model(model, data.validation)
            effects, validation_values = pathway_effects(model, data.validation)
            assert torch.equal(logits, validation_values["normal"])
            paired, stimulus_values = illusion(model, FitIdentity(cell, label, seed))
            if label == "primary":
                assert prediction.population_nll == cell.validation_nll_trained
                assert all(torch.equal(stimulus_values[mode], saved_illusions[cell.cell_id][mode]["logit"]) for mode in stimulus_values)
            row = {"cell_id": cell.cell_id, "group": cell.group, "fit": label, "seed": seed,
                "validation_nll": prediction.population_nll, "delta_nll_vs_primary": prediction.population_nll - cell.validation_nll_trained,
                "best_step": best_step, "stopping_step": stop_step, **effects,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha(checkpoint_path)}
            results.append(row)
            signatures.extend(paired)
            torch.save({"validation_logits": validation_values, "target": data.validation.spike_events,
                "valid_mask": data.validation.valid_mask, "source_image_ids": data.validation.source_image_ids,
                "trial_indices": data.validation.trial_indices, "illusion_logits": stimulus_values}, folder / "evaluation.pt")
            save_json(folder / "results.json", row)
            write_csv(OUT / "per_fit.csv", results)
            write_csv(OUT / "illusion_paired_logits.csv", signatures)
            checks.append({"cell_id": cell.cell_id, "fit": label, "clamps_exact_zero": True,
                "state_unchanged": True, "all_outputs_finite": True, "paired_controls_exact_zero": True,
                "primary_exact_replay": label == "primary", "validation_used_for_selection": False})
            print(f"DONE {cell.cell_id} {label} NLL={prediction.population_nll:.9f} best={best_step} stop={stop_step} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    assert training_count == 8 and len(results) == 12
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    save_json(OUT / "verification.json", {"status": "COMPLETED", "new_fits": training_count,
        "all_source_hashes_unchanged": True, "source_sha256": hashes, "checks": checks})


if __name__ == "__main__":
    main()
