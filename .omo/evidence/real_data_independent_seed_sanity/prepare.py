#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u prepare.py
# Preserve the existing numerical environment; no dependency resolution.
from __future__ import annotations

import json
from dataclasses import asdict
from statistics import median
import subprocess
from typing import Final
from pathlib import Path

import torch

from source import APPLICATION, ORIGINAL, OUT, ROOT, SOURCE, Snapshot, fresh, load_data, save_json, sha
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from training.mechanistic_retina.optimizer import phase1_parameters

GROUPS: Final = ("MC ON", "MC OFF", "PC ON", "PC OFF")


def main() -> None:
    assert not (OUT / "selection.json").exists(), "STOP: selection already exists"
    torch.set_num_threads(2)
    snapshot = Snapshot.model_validate_json((SOURCE / "results.json").read_text())
    manifest = json.loads((SOURCE / "run-manifest.json").read_text())
    assert torch.__version__ == manifest["torch_version"] == "2.6.0+cpu"
    assert snapshot.cell_count == 22 and snapshot.recording_count == 37
    assert [c.cell_id for c in snapshot.cells] == manifest["cell_ids"]
    hashes, drift = {}, []
    for name, digest in manifest["source_sha256"].items():
        path = Path(name)
        if path.suffix == ".py" or path.suffix in {".txt", ".mpg"}:
            hashes[name] = sha(path)
            if hashes[name] != digest:
                drift.append(name)
                assert "models\\mechanistic_retina\\" in name or name.endswith("karamanlis_v1_rf_validation.py"), f"STOP: training/data source changed: {name}"
    for path in [SOURCE / "results.json", SOURCE / "run-manifest.json", SOURCE / "comparison.json",
                 APPLICATION / "input-manifest.json", APPLICATION / "illusion/inputs.pt",
                 APPLICATION / "illusion/responses.pt", APPLICATION / "common.py",
                 ORIGINAL / "metrics.py", ORIGINAL / "stimuli.py",
                 ROOT / "models/mechanistic_retina/canonical_contract.py", *OUT.glob("*.py")]:
        hashes[str(path)] = sha(path)
    prior_hashes = json.loads((SOURCE / "comparison.json").read_text())["source_sha256"]
    input_hashes = json.loads((APPLICATION / "input-manifest.json").read_text())
    rows = []
    for group in GROUPS:
        members = [c for c in snapshot.cells if c.group == group]
        middle = median(c.validation_nll_trained for c in members)
        ordered = sorted(members, key=lambda c: (abs(c.validation_nll_trained - middle), c.cell_id))
        cell = ordered[0]
        rows.append({"cell_id": cell.cell_id, "group": group, "median_nll": middle,
                     "primary_nll": cell.validation_nll_trained, "primary_seed": cell.primary_seed,
                     "fresh_seeds": [cell.primary_seed + 100000, cell.primary_seed + 200000],
                     "fresh_minibatch_seeds": [cell.primary_seed + 100000 + 1000003, cell.primary_seed + 200000 + 1000003],
                     "candidates": [{"cell_id": c.cell_id, "nll": c.validation_nll_trained,
                                     "distance": abs(c.validation_nll_trained - middle)} for c in ordered],
                     "model_config": asdict(cell.configuration), "training_contract": cell.training_contract})
    save_json(OUT / "selection.json", {"saved_before_any_new_training": True,
        "representative_preassignment_found": False,
        "rule": "minimum absolute distance from within-group primary validation NLL median; ties lexical cell ID",
        "seed_rule": "primary seed +100000 / +200000; unchanged protocol derives minibatch seed +1000003",
        "cells": rows})
    git_status = subprocess.run(["git", "status", "--short", "--untracked-files=normal"], cwd=ROOT, capture_output=True, text=True, check=True)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    save_json(OUT / "provenance.json", {"git_commit": git_head, "git_status": git_status.stdout,
        "git_status_stderr": git_status.stderr, "source_sha256": hashes, "historical_source_drift": drift,
        "training_data_evaluation_sources_match_final_manifest": True, "torch_version": torch.__version__,
        "threads": 2, "original_illusion_manifest": input_hashes})
    movie = load_schottdorf_movie_drive(ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg", snapshot.adapter_config)
    checks = []
    for row in rows:
        cell = next(c for c in snapshot.cells if c.cell_id == row["cell_id"])
        for name in ("model-raw.pt", "model-trained.pt", "validation-predictions.pt", "results.json"):
            path = cell.directory / name
            hashes[str(path)] = sha(path)
            if name == "model-trained.pt":
                assert hashes[str(path)] == prior_hashes[str(path)]
        data = load_data(cell, movie, snapshot.adapter_config)
        model = fresh(cell, data, cell.primary_seed)
        raw = torch.load(cell.directory / "model-raw.pt", weights_only=True)
        assert all(torch.equal(v, raw["model"][k]) for k, v in model.state_dict().items())
        train_metrics, _ = evaluate_retinal_model(model, data.train)
        assert train_metrics.population_nll == cell.full_train_nll_raw
        counts = (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad),
                  sum(p.numel() for p in phase1_parameters(model)))
        assert counts == (129, 33, 33)
        model.load_state_dict(torch.load(cell.directory / "model-trained.pt", weights_only=True)["model"], strict=True)
        val_metrics, logits = evaluate_retinal_model(model, data.validation)
        assert val_metrics.population_nll == cell.validation_nll_trained
        assert torch.equal(logits, torch.load(cell.directory / "validation-predictions.pt", weights_only=True)["logits_trained"])
        checks.append({"cell_id": cell.cell_id, "raw_initialization_exact": True, "raw_train_nll_exact": True,
                       "primary_validation_logits_nll_exact": True, "inner_boundaries_exact": True,
                       "parameter_counts": counts})
        print("PREFLIGHT", cell.cell_id, "PASS", flush=True)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    save_json(OUT / "preflight.json", {"status": "PASS", "checks": checks, "source_sha256": hashes,
        "single_cell_core_contract": "N=1 identity mixer and standard geometry; no explicit geometry or multi-cell mixing branch",
        "historical_source_byte_identity": False, "historical_source_drift": drift})


if __name__ == "__main__":
    main()
