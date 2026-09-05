#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy==2.2.6", "pydantic==2.8.2", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B .omo/evidence/compact_causal_cnn_baseline/prepare.py
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Final

import numpy as np
import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines.center_surround_ln import LNError
from data.schottdorf_lee_multirecording import load_schottdorf_cell
from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev
from evaluation.mechanistic_retina.spatial_contrast_source import (
    compare_saved_predictions, load_center_source, load_sources, prediction_nll, verify_split,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from training.mechanistic_retina.center_surround_ln import BATCH_SIZE, MAX_STEPS, SEED, PATIENCE, MIN_DELTA
from training.mechanistic_retina.losses import expected_bernoulli_nll


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    if (OUT / "preflight.json").exists() or (OUT / "inputs").exists():
        raise LNError("STOP: CNN preparation already exists")
    torch.set_num_threads(2)
    sources = load_sources(ROOT)
    sc_root = ROOT / ".omo/evidence/spatial_contrast_adapted"
    sc = json.loads((sc_root / "results.json").read_text())
    sc_preflight = json.loads((sc_root / "preflight.json").read_text())
    ln_rows = json.loads((sources.ln_root / "results.json").read_text())["cells"]
    if sc["status"] != "COMPLETED_CORRECTED" or not sc["frozen_benchmark_accepted"]:
        raise LNError("STOP: corrected frozen SC reference is unavailable")
    sources.hashes.update({str(p): sha256_file(p) for p in sc_root.rglob("*") if p.is_file()})
    for directory in ("baselines", "models/mechanistic_retina", "training/mechanistic_retina", "tests"):
        for path in (ROOT / directory).glob("*.py"):
            sources.hashes[str(path)] = sha256_file(path)
    for path in OUT.glob("*.py"):
        sources.hashes[str(path)] = sha256_file(path)
    for name in ("factorized_ln_split.py", "spatial_contrast_source.py", "schottdorf_multirecording_reporting.py"):
        path = ROOT / "evaluation/mechanistic_retina" / name
        sources.hashes[str(path)] = sha256_file(path)
    (OUT / "inputs").mkdir()
    rows = []
    for cell in sources.population.cells:
        dirname = cell.cell_id.replace("#", "_")
        recordings = tuple(r for r in sources.recordings if r.cell_id == cell.cell_id)
        if tuple(r.recording_id for r in recordings) != cell.recording_ids:
            raise LNError("STOP: recording order changed")
        data = load_schottdorf_cell(recordings, sources.movie, sources.population.adapter_config)
        _, checkpoint = load_center_source(sources, cell)
        prior = next(c for c in sc_preflight["checks"] if c["cell_id"] == cell.cell_id)
        identities = {}
        for label, split, bins in (("train", data.train, cell.train_valid_bins),
                                   ("validation", data.validation, cell.validation_valid_bins)):
            verify_split(split, bins)
            for field, tensor in (("input", split.cone_drive), ("target", split.spike_events), ("mask", split.valid_mask)):
                key = f"{label}_{field}_sha256"
                identities[key] = tensor_sha(tensor)
                if identities[key] != prior[key]:
                    raise LNError(f"STOP: {cell.cell_id} {key} differs from frozen data")
        baseline = next(r for r in sc["cells"] if r["cell_id"] == cell.cell_id)
        for name, root in (("center_surround_ln_nll", sources.ln_root), ("canonical_v1_nll", sources.retinal_root)):
            path = root / "cells" / dirname / "validation-predictions.pt"
            sources.hashes[str(path)] = sha256_file(path)
            saved = compare_saved_predictions(path, data.validation)
            if prediction_nll(saved) != baseline[name]:
                raise LNError("STOP: frozen comparison NLL changed")
        rate = data.train.spike_events[data.train.valid_mask].mean().clamp(1e-6, 1 - 1e-6)
        constant = float(expected_bernoulli_nll(torch.full_like(data.validation.spike_events, torch.logit(rate)),
                                               data.validation.spike_events, data.validation.valid_mask))
        ln_row = next(r for r in ln_rows if r["cell_id"] == cell.cell_id)
        if abs(constant - ln_row["constant_nll"]) > 1e-7:
            raise LNError("STOP: constant comparison differs")
        inner = make_inner_dev(data.train)
        path = OUT / "inputs" / f"{dirname}.pt"
        torch.save(dict(train=asdict(data.train), validation=asdict(data.validation),
                        history=checkpoint.history.model_dump(), cell_id=cell.cell_id), path)
        sources.hashes[str(path)] = sha256_file(path)
        rows.append(dict(cell_id=cell.cell_id, group=f"{cell.retinal_class}_{cell.polarity}", recording_ids=cell.recording_ids,
                         dt_ms=data.dt_ms, train_bins=cell.train_valid_bins, validation_bins=cell.validation_valid_bins,
                         inner_train_bins=int(inner.train.valid_mask.sum()), inner_dev_bins=int(inner.development.valid_mask.sum()),
                         inner_boundaries=[asdict(b) for b in inner.boundaries], tensor_sha256=identities,
                         history=checkpoint.history.model_dump(), constant_nll=constant,
                         ln_nll=baseline["center_surround_ln_nll"], sc_adapted_nll=baseline["sc_adapted_nll"],
                         canonical_v1_nll=baseline["canonical_v1_nll"]))
        print(f"PREPARED {cell.cell_id}: frozen input/target/mask/order exact", flush=True)
    contract = dict(model="compact causal CNN", convolutions=[dict(channels=[1, 4], kernel=[12, 5, 5], temporal_dilation=1),
                        dict(channels=[4, 4], kernel=[9, 3, 3], temporal_dilation=6)], activation="ReLU after each Conv3D",
                    spatial_padding=0, spatial_shapes=[17, 13, 11], temporal_left_padding=[11, 48], receptive_field_bins=60,
                    readout="learned 4x11x11 spatial/channel linear weights, one scalar logit per bin",
                    head="LN: stimulus logit + history_weight * fixed_one_bin_history_state + bias",
                    history_tau="unchanged per-cell LN artifact metadata", sequence_reset="original 150-bin sequences; score bins 30..149",
                    loss="unregularized Bernoulli NLL on identical masks", optimizer="Adam; default betas/eps; weight_decay=0",
                    learning_rates=[1e-3, 3e-4], batch_size=BATCH_SIZE, seed=SEED, max_steps=MAX_STEPS,
                    patience=PATIENCE, min_delta=MIN_DELTA, dev_interval=1, best_step_zero_eligible=True,
                    inner_split="exact make_inner_dev from LN: per-trial 80/20 with 60-bin guard",
                    selection="lowest inner-dev NLL; first LR in declared order on exact tie",
                    refit="fresh initialization and optimizer; selected LR and best-step count on full train",
                    initialization="seeded PyTorch Conv3D defaults; readout U(-1/sqrt(484),+1/sqrt(484)); history weight=0; fitting-only constant logit bias",
                    original_validation_used_for_selection=False, architecture_sweep=False,
                    parameters_per_cell=2990, source_ln_parameters=128, sc_inherited_plus_fitted_raw=68,
                    canonical_parameters_total=129, canonical_parameters_trainable=33,
                    data_runtime=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__))
    payload = dict(status="PREPARED", contract=contract, cells=rows, source_sha256=sources.hashes,
                   git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   git_status=subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True))
    (OUT / "preflight.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
