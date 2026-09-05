#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B -u compute.py
# Use the existing frozen runtime; do not install or resolve new dependencies.
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Final, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
LINEAGE: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
sys.path.insert(0, str(ROOT))

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
from data.schottdorf_lee_spikes import parse_recording_spike_trials


class Cell(BaseModel):
    model_config = ConfigDict(frozen=True)
    cell_id: str
    recording_ids: tuple[str, ...]
    retinal_class: str
    polarity: str
    train_sequences: int
    validation_sequences: int
    train_valid_bins: int
    validation_valid_bins: int
    native_dt_ms: float


class Snapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    causal_contract: str
    cell_count: int
    recording_count: int
    adapter_config: SchottdorfAdapterConfig
    cells: tuple[Cell, ...]


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_sha256: dict[str, str]
    cell_ids: tuple[str, ...]


class Counts(TypedDict):
    all_bins: int
    nonzero_bins: int
    multispike_bins: int
    total_spikes: int
    excess_spikes: int
    multispike_over_all: float
    multispike_over_nonzero: float
    excess_over_total_spikes: float


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def metrics(counts: np.ndarray) -> Counts:
    assert np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0)
    bins, nonzero = counts.size, int(np.count_nonzero(counts))
    multi, spikes = int(np.count_nonzero(counts >= 2)), int(counts.sum())
    excess = int(np.maximum(counts - 1, 0).sum())
    assert excess == spikes - nonzero and bins > 0 and nonzero > 0 and spikes > 0
    return Counts(all_bins=bins, nonzero_bins=nonzero, multispike_bins=multi,
                total_spikes=spikes, excess_spikes=excess,
                multispike_over_all=multi / bins, multispike_over_nonzero=multi / nonzero,
                excess_over_total_spikes=excess / spikes)


def main() -> None:
    # Given integer counts, verify denominators and excess accounting before data I/O.
    test = metrics(np.array([0, 1, 2, 3], dtype=np.int64))
    assert [test[k] for k in ("all_bins", "nonzero_bins", "multispike_bins", "total_spikes", "excess_spikes")] == [4, 3, 2, 6, 3]
    assert test["multispike_over_all"] == .5 and test["multispike_over_nonzero"] == 2 / 3
    assert test["excess_over_total_spikes"] == .5
    snapshot = Snapshot.model_validate_json((LINEAGE / "results.json").read_text())
    manifest = Manifest.model_validate_json((LINEAGE / "run-manifest.json").read_text())
    assert snapshot.causal_contract == "h1-shared-bc-direct-broad-ac"
    assert snapshot.cell_count == len(snapshot.cells) == 22 and snapshot.recording_count == 37
    assert tuple(c.cell_id for c in snapshot.cells) == manifest.cell_ids
    config = snapshot.adapter_config
    assert (config.train_sequence_count, config.validation_sequence_count, config.sequence_steps, config.warmup_steps) == (16, 4, 150, 30)
    expected_hashes = {Path(p).resolve(): h for p, h in manifest.source_sha256.items()}
    required = [ROOT / p for p in (
        "data/retinal_recording.py", "data/schottdorf_lee_2021.py", "data/schottdorf_lee_catalog.py",
        "data/schottdorf_lee_multirecording.py", "data/schottdorf_lee_spikes.py",
        "training/mechanistic_retina/r4_development.py", "training/mechanistic_retina/losses.py",
        "training/mechanistic_retina/real_sampled.py", "evaluation/mechanistic_retina/karamanlis_prediction_baselines.py")]
    movie_path = ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"
    recordings = mc_pc_recordings(ROOT / "data/real/schottdorf_lee_2021_repository/data")
    selected_ids = tuple(rid for c in snapshot.cells for rid in c.recording_ids)
    assert len(selected_ids) == len(set(selected_ids)) == 37
    assert set(selected_ids) == {r.recording_id for r in recordings}
    required += [r.path for r in recordings] + [movie_path, LINEAGE / "run.py"]
    hashes = {}
    for path in required:
        digest = sha(path)
        assert digest == expected_hashes[path.resolve()], f"STOP: source mismatch {path}"
        hashes[str(path)] = digest
    for path in (LINEAGE / "results.json", LINEAGE / "run-manifest.json", Path(__file__)):
        hashes[str(path)] = sha(path)
    for cell in snapshot.cells:
        assert (LINEAGE / "cells" / cell.cell_id.replace("#", "_") / "validation-predictions.pt").is_file()
    print("PREFLIGHT PASS: final manifest, 37 raw files, exact source/loss contract recoverable.", flush=True)
    torch.set_num_threads(2)
    movie = load_schottdorf_movie_drive(movie_path, config)
    per_cell, consistency, pooled = [], [], {}
    for cell in snapshot.cells:
        selected = tuple(r for r in recordings if r.cell_id == cell.cell_id)
        assert tuple(r.recording_id for r in selected) == cell.recording_ids
        assert all((r.retinal_class, r.polarity) == (cell.retinal_class, cell.polarity) for r in selected)
        data = load_schottdorf_cell(selected, movie, config)
        assert data.dt_ms == cell.native_dt_ms == 1000 / 150
        independent = {}
        for recording in selected:
            trials = parse_recording_spike_trials(recording).live_times_ms_by_trial
            for trial, times in enumerate(trials):
                # Same frozen bin arithmetic; independent accumulation before binary conversion.
                indices = np.floor(times.numpy() * 150.0 / 1000.0).astype(np.int64)
                counts = np.zeros(3000, dtype=np.int64)
                np.add.at(counts, indices[(indices >= 0) & (indices < 3000)], 1)
                independent[(recording.recording_id, trial)] = counts
        saved_path = LINEAGE / "cells" / cell.cell_id.replace("#", "_") / "validation-predictions.pt"
        saved = torch.load(saved_path, map_location="cpu", weights_only=True)
        assert set(saved) == {"logits_raw", "logits_trained", "probabilities_trained", "target", "valid_mask", "source_image_ids", "trial_indices"}
        hashes[str(saved_path)] = sha(saved_path)
        group = f"{cell.retinal_class} {cell.polarity}"
        cell_counts = []
        for label, split, nseq, nvalid in (
            ("train", data.train, cell.train_sequences, cell.train_valid_bins),
            ("validation", data.validation, cell.validation_sequences, cell.validation_valid_bins)):
            raw, target, mask = (t.numpy() for t in (split.spike_counts, split.spike_events, split.valid_mask))
            assert raw.shape == target.shape == mask.shape == (nseq, 150, 1)
            assert mask.dtype == bool and int(mask.sum()) == nvalid
            expected_mask = np.broadcast_to((np.arange(150) >= 30)[None, :, None], raw.shape)
            assert np.array_equal(mask, expected_mask), "STOP: exact loss mask unavailable"
            rebuilt = []
            for source_id in split.source_image_ids:
                match = re.fullmatch(r"(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)", source_id)
                assert match is not None
                rid, start, stop, trial = match.groups()
                rebuilt.append(independent[(rid, int(trial) - 1)][int(start):int(stop) + 1, None])
            assert np.array_equal(np.stack(rebuilt), raw), "STOP: integer counts disagree"
            mismatch = int(np.count_nonzero(((raw > 0) != target)[mask]))
            assert mismatch == 0, "STOP: binary target mismatch"
            saved_match = None
            if label == "validation":
                assert torch.equal(saved["target"], split.spike_events)
                assert torch.equal(saved["valid_mask"], split.valid_mask)
                assert tuple(saved["source_image_ids"]) == split.source_image_ids
                assert tuple(saved["trial_indices"]) == split.trial_indices
                saved_match = True
            flat = raw[mask]
            cell_counts.append(flat)
            pooled.setdefault((group, label), []).append(flat)
            row = dict(cell_id=cell.cell_id, group=group, split=label, recording_ids=";".join(cell.recording_ids), **metrics(flat))
            per_cell.append(row)
            consistency.append(dict(cell_id=cell.cell_id, split=label, compared_bins=nvalid,
                mismatches=mismatch, independent_integer_counts_exact=True, exact_loss_mask=True,
                saved_validation_target_mask_and_order_exact=saved_match,
                counts_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                production_target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        assert not set(data.train.source_image_ids) & set(data.validation.source_image_ids)
        joined = np.concatenate(cell_counts)
        per_cell.append(dict(cell_id=cell.cell_id, group=group, split="combined", recording_ids=";".join(cell.recording_ids), **metrics(joined)))
        pooled.setdefault((group, "combined"), []).append(joined)
        print(f"PASS {cell.cell_id}: train={cell.train_valid_bins} validation={cell.validation_valid_bins}", flush=True)
    groups = ("MC ON", "MC OFF", "PC ON", "PC OFF")
    labels = ("train", "validation", "combined")
    group_rows = [dict(group=g, split=s, cells=sum(f"{c.retinal_class} {c.polarity}" == g for c in snapshot.cells),
                       **metrics(np.concatenate(pooled[(g, s)]))) for g in groups for s in labels]
    population = [dict(split=s, **metrics(np.concatenate([v for g in groups for v in pooled[(g, s)]]))) for s in labels]
    assert all(sha(Path(p)) == h for p, h in hashes.items()), "STOP: input changed during computation"
    for name, rows in (("per_cell.csv", per_cell), ("group_summary.csv", group_rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = dict(status="PASS", lineage=str(LINEAGE), cell_count=22, recording_count=37,
        bin_rate_hz=150, dt_ms=1000 / 150, statistics_scope="full-refit train and original validation; valid bins only; each observation counted once",
        exact_mask="each 150-bin segment uses indices 30..149, no additional loss exclusions",
        training_target_evidence="reconstructed with manifest-hash-matched production adapter and original raw spikes; training target was not separately serialized",
        validation_target_evidence="elementwise comparison against all 22 final validation-predictions.pt target/mask/identity fields; no model states loaded",
        model_checkpoints_loaded=False, model_constructed=False, training_run=False,
        mismatch_count=sum(c["mismatches"] for c in consistency),
        compared_bins=sum(c["compared_bins"] for c in consistency),
        population=population, cells=consistency, source_sha256=hashes)
    (OUT / "production_target_consistency.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Schottdorf–Lee 150 Hz multi-spike-bin sanity", "", "STATUS: PASS", "",
        f"Lineage: `{LINEAGE}`", "", "22 cells / 37 recordings; dt=1000/150 ms. No model/checkpoint loading or training.", "",
        "All denominators use the actual loss mask. Train is the final full-train refit split, not inner-train; validation is original held-out validation. Each observed trial/bin is counted once, not once per optimizer visit. Combined concatenates these disjoint splits.", "",
        "Per recording/trial: train segments 0–15, validation 16–19, 150 bins/segment. Each segment uses bins 30–149 (120 bins). Excluded warmup bins are absent from all three ratios.", "",
        "Ratios: A=count>=2 bins/all valid bins; B=count>=2 bins/nonzero valid bins; C=sum(max(count-1,0))/sum(count). Population/group ratios pool integer numerators and denominators, not per-cell percentage averages.", "",
        "| Split | All bins | Nonzero | Count>=2 | Spikes | Excess | A | B | C |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in population:
        lines.append(f"| {r['split']} | {r['all_bins']} | {r['nonzero_bins']} | {r['multispike_bins']} | {r['total_spikes']} | {r['excess_spikes']} | {r['multispike_over_all']:.6%} | {r['multispike_over_nonzero']:.6%} | {r['excess_over_total_spikes']:.6%} |")
    lines += ["", "| Group | Cells | Split | A | B | C |", "|---|---:|---|---:|---:|---:|"]
    for r in group_rows:
        lines.append(f"| {r['group']} | {r['cells']} | {r['split']} | {r['multispike_over_all']:.6%} | {r['multispike_over_nonzero']:.6%} | {r['excess_over_total_spikes']:.6%} |")
    lines += ["", f"Target consistency: {payload['compared_bins']} valid bins checked; 0 mismatches. Raw integer counts independently accumulated with numpy.add.at equal production spike_counts. All 22 saved validation target/mask/source IDs/trial orders match exactly.", "",
        "Training targets/masks are reconstructed from original raw files using unchanged, final-manifest-hash-matched production code; no separately saved training-target tensor is claimed.", "",
        "Source locations: data/schottdorf_lee_multirecording.py:174–226 (integer binning, split and target/mask); training/mechanistic_retina/r4_development.py:86–87,120–131 (loss and full-train refit); training/mechanistic_retina/losses.py:17–28 (masked Bernoulli NLL); evaluation/mechanistic_retina/karamanlis_prediction_baselines.py:169–187 (validation target/mask). Exact source SHA256 and raw file provenance are in production_target_consistency.json.", "",
        "Files: per_cell.csv (66 rows), group_summary.csv (12 rows), production_target_consistency.json, and compute.py (reproducible count-only calculation). No data, targets, masks, or production source files were changed."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(population, indent=2), flush=True)


if __name__ == "__main__":
    main()
