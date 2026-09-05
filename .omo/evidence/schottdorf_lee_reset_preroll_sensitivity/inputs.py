#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "opencv-python"]
# ///
# How to run: imported by run.py using D:/anaconda/python.exe -B.
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Final

import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
SOURCE: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT.parent / "schottdorf_lee_150hz_multispike_sanity"))

from compute import Manifest, Snapshot, sha
from data.retinal_recording import RealSequenceSplit
from data.schottdorf_lee_catalog import SchottdorfRecording, mc_pc_recordings
from data.schottdorf_lee_multirecording import SchottdorfCellwiseData, SchottdorfMovieDrive, load_schottdorf_cell, load_schottdorf_movie_drive
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, build_mechanistic_retina


@dataclass(frozen=True, slots=True)
class Inputs:
    model: MechanisticGraphTemporalRetina
    validation: RealSequenceSplit
    expected_logits: torch.Tensor
    continuous_cones: torch.Tensor
    continuous_history: torch.Tensor
    preroll_cones: torch.Tensor
    preroll_history: torch.Tensor
    stream_indices: torch.Tensor
    scored_indices: torch.Tensor
    stream_ids: tuple[tuple[str, int], ...]
    scored_identity_sha256: str


@dataclass(frozen=True, slots=True)
class Source:
    snapshot: Snapshot
    recordings: tuple[SchottdorfRecording, ...]
    movie: SchottdorfMovieDrive
    hashes: dict[str, str]
    core_source_drift: tuple[str, ...]


def prepare_source() -> Source:
    snapshot = Snapshot.model_validate_json((SOURCE / "results.json").read_text())
    manifest = Manifest.model_validate_json((SOURCE / "run-manifest.json").read_text())
    provenance = json.loads((SOURCE / "comparison.json").read_text())["source_sha256"]
    expected = {Path(p).resolve(): h for p, h in provenance.items()}
    assert snapshot.cell_count == 22 and snapshot.recording_count == 37
    assert tuple(c.cell_id for c in snapshot.cells) == manifest.cell_ids
    assert snapshot.causal_contract == "h1-shared-bc-direct-broad-ac"
    recordings = mc_pc_recordings(ROOT / "data/real/schottdorf_lee_2021_repository/data")
    assert {r.recording_id for r in recordings} == {rid for c in snapshot.cells for rid in c.recording_ids}
    movie_path = ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"
    required = [r.path for r in recordings] + [movie_path]
    required += [SOURCE / "cells" / c.cell_id.replace("#", "_") / "model-trained.pt" for c in snapshot.cells]
    required += [ROOT / p for p in ("data/schottdorf_lee_2021.py", "data/schottdorf_lee_multirecording.py",
        "data/schottdorf_lee_spikes.py", "data/schottdorf_lee_catalog.py", "data/retinal_recording.py",
        "training/mechanistic_retina/losses.py", "training/mechanistic_retina/r4_development.py")]
    hashes = {}
    for path in required:
        hashes[str(path)] = sha(path)
        assert hashes[str(path)] == expected[path.resolve()], f"STOP: frozen source mismatch {path}"
    drift = []
    for path in (ROOT / "models/mechanistic_retina").glob("*.py"):
        hashes[str(path)] = sha(path)
        if hashes[str(path)] != expected.get(path.resolve()):
            drift.append(str(path))
    for path in (SOURCE / "results.json", SOURCE / "run-manifest.json", SOURCE / "comparison.json",
                 OUT.parent / "schottdorf_lee_150hz_multispike_sanity/compute.py", *OUT.glob("*.py")):
        hashes[str(path)] = sha(path)
    return Source(snapshot, recordings, load_schottdorf_movie_drive(movie_path, snapshot.adapter_config), hashes, tuple(drift))


def timelines(data: SchottdorfCellwiseData) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[str, int], ...]]:
    streams = {}
    for split in (data.train, data.validation):
        for row, identity in enumerate(split.source_image_ids):
            match = re.fullmatch(r"(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)", identity)
            assert match is not None
            rid, start, stop, trial = match.groups()
            assert int(stop) - int(start) == 149
            key = (rid, int(trial))
            pieces = streams.setdefault(key, {})
            assert int(start) not in pieces, "STOP: duplicate segment"
            pieces[int(start)] = (split.cone_drive[row], split.spike_events[row])
    cones, histories = [], []
    for pieces in streams.values():
        assert sorted(pieces) == list(range(0, 3000, 150)), "STOP: preceding data discontinuity"
        cones.append(torch.cat([pieces[t][0] for t in sorted(pieces)]))
        histories.append(torch.cat([pieces[t][1] for t in sorted(pieces)]))
    return torch.stack(cones), torch.stack(histories), tuple(streams)


def load_cell(source: Source, index: int) -> Inputs:
    cell = source.snapshot.cells[index]
    recordings = tuple(r for r in source.recordings if r.cell_id == cell.cell_id)
    assert tuple(r.recording_id for r in recordings) == cell.recording_ids
    data = load_schottdorf_cell(recordings, source.movie, source.snapshot.adapter_config)
    val = data.validation
    assert val.spike_events.shape == (cell.validation_sequences, 150, 1)
    assert int(val.valid_mask.sum()) == cell.validation_valid_bins
    assert torch.equal(val.valid_mask, (torch.arange(150) >= 30).view(1, 150, 1).expand_as(val.valid_mask))
    folder = SOURCE / "cells" / cell.cell_id.replace("#", "_")
    saved_path = folder / "validation-predictions.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=True)
    source.hashes[str(saved_path)] = sha(saved_path)
    assert torch.equal(saved["target"], val.spike_events) and torch.equal(saved["valid_mask"], val.valid_mask)
    assert tuple(saved["source_image_ids"]) == val.source_image_ids
    assert tuple(saved["trial_indices"]) == val.trial_indices
    checkpoint = torch.load(folder / "model-trained.pt", map_location="cpu", weights_only=True)
    assert checkpoint["cell_id"] == cell.cell_id and checkpoint["stage"] == "trained"
    config = MechanisticRetinaConfig(**(checkpoint["model_config"] | {"architecture_mode": ArchitectureMode.MECHANISM_IDENTIFIABLE}))
    assert config.dt_ms == data.dt_ms == 1000 / 150
    model = build_mechanistic_retina(config, data.cone_positions_degs, data.cell_positions_degs, data.cell_types, data.polarities)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    cones, history, stream_ids = timelines(data)
    pre_x, pre_y, stream_rows, scored_rows = [], [], [], []
    for row, identity in enumerate(val.source_image_ids):
        match = re.fullmatch(r"(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)", identity)
        assert match is not None
        rid, start, stop, trial = match.groups()
        stream = stream_ids.index((rid, int(trial)))
        first, end = int(start) + 30, int(stop) + 1
        assert first - 60 >= 0 and end <= history.shape[1]
        assert torch.equal(cones[stream, int(start):end], val.cone_drive[row])
        assert torch.equal(history[stream, int(start):end], val.spike_events[row])
        pre_x.append(cones[stream, first - 60:end])
        pre_y.append(history[stream, first - 60:end])
        stream_rows.append(stream)
        scored_rows.append(torch.arange(first, end))
    scored = torch.stack(scored_rows)
    stream_indices = torch.tensor(stream_rows)
    assert torch.equal(history[stream_indices[:, None], scored], val.spike_events[:, 30:])
    pre_cones, pre_history = torch.stack(pre_x), torch.stack(pre_y)
    assert torch.equal(pre_history[:, 60:], val.spike_events[:, 30:])
    assert torch.equal(pre_cones[:, 60:], val.cone_drive[:, 30:])
    identity_bytes = json.dumps(val.source_image_ids).encode() + val.valid_mask.numpy().tobytes()
    return Inputs(model, val, saved["logits_trained"], cones, history, pre_cones, pre_history,
                  stream_indices, scored, stream_ids, hashlib.sha256(identity_bytes).hexdigest())
