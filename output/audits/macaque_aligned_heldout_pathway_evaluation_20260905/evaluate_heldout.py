# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "opencv-python", "pydantic"]
# ///
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
INITIAL = json.loads((OUT / "evidence_manifest.initial.json").read_text())
PRIMARY = ROOT / json.loads((OUT / "production_usage.json").read_text())["aligned_root_resolved_from_analysis_manifest"]
ZERO = ROOT / INITIAL["lineages"]["zero_center_reference"]
LN = ROOT / INITIAL["lineages"]["LN"]
CNN = ROOT / INITIAL["lineages"]["CNN"]
sys.path.insert(0, str(PRIMARY))
sys.path.insert(0, str(ROOT))
from preflight import ADAPTER, MOVIE, REPOSITORY, factory, folder
from baselines.center_surround_ln import CenterSurroundLN
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll

CLAMPS = {
    "H1": frozenset({PathwayClamp.H1}),
    "direct-BC": frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
    "AC": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_logits(model: MechanisticGraphTemporalRetina, split: RealSequenceSplit,
                     clamps: frozenset[PathwayClamp] = frozenset()) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model.forward_sequence(split.cone_drive[i:i+8],
            observed_counts=split.spike_events[i:i+8], clamps=clamps).logits for i in range(0, split.cone_drive.shape[0], 8)])


def main() -> None:
    assert not (OUT / "TEST_CONSUMED.md").exists(), "No automatic restart after consumption"
    assert not (OUT / "evaluation_status.json").exists()
    preflight = json.loads((OUT / "preflight.json").read_text())
    assert preflight["all_passed"] and len(preflight["cells"]) == 88
    assert all(r["passed"] for r in preflight["cells"])
    protocol = json.loads((OUT / "protocol_hash.json").read_text())["sha256"]
    assert sha(OUT / "TEST_PROTOCOL.md") == protocol
    assert str(torch.__version__) == preflight["runtime"]["torch"]
    torch.set_num_threads(2)
    cells = json.loads((ZERO / "results.json").read_text())["cells"]
    alignment_file = ROOT / INITIAL["lineages"]["fixed_alignment_audit"] / "per_cell_results.csv"
    with alignment_file.open(encoding="utf-8") as stream:
        alignment_rows = {r["cell_id"]: r for r in csv.DictReader(stream)}
    records = mc_pc_recordings(REPOSITORY / "data")
    status = dict(status="IN_PROGRESS", protocol_sha256=protocol, completed_cells=0,
        start_utc=datetime.now(timezone.utc).isoformat(), new_training_runs=0, production_modifications=0)
    state_path = OUT / "evaluation_status.json"
    state_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    rows, pathways, sources, checks = [], [], [], []

    def consume(cid: str, label: str) -> None:
        if (OUT / "TEST_CONSUMED.md").exists():
            return
        now = datetime.now(timezone.utc).isoformat()
        text = "# Test consumption record\n\nThis held-out range has now been consumed for confirmatory evaluation.\n\n"
        text += f"First successful target-based NLL UTC: {now}; first cell/model: {cid} / {label}.\n\n"
        text += "The entire declared [20,60) s range (live frames [3000,9000), decoded movie frames [3751,9751)) is conservatively retired at this first result. Completion at creation: PARTIAL; evaluation_status.json records subsequent completion or failure.\n\n"
        text += f"Branch: {INITIAL['branch']}; HEAD: {INITIAL['HEAD']}; protocol SHA256: {protocol}.\n\n"
        text += "Cells: " + ", ".join(c["cell_id"] for c in cells) + ".\n\nRecordings: " + ", ".join(r.recording_id for r in records) + ".\n\n"
        text += "Exact local trials are the immutable TEST_PROTOCOL.md table (137 total). Models: frozen zero Canonical, aligned Canonical, final LN, final CNN; analytical test-target Constant. Aligned conditions: normal, H1-off, direct-BC-off, AC-off; bias/history/remaining parameters fixed. Metrics: valid-bin Bernoulli NLL; paired model/pathway differences and cell-bootstrap intervals; declared logit effects and correlations.\n\n"
        text += "After result-informed changes to architecture, alignment, loss, front-end or center estimation, this range cannot be described as an untouched confirmatory test. No new model selection or range selection is authorized.\n"
        with (OUT / "TEST_CONSUMED.md").open("x", encoding="utf-8") as stream:
            stream.write(text)

    def score(cid: str, label: str, logits: torch.Tensor, split: RealSequenceSplit) -> float:
        assert logits.shape == split.spike_events.shape and torch.isfinite(logits).all()
        loss = expected_bernoulli_nll(logits, split.spike_events, split.valid_mask)
        assert torch.isfinite(loss)
        consume(cid, label)
        return float(loss)

    try:
        for path, digest in INITIAL["input_sha256"].items():
            assert sha(ROOT / path) == digest, "Frozen input changed before test: " + path
        extent = SchottdorfAdapterConfig(**{**asdict(ADAPTER), "train_sequence_count": 20, "validation_sequence_count": 40})
        movie = load_schottdorf_movie_drive(MOVIE, extent)
        gpu_input = dict(protocol_sha256=protocol, movie_sequences=torch.from_numpy(movie.sequences), cells={})
        output_cells = OUT / "cells"
        output_cells.mkdir()
        pattern = re.compile(r"(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)")
        for cell in cells:
            cid = cell["cell_id"]
            selected = tuple(r for r in records if r.cell_id == cid)
            assert list(r.recording_id for r in selected) == cell["recording_ids"]
            data = load_schottdorf_cell(selected, movie, extent)
            split = data.validation
            assert split.cone_drive.shape[1:] == (150, 289)
            assert not split.valid_mask[:, :30].any() and split.valid_mask[:, 30:].all()
            assert torch.equal(split.spike_events, (split.spike_counts > 0).float())
            indices = []
            for index, (identity, trial) in enumerate(zip(split.source_image_ids, split.trial_indices, strict=True)):
                match = pattern.fullmatch(identity)
                assert match is not None
                rid, begin, end, local = match.groups()
                begin, end = int(begin), int(end)
                assert 3000 <= begin <= 8850 and begin % 150 == 0 and end == begin + 149
                indices.append(begin // 150)
                sources.append(dict(cell_id=cid, recording_id=rid, local_trial_one_based=int(local),
                    cell_global_trial_zero_based=trial, sequence_index=index, source_image_id=identity,
                    live_start=begin, live_stop_exclusive=end+1, scored_start=begin+30, scored_stop_exclusive=end+1))
            assert torch.equal(split.cone_drive, torch.from_numpy(movie.sequences)[indices])
            normal_models = {}
            states = {}
            logits = {}
            values = {}
            row = dict(cell_id=cid, group=alignment_rows[cid]["group"], recording_ids=";".join(data.recording_ids),
                trials=data.trial_count, sequences=split.cone_drive.shape[0], valid_bins=int(split.valid_mask.sum()),
                live_start=3000, live_stop_exclusive=9000, protocol_sha256=protocol)
            for label, root in (("zero", ZERO), ("aligned", PRIMARY)):
                cp = torch.load(folder(cid, root) / "model-trained.pt", weights_only=True)
                model = factory(cid, data, cp["cell_positions_degs"])
                model.load_state_dict(cp["model"], strict=True)
                assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33
                normal_models[label], states[label] = model, cp["model"]
                logits[label] = canonical_logits(model, split)
                values[label] = score(cid, label, logits[label], split)
                row[label + "_nll"] = values[label]
            cp = torch.load(folder(cid, LN) / "ln-trained.pt", weights_only=True)
            ln_model = CenterSurroundLN(cp["history"]["dt_ms"], cp["history"]["tau_ms"], cp["seed"])
            ln_model.load_state_dict(cp["model"], strict=True)
            ln_model.eval()
            with torch.no_grad():
                logits["LN"] = torch.cat([ln_model(split.cone_drive[i:i+8], split.spike_events[i:i+8])
                                           for i in range(0, split.cone_drive.shape[0], 8)])
            row["ln_nll"] = score(cid, "LN", logits["LN"], split)
            assert all(torch.equal(v, cp["model"][k]) for k, v in ln_model.state_dict().items())
            rate = float(split.spike_events[split.valid_mask].double().mean())
            row["constant_nll"] = -rate * math.log(rate) - (1-rate) * math.log1p(-rate) if 0 < rate < 1 else 0.0
            row["constant_test_fitted_event_probability"] = rate
            for pathway, clamps in CLAMPS.items():
                off = canonical_logits(normal_models["aligned"], split, clamps)
                off_nll = score(cid, pathway + "-off", off, split)
                difference = (off.double()-logits["aligned"].double())[split.valid_mask]
                logits[pathway + "-off"] = off
                pathways.append(dict(cell_id=cid, group=row["group"], pathway=pathway, normal_nll=values["aligned"],
                    off_nll=off_nll, delta_nll=off_nll-values["aligned"], signed_mean_delta_logit=float(difference.mean()),
                    mean_abs_delta_logit=float(difference.abs().mean()), rms_delta_logit=float(difference.square().mean().sqrt()),
                    full_delta_logit_vector_norm=float(torch.linalg.vector_norm(difference)), valid_bins=difference.numel(),
                    protocol_sha256=protocol))
            for label, model in normal_models.items():
                assert all(torch.equal(v, states[label][k]) for k, v in model.state_dict().items())
                assert all(p.grad is None for p in model.parameters())
            gpu_input["cells"][cid] = dict(sequence_indices=indices, events=split.spike_events)
            torch.save(dict(cell_id=cid, protocol_sha256=protocol, target=split.spike_events, spike_counts=split.spike_counts,
                valid_mask=split.valid_mask, source_image_ids=split.source_image_ids, trial_indices=split.trial_indices,
                logits=logits), output_cells / (cid.replace("#", "_") + "-test-predictions.pt"))
            checks.append(dict(cell_id=cid, input_sha256=tensor_sha(split.cone_drive), target_sha256=tensor_sha(split.spike_events),
                mask_sha256=tensor_sha(split.valid_mask), trial_count=data.trial_count, source_ID_count=len(split.source_image_ids),
                source_order_sha256=hashlib.sha256(json.dumps(split.source_image_ids).encode()).hexdigest(),
                normal_clamp_checkpoint_states_unchanged=True, all_conditions_identical_target_mask_order=True))
            rows.append(row)
            status["completed_cells"] = len(rows)
            state_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"Held-out frozen CPU inference {len(rows)}/22 {cid}", flush=True)
        assert len(rows) == 22 and len(sources) == 5480 and sum(r["valid_bins"] for r in rows) == 657600
        torch.save(gpu_input, OUT / "cnn_test_inputs.pt")
        subprocess.run(["D:/anaconda/envs/snn_env/python.exe", "-B", "-u", str(OUT / "heldout_cnn_gpu.py")], cwd=ROOT, check=True)
        cnn = torch.load(OUT / "cnn_test_logits.pt", weights_only=True)
        assert cnn["protocol_sha256"] == protocol
        for row in rows:
            cid = row["cell_id"]
            path = output_cells / (cid.replace("#", "_") + "-test-predictions.pt")
            saved = torch.load(path, weights_only=True)
            cnn_logits = cnn["logits"][cid]
            assert cnn_logits.shape == saved["target"].shape and torch.isfinite(cnn_logits).all()
            row["cnn_nll"] = float(expected_bernoulli_nll(cnn_logits, saved["target"], saved["valid_mask"]))
            assert math.isfinite(row["cnn_nll"])
            saved["logits"]["CNN"] = cnn_logits
            torch.save(saved, path)
            row["aligned_minus_zero"] = row["aligned_nll"]-row["zero_nll"]
            row["aligned_minus_ln"] = row["aligned_nll"]-row["ln_nll"]
            row["aligned_minus_cnn"] = row["aligned_nll"]-row["cnn_nll"]
        write_csv("per_cell_test_nll.csv", rows)
        write_csv("pathway_per_cell.csv", pathways)
        write_csv("test_source_ranges.csv", sources)
        (OUT / "test_identity_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        assert sha(OUT / "TEST_PROTOCOL.md") == protocol
        status.update(status="COMPLETE", completed_cells=22, completed_utc=datetime.now(timezone.utc).isoformat(),
            predictions_per_cell=7, valid_bins=657600, sequences=5480, checkpoint_states_unchanged=True)
        state_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        with (OUT / "TEST_CONSUMED.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\nCompletion update UTC: {status['completed_utc']}. All 22 cells and all declared frozen conditions completed; source ranges and identities are in test_source_ranges.csv and test_identity_checks.json.\n")
        print("ALL HELD-OUT CONDITIONS COMPLETE; RANGE CONSUMED", flush=True)
    except Exception as error:
        status.update(status="STOP_EXECUTION_FAILED", error=str(error), traceback=traceback.format_exc(),
            consumed=(OUT / "TEST_CONSUMED.md").exists(), stopped_utc=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
