# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy==2.2.6", "opencv-python", "pydantic"]
# ///
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = OUT.parent / "macaque_aligned_heldout_pathway_evaluation_20260905"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD))
from evaluate_heldout import ADAPTER, CLAMPS, MOVIE, PRIMARY, REPOSITORY, CNN, canonical_logits, factory, folder
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import SchottdorfMovieDrive, load_schottdorf_cell, load_schottdorf_movie_drive
from training.mechanistic_retina.losses import expected_bernoulli_nll

CONDITIONS = {"normal": frozenset(), **CLAMPS}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def thash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_out(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_out(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def protocol_hash() -> str:
    expected = json.loads((OUT / "protocol_hash.json").read_text())["sha256"]
    assert sha(OUT / "PROTOCOL.md") == expected
    return expected


def freeze() -> None:
    assert not (OUT / "protocol_hash.json").exists()
    previous = json.loads((OLD / "evidence_manifest.json").read_text())
    hashes = dict(previous["input_sha256"])
    for relative, digest in previous["output_sha256"].items():
        assert sha(OLD / relative) == digest
    for path in OLD.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for relative, digest in hashes.items():
        assert sha(ROOT / relative) == digest, relative
    aligned = [r for r in previous["checkpoint_identity"] if r["model"] == "aligned"]
    assert len(aligned) == 22
    assert {str((ROOT / r["path"]).parents[2]) for r in aligned} == {str(PRIMARY)}
    json_out("evidence_manifest.initial.json", dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        branch=previous["branch"], HEAD=previous["HEAD"], input_sha256=hashes, aligned_checkpoint_identity=aligned,
        git_status_at_start=subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True),
        prior_protocol_sha256=previous["protocol_sha256"]))
    json_out("protocol_hash.json", dict(sha256=sha(OUT / "PROTOCOL.md"), timestamp_utc=datetime.now(timezone.utc).isoformat(),
        bootstrap_seed=2026090502, solver_rate_tolerance=1e-12, solver_width_tolerance=1e-12, solver_max_iterations=200,
        before_new_inference=True, before_bias_fitting=True))
    print("PROTOCOL FROZEN", protocol_hash(), flush=True)


def preflight() -> None:
    digest = protocol_hash()
    assert not (OUT / "preflight.json").exists()
    torch.set_num_threads(2)
    assert str(torch.__version__) == "2.6.0+cpu"
    manifest = json.loads((OUT / "evidence_manifest.initial.json").read_text())
    for relative, expected in manifest["input_sha256"].items():
        assert sha(ROOT / relative) == expected
    rows, training_sources, bundles = [], [], {}
    state = dict(status="IN_PROGRESS", all_passed=False, protocol_sha256=digest, cells=rows, bias_fits=0)
    json_out("preflight.json", state)
    try:
        cfg = SchottdorfAdapterConfig(**{**asdict(ADAPTER), "train_sequence_count": 20, "validation_sequence_count": 40})
        movie = load_schottdorf_movie_drive(MOVIE, cfg)
        development_movie = SchottdorfMovieDrive(movie.sequences[:20], movie.cone_positions_degs, movie.dt_ms, movie.stimulus_rate_hz)
        records = mc_pc_recordings(REPOSITORY / "data")
        cells = json.loads((PRIMARY / "results.json").read_text())["cells"]
        groups = {r["cell_id"]: r["group"] for r in read_csv(OLD / "per_cell_test_nll.csv")}
        test_valid = 0
        for cell in cells:
            cid = cell["cell_id"]
            selected = tuple(r for r in records if r.cell_id == cid)
            assert list(r.recording_id for r in selected) == cell["recording_ids"]
            data = load_schottdorf_cell(selected, development_movie, ADAPTER)
            train = data.train
            original = torch.load(CNN / "inputs" / (cid.replace("#", "_") + ".pt"), weights_only=True)["train"]
            for key in ("cone_drive", "spike_counts", "spike_events", "valid_mask"):
                assert torch.equal(getattr(train, key), original[key])
            assert tuple(original["source_image_ids"]) == train.source_image_ids
            assert tuple(original["trial_indices"]) == train.trial_indices
            for identity, trial in zip(train.source_image_ids, train.trial_indices, strict=True):
                training_sources.append(dict(cell_id=cid, source_image_id=identity, cell_global_trial_zero_based=trial))
            test = load_schottdorf_cell(selected, movie, cfg).validation
            saved = torch.load(OLD / "cells" / (cid.replace("#", "_") + "-test-predictions.pt"), weights_only=True)
            assert torch.equal(saved["target"], test.spike_events) and torch.equal(saved["valid_mask"], test.valid_mask)
            assert torch.equal(saved["spike_counts"], test.spike_counts)
            assert tuple(saved["source_image_ids"]) == test.source_image_ids and tuple(saved["trial_indices"]) == test.trial_indices
            cp_path = folder(cid, PRIMARY) / "model-trained.pt"
            cp = torch.load(cp_path, weights_only=True)
            assert cp["steps"] == cell["best_step"] == cp["training_contract"]["fresh_full_train_refit_steps"]
            assert not cp["training_contract"]["original_validation_used_for_selection"]
            model = factory(cid, data, cp["cell_positions_degs"])
            model.load_state_dict(cp["model"], strict=True)
            train_logits = {}
            condition_checks = []
            for condition, clamps in CONDITIONS.items():
                test_logits = canonical_logits(model, test, clamps)
                old_logits = saved["logits"]["aligned" if condition == "normal" else condition + "-off"]
                assert torch.equal(test_logits, old_logits), f"STOP: heldout logits mismatch {cid} {condition}"
                nll = float(expected_bernoulli_nll(test_logits, test.spike_events, test.valid_mask))
                old_nll = float(expected_bernoulli_nll(old_logits, saved["target"], saved["valid_mask"]))
                assert nll == old_nll
                train_logits[condition] = canonical_logits(model, train, clamps)
                assert torch.isfinite(train_logits[condition]).all()
                condition_checks.append(dict(condition=condition, heldout_logits_bitwise_exact=True, heldout_nll_exact=True,
                    logits_sha256=thash(old_logits), training_logits_sha256=thash(train_logits[condition]), original_float32_nll=nll))
            assert all(torch.equal(value, cp["model"][key]) for key, value in model.state_dict().items())
            assert all(p.grad is None for p in model.parameters())
            bundles[cid] = dict(group=groups[cid], target=train.spike_events, valid_mask=train.valid_mask,
                source_image_ids=train.source_image_ids, trial_indices=train.trial_indices, logits=train_logits)
            rows.append(dict(cell_id=cid, passed=True, conditions=condition_checks, checkpoint_sha256=sha(cp_path),
                train_target_sha256=thash(train.spike_events), train_mask_sha256=thash(train.valid_mask),
                heldout_target_sha256=thash(test.spike_events), heldout_mask_sha256=thash(test.valid_mask),
                train_source_order_exact=True, heldout_source_order_exact=True, history="unchanged production strictly-past observed binary events",
                train_bins=int(train.valid_mask.sum()), heldout_bins=int(test.valid_mask.sum()), checkpoint_unchanged=True))
            test_valid += int(test.valid_mask.sum())
            json_out("preflight.json", state)
            print("EXACT REPLAY", len(rows), "/22", cid, flush=True)
        assert len(rows) == 22 and test_valid == 657600 and len(training_sources) == 2192
        torch.save(dict(protocol_sha256=digest, cells=bundles), OUT / "training_only_logits.pt")
        csv_out("training_source_ids.csv", training_sources)
        (OUT / "heldout_source_ids.csv").write_bytes((OLD / "test_source_ranges.csv").read_bytes())
        state.update(status="PASSED", all_passed=True, train_valid_bins=sum(r["train_bins"] for r in rows),
            heldout_valid_bins=test_valid, training_only_logits_sha256=sha(OUT / "training_only_logits.pt"),
            completed_utc=datetime.now(timezone.utc).isoformat())
        json_out("preflight.json", state)
    except Exception as error:
        state.update(status="STOP", error=str(error))
        json_out("preflight.json", state)
        raise


def nll(z: torch.Tensor, y: torch.Tensor) -> float:
    return float((torch.nn.functional.softplus(z) - y*z).mean())


def solve(z: torch.Tensor, y: torch.Tensor) -> dict:
    p = float(y.mean())
    assert 0 < p < 1, "STOP: no finite bias optimum for degenerate training rate"
    odds = math.log(p)-math.log1p(-p)
    lo, hi = odds-float(z.max()), odds-float(z.min())
    initial_lo, initial_hi = lo, hi
    for iteration in range(1, 201):
        bias = (lo+hi)/2
        predicted = float(torch.sigmoid(z+bias).mean())
        residual = predicted-p
        if abs(residual) <= 1e-12:
            break
        assert hi-lo > 1e-12, "STOP: bracket tolerance reached without rate convergence"
        if residual > 0:
            hi = bias
        else:
            lo = bias
    else:
        raise RuntimeError("STOP: scalar bisection did not converge")
    before, after = nll(z, y), nll(z+bias, y)
    assert after <= before+1e-12
    return dict(fitted_bias=bias, convergence_residual_mean_rate=residual, convergence_residual_total_events=residual*y.numel(),
        iterations=iteration, train_valid_bins=y.numel(), train_observed_rate=p,
        train_predicted_rate_before=float(torch.sigmoid(z).mean()), train_predicted_rate_after=predicted,
        train_nll_before=before, train_nll_after=after, bracket_lower=initial_lo, bracket_upper=initial_hi,
        solver_rate_tolerance=1e-12, solver_width_tolerance=1e-12, converged=True)


def fit() -> None:
    digest = protocol_hash()
    gate = json.loads((OUT / "preflight.json").read_text())
    assert gate["all_passed"] and gate["status"] == "PASSED"
    assert sha(OUT / "training_only_logits.pt") == gate["training_only_logits_sha256"]
    assert not (OUT / "bias_fit_lock.json").exists()
    torch.set_num_threads(2)
    data = torch.load(OUT / "training_only_logits.pt", weights_only=True)
    assert data["protocol_sha256"] == digest
    rows = []
    for cid, saved in data["cells"].items():
        y = saved["target"][saved["valid_mask"]].double()
        for condition, logits in saved["logits"].items():
            values = solve(logits[saved["valid_mask"]].double(), y)
            rows.append(dict(cell_id=cid, group=saved["group"], condition=condition, **values,
                training_logits_sha256=thash(logits), training_target_sha256=thash(saved["target"]),
                training_mask_sha256=thash(saved["valid_mask"]), protocol_sha256=digest))
    assert len(rows) == 88
    csv_out("bias_fit_per_cell.csv", rows)
    json_out("bias_fit_lock.json", dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest,
        bias_fit_per_cell_sha256=sha(OUT / "bias_fit_per_cell.csv"), fitted_scalar_count=88,
        calibration_inputs_only=["training_only_logits.pt"], heldout_targets_read_by_solver=False,
        before_any_recalibrated_test_NLL=True))
    print("88 TRAINING-ONLY SCALARS FITTED AND HASH-FROZEN", flush=True)


if __name__ == "__main__":
    {"freeze": freeze, "preflight": preflight, "fit": fit}[sys.argv[1]]()
