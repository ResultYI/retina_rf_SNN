from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

import torch

from data.retinal_recording import RealSequenceSplit
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import public_recordings
from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev
from experiments.retipath_multiscale_v0.circuit import area_integral_weights
from experiments.retipath_population_v0_1.circuit import BCOutput, PopulationRetipath
from experiments.retipath_population_v0_1.real_data import (
    history_events, load_recording_train, load_training_movie, physical_stimulus,
    recording_mapping,
)
from training.mechanistic_retina.center_surround_ln import DevelopmentStop, MIN_DELTA, PATIENCE
from training.mechanistic_retina.losses import expected_bernoulli_nll


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/real_data/retipath_population_r1"
PROTOCOL = ROOT / "docs/RETIPATH_REAL_R1_PROTOCOL.md"
R0 = ROOT / "output/real_data/retipath_population_r0_preflight"
MOVIE = ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"
RECORDINGS = ROOT / "data/real/schottdorf_lee_2021_repository/data"
CANONICAL = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
BASELINES = {
    "LN": ROOT / "output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830",
    "CNN": ROOT / ".omo/evidence/compact_causal_cnn_baseline",
    "Canonical": CANONICAL,
}
CONFIG = SchottdorfAdapterConfig()
LR, BATCH, MAX_STEPS = 0.03, 4, 1000
DTYPE = torch.float64
ATOL = 1e-10


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    a = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(a.dtype), tuple(a.shape))).encode())
    h.update(a.numpy().tobytes())
    return h.hexdigest()


def state_hash(state):
    return hashlib.sha256(json.dumps(
        {k: tensor_hash(v) for k, v in state.items()}, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_tensor_file(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)


def write_csv(path, rows):
    require(bool(rows), "empty CSV")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Cell:
    cell_id: str
    type: str
    recording_ids: tuple[str, ...]
    trials: int
    train_sequences: int
    train_bins: int
    validation_bins: int
    seed: int

    @property
    def slug(self):
        return self.cell_id.replace("#", "_")


def protocol_contract():
    text = PROTOCOL.read_text(encoding="utf-8")
    cells, hashes = [], {}
    for line in text.splitlines():
        parts = [x.strip().strip("`") for x in line.split("|")[1:-1]]
        if re.match(r"^\| [0-9]+#[0-9]+ \|", line):
            cid, typ, recs, *counts = parts
            cells.append(Cell(cid, typ, tuple(x.strip() for x in recs.split(";")),
                              *map(int, counts)))
        if len(parts) == 2 and re.fullmatch("[0-9a-f]{64}", parts[1]):
            hashes[parts[0]] = parts[1]
    require(len(hashes) == 16, "protocol source lock count differs")
    require(len(cells) == 9 and sum(c.type == "MC ON" for c in cells) == 5
            and sum(c.type == "MC OFF" for c in cells) == 4, "cohort contract differs")
    require(sum(len(c.recording_ids) for c in cells) == 16, "recording count differs")
    require(sum(c.train_bins for c in cells) == 107520 and
            sum(c.validation_bins for c in cells) == 26880, "bin counts differ")
    return tuple(cells), hashes


def verify_sources(hashes):
    actual = {name: sha256(ROOT / name) for name in hashes}
    require(actual == hashes, f"frozen source mismatch: {[k for k in hashes if actual[k] != hashes[k]]}")
    return actual


def optimizer_for(model):
    grouped = model.parameter_groups()
    params = [p for values in grouped.values() for p in values]
    require(len({id(p) for p in params}) == len(params), "duplicate optimizer parameter")
    require({id(p) for p in params} == {id(p) for p in model.parameters()}, "missing optimizer parameter")
    require(sum(p.numel() for p in params) == 359 and all(p.requires_grad for p in params),
            "trainable scalar contract differs")
    optimizer = torch.optim.Adam(params, lr=LR, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0, amsgrad=False)
    return optimizer, params


@dataclass(frozen=True)
class TrainingBundle:
    cell: Cell
    train: RealSequenceSplit
    centers: torch.Tensor
    mapping: object
    parser_metadata: tuple[dict, ...]


def concatenate_recording_splits(splits):
    offsets, offset = [], 0
    for split in splits:
        offsets.append(offset)
        offset += len(set(split.trial_indices))
    return RealSequenceSplit(
        *(torch.cat([getattr(s, key) for s in splits]) for key in
          ("cone_drive", "spike_counts", "spike_events", "valid_mask")),
        tuple(name for s in splits for name in s.source_image_ids),
        tuple(t + off for s, off in zip(splits, offsets, strict=True) for t in s.trial_indices),
    )


def load_cell_train_only(cell, movie, catalog):
    selected = [catalog[rid] for rid in cell.recording_ids]
    require(all(r.cell_id == cell.cell_id and f"{r.retinal_class} {r.polarity}" == cell.type
                and r.canonical_cell_type == "parasol" for r in selected), "cell mapping differs")
    splits, metadata = [], []
    for recording in selected:
        split, meta = load_recording_train(recording, movie, CONFIG)
        splits.append(split)
        metadata.append({"recording_id": recording.recording_id, **meta,
                         "raw_file_parser_scope": "whole original text file",
                         "binned_returned_window": [0, 2400],
                         "outer_validation_tensor_returned": False})
    return TrainingBundle(cell, concatenate_recording_splits(splits), movie.cone_positions_degs,
                          recording_mapping(selected[0]), tuple(metadata))


def select_rows(split, index):
    return split.cone_drive[index], split.spike_events[index], split.valid_mask[index]


def forward(model, values, target, bundle):
    stimulus = physical_stimulus(values, bundle.centers, CONFIG, dtype=DTYPE)
    events = history_events(target, bundle.mapping, dtype=DTYPE)
    trace = model(stimulus, observed_events=events)
    port = bundle.mapping.output_index
    return trace.outputs["logit"][..., port:port + 1]


def nll(logits, targets, mask):
    require(bool(mask.any()), "empty scored mask")
    return expected_bernoulli_nll(logits, targets.to(DTYPE), mask)


@torch.no_grad()
def score(model, split, bundle):
    model.eval()
    logits = forward(model, split.cone_drive, split.spike_events, bundle)
    value = float(nll(logits, split.spike_events, split.valid_mask))
    require(bool(torch.isfinite(torch.tensor(value))), "nonfinite NLL")
    return value


def split_evidence(split):
    return {"shape": list(split.cone_drive.shape), "target_shape": list(split.spike_events.shape),
            "scored_bins": int(split.valid_mask.sum()),
            "tensor_sha256": {k: tensor_hash(getattr(split, k)) for k in
                              ("cone_drive", "spike_counts", "spike_events", "valid_mask")},
            "source_image_ids": split.source_image_ids, "trial_indices": split.trial_indices}


def check_split(bundle):
    cell, s = bundle.cell, bundle.train
    require(s.cone_drive.shape == (cell.train_sequences, 150, 289), "input shape differs")
    require(s.spike_events.shape == s.valid_mask.shape == (cell.train_sequences, 150, 1), "target shape differs")
    expected_mask = (torch.arange(150) >= 30)[None, :, None].expand_as(s.valid_mask)
    require(torch.equal(s.valid_mask, expected_mask), "warmup mask differs")
    require(torch.equal(s.spike_events, (s.spike_counts > 0).to(s.spike_events)), "occupancy differs")
    require(int(s.valid_mask.sum()) == cell.train_bins and len(set(s.trial_indices)) == cell.trials,
            "train bin/trial count differs")
    require(all(int(re.search(r"live-frames-(\d+)-(\d+)", x)[2]) < 2400 for x in s.source_image_ids),
            "outer frame in trainer")
    inner = make_inner_dev(s)
    require(int(inner.train.valid_mask.sum()) == cell.trials * 1470
            and int(inner.development.valid_mask.sum()) == cell.trials * 390, "inner bin counts differ")
    require(all((b.trial_start, b.fit_stop, b.dev_start, b.trial_stop) == (0, 1860, 1920, 2400)
                for b in inner.boundaries), "inner boundary differs")
    for part, dev in ((inner.train, False), (inner.development, True)):
        for i, name in enumerate(part.source_image_ids):
            first = int(re.search(r"live-frames-(\d+)", name)[1])
            positions = first + torch.arange(150)
            support = positions >= 1860 if dev else positions < 1860
            expected = (torch.arange(150) >= 30) & (positions >= 1920 if dev else positions < 1860)
            require(torch.equal(part.valid_mask[i, :, 0], expected), "exact inner mask differs")
            require(bool((part.cone_drive[i, ~support] == 0).all()) and
                    bool((part.spike_events[i, ~support] == 0).all()), "inner input-support leak")
    return inner


@torch.no_grad()
def check_forward(model, bundle):
    values, target, _ = select_rows(bundle.train, slice(0, 2))
    stim = physical_stimulus(values, bundle.centers, CONFIG, dtype=DTYPE)
    events = history_events(target, bundle.mapping, dtype=DTYPE)
    initial = state_hash(model.state_dict())
    trace = model(stim, observed_events=events)
    require(all(bool(torch.isfinite(v).all()) for group in
                (trace.inputs, trace.states, trace.outputs, trace.initial_state) for v in group.values()),
            "nonfinite preflight trace")
    q_weights = area_integral_weights(stim.pixel_bounds, model.input_xy)
    require(torch.allclose(q_weights.sum(-1), torch.ones(25, dtype=DTYPE), rtol=0, atol=2e-6),
            "physical support missing")
    require(torch.equal(trace.inputs["q"], stim.values @ q_weights.T), "physical Q disagrees")
    require(abs(float(stim.pixel_bounds.max()) - .458203125) < ATOL, "FOV differs")
    cut = 75
    prefix_stim = replace(stim, values=stim.values[:, :cut + 1],
                          time_ms=stim.time_ms[:cut + 1], input_valid=stim.input_valid[:, :cut + 1])
    prefix = model(prefix_stim, observed_events=events[:, :cut + 1])
    changed = events.clone()
    changed[:, cut:, bundle.mapping.output_index] = 1 - changed[:, cut:, bundle.mapping.output_index]
    event_trace = model(stim, observed_events=changed)
    repeated = model(stim, observed_events=events)
    single = model(replace(stim, values=stim.values[:1], input_valid=stim.input_valid[:1]),
                   observed_events=events[:1])
    errors = {
        "future_input_prefix": float((trace.outputs["logit"][:, :cut + 1] - prefix.outputs["logit"]).abs().max()),
        "current_future_event_prefix": float((trace.outputs["logit"][:, :cut + 1] - event_trace.outputs["logit"][:, :cut + 1]).abs().max()),
        "reset_replay": float((trace.outputs["logit"] - repeated.outputs["logit"]).abs().max()),
        "batch_state_isolation": float((trace.outputs["logit"][:1] - single.outputs["logit"]).abs().max()),
    }
    require(max(errors.values()) <= ATOL, f"causality/reset failure: {errors}")
    require(bool((trace.states["history"][:, 0] == 0).all()), "history initial state differs")
    require(all(torch.equal(v, torch.full_like(v, 2 / 9 if name == "V" else 0))
                for name, v in trace.initial_state.items()), "baseline reset differs")
    require(float((trace.outputs["logit"][:, cut + 1:] - event_trace.outputs["logit"][:, cut + 1:]).abs().max()) > 0,
            "strict past history is disconnected")
    require(state_hash(model.state_dict()) == initial, "preflight changed parameters")
    return {"passed": True, "absolute_errors": errors, "atol": ATOL,
            "tolerance_source": "R0 numerical correctness tolerance, not prediction success threshold",
            "event_change_purpose": "current/future event leakage assertion, no fitted stimulus experiment",
            "Q_weights_sha256": tensor_hash(q_weights), "physical_coverage": "FULL_TEMPLATE_ANCESTOR_SUPPORT",
            "recorded_port": bundle.mapping.output_index, "state_dict_unchanged": True}


def parameter_audit(model, optimizer, initial):
    updated = {name: int((p.detach() != initial[name]).sum()) for name, p in model.named_parameters()}
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "optimizer_listed": sum(p.numel() for g in optimizer.param_groups for p in g["params"]),
            "actually_updated": sum(updated.values()), "updated_scalars_by_tensor": updated,
            "optimizer_steps_by_tensor": {n: int(optimizer.state[p].get("step", 0))
                                           for n, p in model.named_parameters()},
            "count_unit": "raw_scalar; optimizer_steps are per tensor"}


def fit_phase(model, split, bundle, *, development, steps, path):
    torch.manual_seed(bundle.cell.seed)
    generator = torch.Generator().manual_seed(bundle.cell.seed + 1000003)
    optimizer, params = optimizer_for(model)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_hash = state_hash(initial)
    first_dev = score(model, development, bundle) if development is not None else None
    status = DevelopmentStop(first_dev, 0, first_dev, 0) if development is not None else None
    fields = ["step", "sampled_indices", "data_nll", "hierarchy_penalty", "objective",
              "inner_dev_nll", "best_step", "stale_steps", "absent_gradient_tensors"]
    stop_step = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(step=0, inner_dev_nll=first_dev, hierarchy_penalty=float(model.hierarchy_penalty().detach()),
                             best_step=0 if status else None))
        stream.flush()
        for step in range(1, steps + 1):
            model.train()
            index = torch.randint(split.cone_drive.shape[0], (BATCH,), generator=generator)
            x, y, mask = select_rows(split, index)
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model, x, y, bundle)
            data_loss = nll(logits, y, mask)
            penalty = model.hierarchy_penalty()
            objective = data_loss + penalty
            require(bool(torch.isfinite(objective)), "nonfinite objective before optimizer")
            objective.backward()
            missing = [n for n, p in model.named_parameters() if p.grad is None]
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params),
                    f"nonfinite gradient at {bundle.cell.cell_id} step {step}")
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in params), "nonfinite parameter after optimizer")
            stop_step = step
            dev_nll = score(model, development, bundle) if development is not None else None
            if status is not None:
                status = status.observe(dev_nll, step)
            writer.writerow(dict(step=step, sampled_indices=json.dumps(index.tolist()),
                                 data_nll=float(data_loss.detach()), hierarchy_penalty=float(penalty.detach()),
                                 objective=float(objective.detach()), inner_dev_nll=dev_nll,
                                 best_step=status.best_step if status else None,
                                 stale_steps=status.stale_steps if status else None,
                                 absent_gradient_tensors=json.dumps(missing)))
            stream.flush()
            if step % 50 == 0 or (status is not None and status.stopped) or step == steps:
                print(f"{bundle.cell.cell_id} {path.stem} step={step} K={status.best_step if status else steps} "
                      f"data={float(data_loss.detach()):.9f} P={float(penalty.detach()):.9f} dev={dev_nll}", flush=True)
            if status is not None and status.stopped:
                break
    return {"initial_state_sha256": initial_hash, "end_state_sha256": state_hash(model.state_dict()),
            "best_step": status.best_step if status else stop_step, "stop_step": stop_step,
            "best_dev_nll": status.best_nll if status else None,
            "final_full_split_data_nll": score(model, split, bundle),
            "final_hierarchy_penalty": float(model.hierarchy_penalty().detach()),
            "parameter_audit": parameter_audit(model, optimizer, initial), "gradients_finite": True}


def manifest(status):
    files = {p.relative_to(OUT).as_posix(): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file()
             and p.name not in {"manifest.json", "training-manifest.json"}}
    write_json(OUT / ("training-manifest.json" if status == "CHECKPOINTS_LOCKED" else "manifest.json"),
               {"status": status, "created_utc": utc(), "files_sha256": files})


def train():
    require(not OUT.exists(), "R1 output already exists; refusing overwrite or implicit rerun")
    cells, frozen = protocol_contract()
    verify_sources(frozen)
    require((PATIENCE, MIN_DELTA) == (200, 1e-7), "stop contract differs")
    require(asdict(CONFIG) == dict(train_sequence_count=16, validation_sequence_count=4,
                                 sequence_steps=150, warmup_steps=30, crop_pixels=51, pool_factor=3),
            "adapter contract differs")
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    (OUT / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    extra = ["experiments/retipath_population_v0_1/real_r1.py",
             "experiments/retipath_population_v0_1/evaluate_real_r1.py",
             "models/mechanistic_retina/state.py", "models/mechanistic_retina/retipath_spatial_ei.py",
             "baselines/center_surround_ln.py", "data/retinal_recording.py"]
    source_hashes = frozen | {p: sha256(ROOT / p) for p in extra}
    for name in source_hashes:
        dest = OUT / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    write_json(OUT / "source_lock.json", {"created_utc": utc(), "source_sha256": source_hashes,
               "protocol_sha256": sha256(PROTOCOL), "protocol_frozen_hashes_match": True})
    write_json(OUT / "protocol.json", {"schema": "population_real_r1_v1", "created_utc": utc(),
               "cohort": [asdict(c) for c in cells], "adapter": asdict(CONFIG), "optimizer": "Adam",
               "lr": LR, "batch": BATCH, "max_inner_updates": MAX_STEPS, "patience": PATIENCE,
               "min_delta": MIN_DELTA, "clip": None, "dtype": "float64", "device": "cpu", "threads": 2,
               "objective": "recorded_RGC_Bernoulli_NLL + current_hierarchy_penalty",
               "BC_output": "LegacyPReLU", "auxiliary_loss": False, "independent_real_test": "NONE",
               "outer_validation_policy": "separate evaluator process after all 9 final checkpoints locked"})
    catalog = {r.recording_id: r for r in public_recordings(RECORDINGS)}
    old = json.loads((CANONICAL / "run-manifest.json").read_text(encoding="utf-8"))["source_sha256"]
    old = {Path(p).resolve(): digest for p, digest in old.items()}
    data_paths = [MOVIE, *[catalog[r].path for c in cells for r in c.recording_ids]]
    data_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in data_paths}
    require(all(data_hashes[p.relative_to(ROOT).as_posix()] == old[p.resolve()] for p in data_paths),
            "legacy raw asset hash differs")
    write_json(OUT / "data_lock.json", {"created_utc": utc(), "data_sha256": data_hashes,
               "legacy_asset_hashes_match": True, "hash_access_is_not_target_materialization": True})
    r0 = {r["cell_id"]: r for r in csv.DictReader((R0 / "per_cell_preflight.csv").open(encoding="utf-8-sig"))}
    require(all(r0[c.cell_id]["recordings"].split(";") == list(c.recording_ids) and
                r0[c.cell_id]["type"] == c.type and r0[c.cell_id]["forward_finite"] == "True" for c in cells),
            "R0 cohort mismatch")
    print("source/raw identity locks passed; loading train-only prefix", flush=True)
    movie = load_training_movie(MOVIE, CONFIG)
    require(movie.sequences.shape == (16, 150, 289), "train movie exposes outer frames")
    bundles = [load_cell_train_only(c, movie, catalog) for c in cells]
    models = [PopulationRetipath(bc_output=BCOutput.LEGACY_PRELU, dtype=DTYPE) for c in cells]
    pointers = [p.data_ptr() for model in models for p in model.parameters()]
    require(len(pointers) == len(set(pointers)), "cross-cell trainable storage alias")
    checks = []
    for bundle, model in zip(bundles, models, strict=True):
        inner = check_split(bundle)
        optimizer, params = optimizer_for(model)
        require(all(not optimizer.state[p] for p in params), "optimizer has prior state")
        check = check_forward(model, bundle)
        x, y, mask = select_rows(inner.train, slice(0, BATCH))
        model.regularized_objective(nll(forward(model, x, y, bundle), y, mask)).backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params), "preflight gradient nonfinite")
        optimizer.zero_grad(set_to_none=True)
        evidence = {"cell": asdict(bundle.cell), "full_train": split_evidence(bundle.train),
                    "inner_train": split_evidence(inner.train), "inner_dev": split_evidence(inner.development),
                    "boundaries": [asdict(b) for b in inner.boundaries], "parser": bundle.parser_metadata}
        cell_dir = OUT / "cells" / bundle.cell.slug
        cell_dir.mkdir(parents=True, exist_ok=False)
        write_json(cell_dir / "training_data_contract.json", evidence)
        checks.append({"cell_id": bundle.cell.cell_id, "trainable": 359, "optimizer_listed": 359,
                       "optimizer_steps": 0, "forward": check, "split_masks_pass": True})
    del models, model, optimizer, params, inner
    write_json(OUT / "preflight.json", {"passed": True, "completed_utc": utc(), "cells": checks,
               "cross_cell_storage_unique": True, "outer_validation_materialized": False,
               "whole_file_parser_scope_disclosed": True, "scientific_config_changed": False})
    print("all 9 cells passed preflight; starting frozen inner/refit fits", flush=True)
    completed = []
    for bundle in bundles:
        c = bundle.cell
        cell_dir = OUT / "cells" / c.slug
        write_json(cell_dir / "STARTED.json", {"started_utc": utc(), "cell_id": c.cell_id})
        inner = make_inner_dev(bundle.train)
        model = PopulationRetipath(bc_output=BCOutput.LEGACY_PRELU, dtype=DTYPE)
        fit = fit_phase(model, inner.train, bundle, development=inner.development,
                        steps=MAX_STEPS, path=cell_dir / "inner-trajectory.csv")
        write_json(cell_dir / "inner-summary.json", fit)
        del model
        model = PopulationRetipath(bc_output=BCOutput.LEGACY_PRELU, dtype=DTYPE)
        require(state_hash(model.state_dict()) == fit["initial_state_sha256"], "refit initial state differs")
        refit = fit_phase(model, bundle.train, bundle, development=None,
                         steps=fit["best_step"], path=cell_dir / "refit-trajectory.csv")
        require(refit["stop_step"] == fit["best_step"], "refit step count differs")
        p_constant = float(bundle.train.spike_events[bundle.train.valid_mask].mean().clamp(1e-6, 1-1e-6))
        ckpt = {"schema": "population_real_r1_final_v1", "cell": asdict(c), "model": model.state_dict(),
                "K": fit["best_step"], "inner_stop_step": fit["stop_step"],
                "initial_state_sha256": fit["initial_state_sha256"], "final_state_sha256": refit["end_state_sha256"],
                "protocol_sha256": sha256(PROTOCOL), "constant_probability": p_constant,
                "bc_output": "LegacyPReLU", "source_lock_sha256": sha256(OUT / "source_lock.json")}
        save_tensor_file(cell_dir / "final.pt", ckpt)
        summary = {"cell_id": c.cell_id, "type": c.type, "seed": c.seed, "inner": fit, "refit": refit,
                   "constant_probability": p_constant, "final_checkpoint_sha256": sha256(cell_dir / "final.pt"),
                   "completed_utc": utc(), "status": "COMPLETE"}
        write_json(cell_dir / "training-summary.json", summary)
        completed.append(summary)
        print(f"COMPLETE {c.cell_id}: K={fit['best_step']} stop={fit['stop_step']}", flush=True)
    verify_sources(source_hashes)
    lock = {"schema": "population_real_r1_checkpoint_lock_v1", "locked_utc": utc(),
            "cohort_complete": len(completed) == 9, "cell_ids": [c.cell_id for c in cells],
            "recording_ids": [r for c in cells for r in c.recording_ids],
            "protocol_sha256": sha256(PROTOCOL), "source_lock_sha256": sha256(OUT / "source_lock.json"),
            "preflight_sha256": sha256(OUT / "preflight.json"),
            "checkpoints": {f"cells/{c.slug}/final.pt": sha256(OUT / "cells" / c.slug / "final.pt") for c in cells},
            "training_summaries": {f"cells/{c.slug}/training-summary.json": sha256(OUT / "cells" / c.slug / "training-summary.json") for c in cells},
            "outer_validation_materialized_before_lock": False}
    require(lock["cohort_complete"] and len(lock["checkpoints"]) == 9, "incomplete checkpoint lock")
    write_json(OUT / "CHECKPOINT_LOCK.json", lock)
    write_json(OUT / "training_summary.json", {"cells": completed, "status": "CHECKPOINTS_LOCKED"})
    manifest("CHECKPOINTS_LOCKED")
    print("CHECKPOINT_LOCK complete. Trainer exits without outer-validation access.", flush=True)


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        if OUT.exists() and not (OUT / "failure.json").exists():
            write_json(OUT / "failure.json", {"phase": "training_or_preflight", "utc": utc(),
                       "error": str(exc), "traceback": traceback.format_exc(), "automatic_retry": False})
        traceback.print_exc()
        sys.exit(1)
