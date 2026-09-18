from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "work")]
import numpy as np
import torch
from work.retipath_first_nonlinearity_locus.model import build, observe
from retipath_phase2_common import load_train, load_development, evaluate, minibatches
from retipath_final_common import identity
from retipath_spatial_ei_pilot import sha, tensor_sha
from training.mechanistic_retina.losses import expected_bernoulli_nll
import retipath_f2_localization_pilot as f2

CELLS = ("67#6", "67#7", "68#3", "68#10")
SEED = 2026091301
UPDATES = 200
OUT = ROOT / "output/experiments/retipath_first_nonlinearity_locus_fast_screen_20260916"
MANIFEST = ROOT / "output/evaluations/retipath_gain_canonicalization_20260915/migration_manifest.json"
PHASE2 = ROOT / "output/experiments/retipath_spatial_ei_phase2_population/checkpoints/protocol.json"
F2_DIR = ROOT / "output/evaluations/retipath_f2_localization_pilot_20260916"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=lambda v: v.item())
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_sha(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        digest.update(tensor_sha(value).encode())
    return digest.hexdigest()


def preflight(cp, train, expected_parameters):
    m0, p0 = build(cp, "M0")
    m1, p1 = build(cp, "M1")
    trainable = {name: p.numel() for name, p in m0.named_parameters() if p.requires_grad}
    assert trainable == expected_parameters
    assert trainable == {name: p.numel() for name, p in m1.named_parameters() if p.requires_grad}
    assert state_sha(state(m0)) == state_sha(state(m1)) == state_sha(cp["model"])
    assert set(name for name in trainable if name == "theta") == {"theta"}
    x, y = train.cone_drive[:2], train.spike_events[:2]
    evidence = {"trainable_scalar_count_M0": sum(p.numel() for p in p0),
                "trainable_scalar_count_M1": sum(p.numel() for p in p1), "trainable_names": trainable,
                "initial_state_dict_identical": True, "initial_alpha": float(m1.alpha.detach())}
    with torch.no_grad():
        for condition, model in (("M0", m0), ("M1", m1)):
            raw = model(x, observed_counts=y)
            traced = observe(model, x, y)
            torch.testing.assert_close(raw.logits, traced["logit"], rtol=0, atol=0)
            torch.testing.assert_close(raw.spike_probability, traced["probability"], rtol=0, atol=0)
            evidence[condition + "_trace_forward_bitwise"] = True
        parts = m1.spatial_components(x)
        features = m1.feature_bank(parts.h1.modulated_cones, mixer=m1.shared_subunits)
        weighted = features * m1.bipolar.positive_weights().repeat(1, 2, 1, 1)[None, None]
        expected = weighted.sum(-1).transpose(-1, -2)
        torch.testing.assert_close(parts.direct, expected[..., :2], rtol=0, atol=0)
        torch.testing.assert_close(parts.broad, expected[..., 2:], rtol=0, atol=0)
        actual_early = m1.effective_input(x)
        torch.testing.assert_close(parts.h1.modulated_cones, actual_early - parts.h1.surround, rtol=0, atol=0)
        theta0 = m0.theta.clone()
        theta1 = m1.theta.clone()
        m0.theta.zero_(); m1.theta.zero_()
        torch.testing.assert_close(m0(x, observed_counts=y).logits, m1(x, observed_counts=y).logits, rtol=0, atol=0)
        m0.theta.copy_(theta0); m1.theta.copy_(theta1)
    evidence["M1_old_BC_location_identity"] = True
    evidence["alpha_one_M0_M1_bitwise_equal"] = True
    for condition, model, params in (("M0", m0, p0), ("M1", m1, p1)):
        expected_bernoulli_nll(model(x, observed_counts=y).logits, y, train.valid_mask[:2]).backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
        assert float(model.theta.grad.abs()) > 0
        evidence[condition + "_theta_gradient"] = float(model.theta.grad)
    phi = m1.effective_input(torch.tensor([1.0, -1.0]))
    d_pos = torch.autograd.grad(phi[0], m1.theta, retain_graph=True)[0]
    d_neg = torch.autograd.grad(phi[1], m1.theta)[0]
    assert float(d_pos) == 0 and torch.equal(d_neg, -m1.alpha.detach())
    evidence["no_new_alpha_scale_gauge"] = "phi_alpha(1)=1 fixes multiplicative scale; phi_alpha(-1)=-alpha fixes the negative/positive ratio. No extra strength or second slope was introduced; this does not claim global parameter identifiability."
    return evidence


def save_checkpoint(path, cp, condition, step, model_state, train_nll, dev_nll, label):
    result = {key: value for key, value in cp.items() if key != "model"}
    result.update(model=model_state, experiment="first_nonlinearity_locus_fast_screen", condition=condition,
                  phase="matched_continuation", training_updates=step, seed=SEED, step=step,
                  selection_label=label, train_nll=train_nll, dev_nll=dev_nll,
                  model_class="CanonicalGainRetiPath" if condition == "M0" else "ConeRelatedEffectiveInputNonlinearityCandidate",
                  entry_point=("models.mechanistic_retina.retipath_canonical_gain.CanonicalGainRetiPath" if condition == "M0" else
                               "work.retipath_first_nonlinearity_locus.model.ConeRelatedEffectiveInputNonlinearityCandidate"),
                  formal_registry_updated=False, alpha_bounds=[0.05, 2.0])
    with path.open("xb") as handle:
        torch.save(result, handle)


def fit(cp, condition, train, dev, schedule, source_hash):
    started = time.monotonic()
    model, params = build(cp, condition)
    fixed = {name: value.clone() for name, value in model.named_buffers()}
    fixed.update({name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad})
    optimizer = torch.optim.Adam(params, lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in params}
    directory = OUT / "checkpoints" / cp["cell_id"].replace("#", "_") / condition
    directory.mkdir(parents=True)
    curves, best_state, best = [], None, None
    max_norm, clipped, bound_hits = 0.0, 0, 0
    grad_seen = {name: False for name, p in model.named_parameters() if p.requires_grad}
    for step in range(UPDATES + 1):
        if step:
            model.train()
            ids = schedule[step - 1]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train.cone_drive[ids], observed_counts=train.spike_events[ids]).logits
            loss = expected_bernoulli_nll(logits, train.spike_events[ids], train.valid_mask[ids])
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
                    grad_seen[name] |= bool((parameter.grad != 0).any())
            norm = float(torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True))
            max_norm = max(max_norm, norm); clipped += int(norm > 5)
            optimizer.step()
            model.project_mechanism_parameters()
            assert all(bool(torch.isfinite(p).all()) for p in params)
            bound_hits += int(abs(float(model.theta.detach()) - np.log(0.05)) < 1e-7 or abs(float(model.theta.detach()) - np.log(2.0)) < 1e-7)
        if step % 20:
            continue
        train_nll, _ = evaluate(model, train)
        dev_nll, _ = evaluate(model, dev)
        assert np.isfinite(train_nll) and np.isfinite(dev_nll)
        current = model.state_dict()
        assert all(torch.equal(value, current[name]) for name, value in fixed.items())
        row = {"step": step, "train_nll": train_nll, "dev_nll": dev_nll,
               "alpha": float(model.alpha.detach()), "theta": float(model.theta.detach())}
        curves.append(row)
        if step == 0:
            save_checkpoint(directory / "initial.pt", cp, condition, step, state(model), train_nll, dev_nll, "initial")
        if best is None or dev_nll < best["dev_nll"]:
            best, best_state = row.copy(), state(model)
    assert all(int(optimizer.state[p]["step"]) == UPDATES for p in params)
    last = curves[-1]
    save_checkpoint(directory / "final.pt", cp, condition, UPDATES, state(model), last["train_nll"], last["dev_nll"], "fixed_step200")
    save_checkpoint(directory / "best.pt", cp, condition, best["step"], best_state, best["train_nll"], best["dev_nll"], "minimum_development_among_0_20_to_200")
    replay_cp = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    replay, _ = build(replay_cp, condition)
    replay_nll, _ = evaluate(replay, dev)
    assert replay_nll == best["dev_nll"]
    result = {"cell": cp["cell_id"], "condition": condition, "seed": SEED, "curves": curves,
              "selected": best, "initial": curves[0], "final": last,
              "best_train_nll": min(row["train_nll"] for row in curves),
              "best_train_step": min(curves, key=lambda row: row["train_nll"])["step"],
              "optimizer_updates": UPDATES, "all_expected_parameters_in_optimizer": True,
              "all_gradients_finite": True, "nonzero_gradient_seen": grad_seen,
              "frozen_buffers_and_parameters_unchanged": True, "bounds_hit_updates": bound_hits,
              "maximum_gradient_norm": max_norm, "clipped_updates": clipped,
              "selected_checkpoint_strict_replay_NLL_exact": True,
              "source_checkpoint_sha256": source_hash, "schedule_sha256": tensor_sha(schedule),
              "checkpoint_files": {label: {"path": str((directory / (label + ".pt")).relative_to(ROOT)),
                                            "sha256": sha(directory / (label + ".pt"))} for label in ("initial", "final", "best")},
              "elapsed_seconds": time.monotonic() - started}
    save_json(directory / "training_record.json", result)
    print(f"{cp['cell_id']} {condition}: 200/200; dev {curves[0]['dev_nll']:.6f} -> {last['dev_nll']:.6f}; selected {best['step']}; alpha {curves[0]['alpha']:.4f} -> {last['alpha']:.4f}", flush=True)
    return result


def check_f2(fits, selection_hash, stimulus_hash):
    assert sha(OUT / "checkpoints/selection_lock.json") == selection_hash
    assert sha(F2_DIR / "stimuli.npz") == stimulus_hash
    with np.load(F2_DIR / "stimuli.npz", allow_pickle=False) as saved:
        stimulus, time_s = saved["stimulus_weber"], saved["time_s"]
        cases = json.loads(str(saved["cases_json"]))
    assert stimulus.shape == (7, 600, 289) and float(stimulus.min()) >= -1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, checks = [], []
    for fit_record in fits:
        ref = fit_record["checkpoint_files"]["best"]
        assert sha(ROOT / ref["path"]) == ref["sha256"]
        cp = torch.load(ROOT / ref["path"], map_location="cpu", weights_only=True)
        model, _ = build(cp, cp["condition"], dtype=torch.float64, device="cuda")
        model.eval().requires_grad_(False)
        before = state(model)
        with torch.no_grad():
            x = torch.tensor(stimulus, device="cuda", dtype=torch.float64)
            y = torch.zeros((7, 600, 1), device="cuda", dtype=torch.float64)
            traced = observe(model, x, y)
            direct = model(x, observed_counts=y)
            logit_error = float((traced["logit"] - direct.logits).abs().max())
            probability_error = float((traced["probability"] - direct.spike_probability).abs().max())
            logit_tolerance = 64 * torch.finfo(torch.float64).eps * max(1.0, float(direct.logits.abs().max()))
            torch.testing.assert_close(traced["logit"], direct.logits, rtol=0, atol=logit_tolerance)
            torch.testing.assert_close(traced["probability"], direct.spike_probability,
                                       rtol=0, atol=64 * torch.finfo(torch.float64).eps)
            assert all(bool(torch.isfinite(value).all()) for value in traced.values())
            assert torch.count_nonzero(traced["history_state"]) == 0
        assert state_sha(before) == state_sha(state(model))
        for field in f2.FIELDS:
            value = traced[field].cpu().numpy()
            coefficient = f2.project(value, time_s)
            floor = f2.FLOOR_MULTIPLIER * np.finfo(np.float64).eps * np.max(abs(value[:, f2.MEASURE]), axis=1)
            for group, c in f2.groups(field, coefficient):
                floors = dict(f2.groups(field, floor))[group]
                denominator = f2.rms(c[-1, 0])
                denominator_floor = f2.rms(floors[-1])
                for index, case in enumerate(cases):
                    amplitude_1, amplitude_2 = f2.rms(c[index, 0]), f2.rms(c[index, 1])
                    noise_floor = f2.rms(floors[index])
                    valid = denominator > denominator_floor
                    rows.append({"cell": cp["cell_id"], "seed": SEED, "condition": cp["condition"],
                                 "selected_step": cp["step"], "selected_alpha": float(model.alpha), **case,
                                 "trace": field, "group": group, "F1": amplitude_1, "F2": amplitude_2,
                                 "uniform_F1": denominator, "F2_over_uniform_F1": amplitude_2 / denominator if valid else None,
                                 "roundoff_reference": noise_floor, "uniform_roundoff_reference": denominator_floor,
                                 "F2_above_roundoff": amplitude_2 > noise_floor,
                                 "normalization_status": "VALID" if valid else "UNRESOLVED_DENOMINATOR",
                                 "selected_checkpoint_sha256": ref["sha256"]})
        checks.append({"cell": cp["cell_id"], "condition": cp["condition"],
                       "trace_forward_bitwise": logit_error == 0 and probability_error == 0,
                       "trace_forward_max_abs_logit_error": logit_error,
                       "trace_forward_max_abs_probability_error": probability_error,
                       "trace_forward_logit_tolerance": logit_tolerance,
                       "trace_forward_within_float64_roundoff": True,
                       "inference_state_unchanged": True, "all_traces_finite": True, "zero_history": True})
        print(f"F2 {cp['cell_id']} {cp['condition']}: frozen selected step {cp['step']}", flush=True)
    save_csv(OUT / "f2_comparison.csv", rows)
    assert sha(OUT / "checkpoints/selection_lock.json") == selection_hash
    return rows, checks


def run():
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError("Refusing automatic repeat or overwrite")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    phase2, manifest, f2_protocol = read_json(PHASE2), read_json(MANIFEST), read_json(F2_DIR / "protocol.json")
    inputs, evidence, references = {}, {}, {}
    for cell in CELLS:
        entry = next(e for e in manifest["checkpoints"] if e["cell"] == cell and e["seed"] == SEED)
        ref = phase2["cells"][cell]
        assert sha(ROOT / entry["path"]) == entry["sha256"] and sha(Path(ref["input_path"])) == ref["input_sha256"]
        cp = torch.load(ROOT / entry["path"], map_location="cpu", weights_only=True)
        assert cp["gain_schema"] == "retipath_canonical_gain_v1" and cp["training_updates"] == 0
        assert cp["cell_id"] == cell and cp["seed"] == SEED
        train, dev = load_train(ref), load_development(ref)
        schedule = minibatches(len(train.cone_drive), SEED, steps=UPDATES)
        evidence[cell] = preflight(cp, train, entry["new_trainable"])
        references[cell] = {"checkpoint": {key: entry[key] for key in ("path", "sha256")},
                            "input_path": ref["input_path"], "input_sha256": ref["input_sha256"],
                            "train_identity": identity(train), "development_identity": identity(dev),
                            "train_sequences": len(train.cone_drive), "development_sequences": len(dev.cone_drive),
                            "schedule_sha256": tensor_sha(schedule), "schedule": schedule.tolist()}
        inputs[cell] = (cp, train, dev, schedule)
        print(f"Preflight {cell}: 33/33 parameters, alpha-only relocation verified", flush=True)
    source_paths = [Path(__file__), Path(__file__).with_name("model.py"), MANIFEST, PHASE2,
                    ROOT / "work/retipath_f2_localization_pilot.py", ROOT / "work/retipath_phase2_common.py",
                    ROOT / "training/mechanistic_retina/losses.py", ROOT / "evaluation/mechanistic_retina/mechanism_observation.py"]
    source_paths += sorted((ROOT / "models/mechanistic_retina").glob("*.py"))
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    protocol = {
        "task": "FAST SCREEN first-nonlinearity locus comparison", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "locked_before_training": True, "cells": list(CELLS), "seed": SEED,
        "M0": "current canonical BC-first RetiPath; matched continuation",
        "M1": "cone-related effective input nonlinearity candidate; existing theta applied pointwise to raw stimulus before H1; old BC transform is identity",
        "phi": "phi_alpha(x)=x for x>=0, alpha*x for x<0; alpha=exp(theta)",
        "alpha_bounds": [0.05, 2.0], "new_trainable_scalars": 0, "trainable_scalar_count": {"M0": 33, "M1": 33},
        "initialization": "strict same canonical state_dict including theta, all shared parameters, fixed RMS and geometry; fresh Adam state in both conditions",
        "training": {"optimizer": "Adam", "lr": 0.003, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
                     "batch_size": 4, "updates": UPDATES, "gradient_clip_norm": 5, "evaluation_every": 20,
                     "scheduler": None, "early_stopping": False, "restarts": 0, "device": "CPU", "dtype": "float32", "threads": 1},
        "data_contract": "cached train [0,16), development [16,20); 17x17 L+M Weber,150Hz Bernoulli occupancy;150-bin independent sequences,30 warmup,120 scored; supplied strictly-past observed history and identical masks",
        "selection": "per cell and condition, minimum dev NLL among 0,20,...,200; earliest exact tie; save initial/final/best; freeze all eight best checkpoints before any F2 inference",
        "primary_prediction": "M1 selected-development NLL minus M0 selected-development NLL; equal-cell mean; descriptive FAST SCREEN, no independent test",
        "fixed_budget_prediction": "also report fixed step200 delta and all training/development trajectories; selection and final are never substituted after seeing results",
        "F2_contract_path": str((F2_DIR / "protocol.json").relative_to(ROOT)), "F2_contract_sha256": sha(F2_DIR / "protocol.json"),
        "F2_stimuli_path": str((F2_DIR / "stimuli.npz").relative_to(ROOT)), "F2_stimuli_sha256": sha(F2_DIR / "stimuli.npz"),
        "F2_definition": {key: f2_protocol[key] for key in ("Fourier", "primary_reduction", "groups", "roundoff_reference", "normalization_guard")},
        "F2_condition": "saved seven physically valid sequences; 4 Hz, max Weber amplitude0.5, widths1/2/4, phases0/0.5, matched full-field F1;8+8cycles; FIX_HISTORY_ZERO; float64 CUDA inference copy",
        "mechanism_gate": "all four selected M1 H1 graph/state/feedback traces exceed frozen roundoff floor for all six gratings with valid uniform denominator; corresponding M0 H1 traces stay at floor",
        "verdict_rules": {"NLL_numerical_tie": 1e-6, "mean_material_change_reference": 1e-4,
                          "source_for_numeric_conventions": "existing work/retipath_geometry_fast_screen/run.py FAST SCREEN conventions; not biological equivalence margins",
                          "systematic_prediction_harm": "mean delta>1e-4 and >=3/4 losses at selected or fixed step200",
                          "PROMISING_EARLY_NONLINEARITY": "mechanism gate passes, selected mean delta<=1e-4, <=2/4 selected losses, and no systematic harm at step200",
                          "REJECT_EARLY_NONLINEARITY": "systematic prediction harm or numerical degeneration; also if all selected M1 H1 F2 remains at floor",
                          "MIXED": "all other cases; no budget or parameter rescue"},
        "alpha_reporting": "initial/selected/final alpha, theta, abs(alpha-1), full alpha trajectory and bound hits; no result-chosen identity cutoff",
        "scale_gauge_reason": "fixed positive slope=1 prevents alpha being an overall multiplicative scale; normalized pathway composition and canonical G_E/G_I retained; no second alpha",
        "sources_sha256": hashes, "references": references, "preflight_correctness": evidence,
        "prohibited": ["population long training", "additional seeds/updates/restarts", "F2-based selection", "new test intervals", "DoG/radius fitting", "new supports/states", "pathway clamps", "formal model default/registry changes"],
    }
    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol)
    fits = []
    for cell in CELLS:
        cp, train, dev, schedule = inputs[cell]
        for condition in ("M0", "M1"):
            fits.append(fit(cp, condition, train, dev, schedule, references[cell]["checkpoint"]["sha256"]))
    selected = {"locked_utc": datetime.now(timezone.utc).isoformat(), "F2_inference_started": False,
                "criterion": protocol["selection"], "models": [{"cell": fit["cell"], "condition": fit["condition"],
                "selected_step": fit["selected"]["step"], "selected_dev_nll": fit["selected"]["dev_nll"],
                **fit["checkpoint_files"]["best"]} for fit in fits]}
    save_json(OUT / "checkpoints/selection_lock.json", selected)
    selection_hash = sha(OUT / "checkpoints/selection_lock.json")
    indexed = {(record["cell"], record["condition"]): record for record in fits}
    predictions = []
    for cell in CELLS:
        row = {"cell": cell, "seed": SEED, "polarity": inputs[cell][0]["polarities"][0]}
        for condition in ("M0", "M1"):
            fit_record = indexed[cell, condition]
            row["NLL_" + condition] = fit_record["selected"]["dev_nll"]
            for label in ("initial", "selected", "final"):
                for name, value in fit_record[label].items():
                    row[f"{condition}_{label}_{name}"] = value
            row[condition + "_best_train_nll"] = fit_record["best_train_nll"]
        row["Delta_M1_minus_M0"] = row["NLL_M1"] - row["NLL_M0"]
        row["Delta_step200"] = row["M1_final_dev_nll"] - row["M0_final_dev_nll"]
        predictions.append(row)
    save_csv(OUT / "per_cell_prediction.csv", predictions)
    finish_analysis(fits, predictions, selection_hash, protocol, started)


def finish_analysis(fits, predictions, selection_hash, protocol, started):
    f2_rows, f2_checks = check_f2(fits, selection_hash, protocol["F2_stimuli_sha256"])
    comparisons = {}
    for endpoint, field in (("selected", "Delta_M1_minus_M0"), ("step200", "Delta_step200")):
        delta = np.array([row[field] for row in predictions])
        comparisons[endpoint] = {"mean": float(delta.mean()), "median": float(np.median(delta)),
                                 "wins": int(np.sum(delta < -1e-6)), "losses": int(np.sum(delta > 1e-6)),
                                 "ties": int(np.sum(abs(delta) <= 1e-6)), "per_cell_delta": dict(zip(CELLS, delta.tolist())),
                                 "mean_NLL_M0": float(np.mean([row["NLL_M0"] if endpoint == "selected" else row["M0_final_dev_nll"] for row in predictions])),
                                 "mean_NLL_M1": float(np.mean([row["NLL_M1"] if endpoint == "selected" else row["M1_final_dev_nll"] for row in predictions]))}
    h1 = [r for r in f2_rows if r["kind"] == "grating" and r["trace"] in ("h1_graph_drive", "h1_state", "h1_feedback")]
    gate = all(r["normalization_status"] == "VALID" and r["F2_above_roundoff"] == (r["condition"] == "M1") for r in h1)
    harm = any(c["mean"] > 1e-4 and c["losses"] >= 3 for c in comparisons.values())
    absent = all(not r["F2_above_roundoff"] for r in h1 if r["condition"] == "M1")
    verdict = ("REJECT_EARLY_NONLINEARITY" if harm or absent else
               "PROMISING_EARLY_NONLINEARITY" if gate and comparisons["selected"]["mean"] <= 1e-4 and comparisons["selected"]["losses"] <= 2
               else "MIXED")
    summary = {"verdict": verdict, "mechanism_gate_pass": gate, "systematic_prediction_harm": harm,
               "prediction_comparison": comparisons, "fits": fits, "F2_checks": f2_checks,
               "selection_lock_sha256": selection_hash, "protocol_sha256": sha(OUT / "protocol.json"),
               "preflight_correctness": protocol["preflight_correctness"], "F2_stimuli_unchanged": sha(F2_DIR / "stimuli.npz") == protocol["F2_stimuli_sha256"],
               "source_hash_changes": {name: {"locked": value, "current": sha(ROOT / name)} for name, value in protocol["sources_sha256"].items() if sha(ROOT / name) != value},
               "all_model_source_files_unchanged": all(sha(ROOT / name) == value for name, value in protocol["sources_sha256"].items() if name.startswith("models/")),
               "all_formal_checkpoints_unchanged": all(sha(ROOT / refs["checkpoint"]["path"]) == refs["checkpoint"]["sha256"] for refs in protocol["references"].values()),
               "total_optimizer_updates": 8 * UPDATES, "seeds": [SEED], "final_analysis_seconds": time.monotonic() - started,
               "artifact_sha256": {name: sha(OUT / name) for name in ("protocol.json", "per_cell_prediction.csv", "f2_comparison.csv")},
               "scope": "four-cell architecture FAST SCREEN, not population confirmation or physiological validation; no follow-on training"}
    save_json(OUT / "summary.json", summary)
    print(f"COMPLETE {verdict}; selected mean delta {comparisons['selected']['mean']:+.7f}; step200 {comparisons['step200']['mean']:+.7f}", flush=True)


def resume_frozen_analysis():
    started = time.monotonic()
    torch.set_num_threads(1)
    protocol = read_json(OUT / "protocol.json")
    lock_path = OUT / "checkpoints/selection_lock.json"
    lock = read_json(lock_path)
    assert not (OUT / "summary.json").exists() and not (OUT / "f2_comparison.csv").exists()
    allowed_change = str(Path(__file__).relative_to(ROOT))
    assert all(sha(ROOT / name) == value for name, value in protocol["sources_sha256"].items() if name != allowed_change)
    fits = []
    for cell in CELLS:
        for condition in ("M0", "M1"):
            record = read_json(OUT / "checkpoints" / cell.replace("#", "_") / condition / "training_record.json")
            assert record["optimizer_updates"] == UPDATES
            selected = next(item for item in lock["models"] if item["cell"] == cell and item["condition"] == condition)
            assert record["checkpoint_files"]["best"]["sha256"] == selected["sha256"] == sha(ROOT / selected["path"])
            assert record["selected"]["step"] == selected["selected_step"]
            fits.append(record)
    with (OUT / "per_cell_prediction.csv").open(encoding="utf-8", newline="") as handle:
        predictions = [{key: (float(value) if key not in ("cell", "polarity") else value)
                        for key, value in row.items()} for row in csv.DictReader(handle)]
    save_json(OUT / "checkpoints/inference_check_note.json", {
        "reason": "Initial float64 CUDA trace/forward bitwise comparison differed by 3.774758283725532e-15 (2/4200 logits). CPU float32 preflight was bitwise exact.",
        "repair": "Retain and report actual errors; bound logits by 64*eps64*max(1,max_abs_logit), probability by 64*eps64. Frozen F2 projection and 4096*eps64 localization floor unchanged.",
        "training_repeated": False, "selection_lock_unchanged": True,
        "old_runner_sha256": protocol["sources_sha256"][allowed_change], "new_runner_sha256": sha(Path(__file__)),
    })
    finish_analysis(fits, predictions, sha(lock_path), protocol, started)


if __name__ == "__main__":
    if sys.argv[1:] == ["--finish-frozen-analysis"]:
        resume_frozen_analysis()
    elif not sys.argv[1:]:
        run()
    else:
        raise ValueError("Unsupported arguments")
