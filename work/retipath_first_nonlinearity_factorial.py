from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "work")]
import numpy as np
import torch
from retipath_first_nonlinearity_locus.model import build, observe
from retipath_phase2_common import load_development, evaluate
from retipath_final_common import identity
from retipath_spatial_ei_pilot import sha
import retipath_f2_localization_pilot as f2

OUT = ROOT / "output/evaluations/retipath_first_nonlinearity_locus_factorial_20260916"
MANIFEST = ROOT / "output/evaluations/retipath_gain_canonicalization_20260915/migration_manifest.json"
PREVIOUS = ROOT / "output/experiments/retipath_first_nonlinearity_locus_fast_screen_20260916/protocol.json"
F2_DIR = ROOT / "output/evaluations/retipath_f2_localization_pilot_20260916"
CELLS = ("67#6", "67#7", "68#3", "68#10")
SEED = 2026091301
CONDITIONS = {"A": (False, True), "B": (True, False), "C": (True, True), "D": (False, False)}
NAMES = {"A": "CURRENT", "B": "RELOCATED", "C": "BOTH_DIAGNOSTIC", "D": "LINEARIZED_DIAGNOSTIC"}
TRACE_FIELDS = ("pre_h1_input", "h1_feedback", "bc_direct_effective_drive", "bc_broad_effective_drive",
                "effective_drive_E", "effective_drive_I", "membrane_readout", "logit")
F2_FIELDS = ("h1_graph_drive", "h1_state", "h1_feedback", "bc_direct_effective_drive",
             "bc_broad_effective_drive", "effective_drive_E", "effective_drive_I", "membrane_readout", "logit", "probability")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def transformed(x, alpha, early):
    return torch.where(x >= 0, x, alpha * x) if early else x


def frozen_model(cp, condition, *, dtype=torch.float32, device="cpu"):
    early, late = CONDITIONS[condition]
    model, _ = build(cp, "M0" if late else "M1", dtype=dtype, device=device)
    if condition == "C":
        model.h1.register_forward_pre_hook(lambda module, args: (transformed(args[0], model.alpha, True),))
    elif condition == "D":
        model.effective_input = lambda x: x
    model.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    return model


def state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def unchanged(model, before):
    return all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def trace_development(model, dev, condition, logits):
    chunks = {field: [] for field in TRACE_FIELDS}
    max_error = 0.0
    for start in range(0, len(dev.cone_drive), 8):
        x, y = dev.cone_drive[start:start + 8], dev.spike_events[start:start + 8]
        traced = observe(model, x, y)
        traced["pre_h1_input"] = transformed(x, model.alpha, CONDITIONS[condition][0])
        error = float((traced["logit"] - logits[start:start + 8]).abs().max())
        max_error = max(max_error, error)
        torch.testing.assert_close(traced["logit"], logits[start:start + 8], rtol=0, atol=16 * torch.finfo(torch.float32).eps)
        for field in TRACE_FIELDS:
            chunks[field].append(traced[field].cpu())
    mask = dev.valid_mask.reshape(len(dev.cone_drive), 150, -1)
    assert mask.shape[-1] == 1
    return {field: torch.cat(values)[mask[..., 0]].double().numpy().reshape(-1)
            for field, values in chunks.items()}, max_error


def f2_rows(model, cell, condition, x, time_s, cases):
    y = torch.zeros((2, 600, 1), dtype=torch.float64, device=x.device)
    traced = observe(model, x, y)
    assert all(bool(torch.isfinite(value).all()) for value in traced.values())
    assert torch.count_nonzero(traced["history_state"]) == 0
    rows = []
    for field in F2_FIELDS:
        value = traced[field].cpu().numpy()
        c = f2.project(value, time_s)
        floor = f2.FLOOR_MULTIPLIER * np.finfo(np.float64).eps * np.max(abs(value[:, f2.MEASURE]), axis=1)
        for group, coefficients in f2.groups(field, c):
            floors = dict(f2.groups(field, floor))[group]
            denominator, denominator_floor = f2.rms(coefficients[1, 0]), f2.rms(floors[1])
            valid = denominator > denominator_floor
            for index, case in enumerate(cases):
                a1, a2, floor_group = f2.rms(coefficients[index, 0]), f2.rms(coefficients[index, 1]), f2.rms(floors[index])
                rows.append({"cell": cell, "seed": SEED, "condition": condition, "condition_name": NAMES[condition],
                             "alpha": float(model.alpha), "case": case["case"], "kind": case["kind"],
                             "trace": field, "group": group, "F1": a1, "F2": a2,
                             "uniform_F1": denominator, "F2_over_uniform_F1": a2 / denominator if valid else None,
                             "roundoff_reference": floor_group, "F2_above_roundoff": a2 > floor_group,
                             "normalization_status": "VALID" if valid else "UNRESOLVED_DENOMINATOR"})
    return rows


def effects(nll):
    a, b, c, d = (nll[condition] for condition in "ABCD")
    return {"early_addition_effect": c - a, "bc_removal_given_early": b - c,
            "bc_removal_without_early": d - a, "early_addition_without_bc": b - d,
            "factorial_interaction": (c - a) - (b - d), "relocation_total_B_minus_A": b - a}


def run():
    if OUT.exists():
        raise FileExistsError("No automatic repeat or overwrite")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, previous = read_json(MANIFEST), read_json(PREVIOUS)
    entries = [next(e for e in manifest["checkpoints"] if e["cell"] == cell and e["seed"] == SEED) for cell in CELLS]
    for entry in entries:
        assert entry["sha256"] == sha(ROOT / entry["path"])
        assert entry["path"] == previous["references"][entry["cell"]]["checkpoint"]["path"]
    sources = [Path(__file__), ROOT / "work/retipath_first_nonlinearity_locus/model.py",
               ROOT / "work/retipath_phase2_common.py", ROOT / "work/retipath_final_common.py",
               ROOT / "work/retipath_f2_localization_pilot.py",
               ROOT / "evaluation/mechanistic_retina/mechanism_observation.py"]
    sources += sorted((ROOT / "models/mechanistic_retina").glob("*.py"))
    protocol = {
        "task": "first-nonlinearity locus 2x2 frozen diagnostic", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "locked_before_response": True, "cells": list(CELLS), "seed": SEED,
        "checkpoints": [{key: e[key] for key in ("cell", "seed", "path", "sha256")} for e in entries],
        "manifest_sha256": sha(MANIFEST), "previous_protocol_sha256": sha(PREVIOUS),
        "conditions": {key: {"name": NAMES[key], "pre_H1_phi": flags[0], "BC_common_branch_phi": flags[1]}
                       for key, flags in CONDITIONS.items()},
        "alpha": "one original frozen alpha=exp(theta), reused at both enabled locations in C; no second coordinate",
        "data": "Only development [16,20); mmap cache metadata then access only development tensors. No train tensors, test blocks or new targets.",
        "data_contract": "150Hz Bernoulli occupancy, original 150-bin independent sequences,30-bin warmup,120 scored; original mask and strictly-past observed history; CPU float32 forward, float64 NLL reduction",
        "development_identities": {cell: previous["references"][cell]["development_identity"] for cell in CELLS},
        "checkpoint_selection": None, "training_updates": 0, "new_checkpoints": 0,
        "effects": {"early_addition_effect": "C-A", "bc_removal_given_early": "B-C",
                    "bc_removal_without_early": "D-A", "early_addition_without_bc": "B-D",
                    "factorial_interaction": "(C-A)-(B-D)", "equal_weight": "mean across four biological cells; no significance inference"},
        "attribution_rule": "Mainly early addition only if both mean early-addition contrasts are positive and the smaller exceeds both absolute mean BC-removal contrasts, with both early-addition contrasts positive in >=3/4 cells; symmetric rule for mainly BC removal. Otherwise STRONG_INTERACTION_OR_MIXED. Compare both paths; this ordering rule is descriptive, not a statistical or biological threshold.",
        "F2": {"stimuli_sha256": sha(F2_DIR / "stimuli.npz"), "protocol_sha256": sha(F2_DIR / "protocol.json"),
               "cases": ["grating_w2_phase0", "uniform_full_field"], "frequency_hz": 4,
               "max_Weber_amplitude": 0.5, "pre_roll_cycles": 8, "measurement_cycles": 8,
               "history": "FIX_HISTORY_ZERO", "dtype_device": "float64 CUDA inference copies",
               "Fourier_reduction": "unchanged frozen per-component signed projection then RMS; denominator is matched uniform F1; frozen 4096*eps64 raw-amplitude roundoff reference",
               "trace_fields": list(F2_FIELDS)},
        "trace_diagnostic": {"cell": "67#6", "fields": list(TRACE_FIELDS),
                             "reduction": "same scored development bins; flatten original components after masking. Mean, RMS, centered SD, negative fraction; shift/difference RMS relative to A. No recentering or normalization applied to inputs.",
                             "raw_trace_saved": False},
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "boundaries": "C and D diagnostic only; no architecture acceptance, biology validation, optimizer, checkpoint writes, response-driven corrections or population run",
    }
    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol)
    protocol_hash = sha(OUT / "protocol.json")
    with np.load(F2_DIR / "stimuli.npz", allow_pickle=False) as bank:
        all_cases = json.loads(str(bank["cases_json"]))
        ids = [next(i for i, case in enumerate(all_cases) if case["case"] == name) for name in protocol["F2"]["cases"]]
        stimuli, time_s, cases = bank["stimulus_weber"][ids].copy(), bank["time_s"].copy(), [all_cases[i] for i in ids]
    assert stimuli.shape == (2, 600, 289) and float(stimuli.min()) >= -1
    x_f2 = torch.tensor(stimuli, dtype=torch.float64, device="cuda")
    predictions, harmonics, traces, checks = [], [], [], []
    with torch.inference_mode():
        for entry in entries:
            cell = entry["cell"]
            cp = torch.load(ROOT / entry["path"], map_location="cpu", weights_only=True)
            assert cp["gain_schema"] == "retipath_canonical_gain_v1" and cp["training_updates"] == 0
            assert cp["cell_id"] == cell and cp["seed"] == SEED
            dev = load_development(previous["references"][cell])
            assert identity(dev) == protocol["development_identities"][cell]
            nll, local, parts, cell_checks = {}, {}, {}, {}
            for condition in CONDITIONS:
                model = frozen_model(cp, condition)
                before = state(model)
                assert all(torch.equal(before[key], value) for key, value in cp["model"].items())
                nll[condition], logits = evaluate(model, dev)
                part = model.spatial_components(dev.cone_drive[:1])
                parts[condition] = part
                feature = model.feature_bank(part.h1.modulated_cones, mixer=model.shared_subunits)
                weighted = feature * model.bipolar.positive_weights().repeat(1, 2, 1, 1)[None, None]
                pooled = weighted.sum((-1, -2))
                slope = torch.where(pooled >= 0, torch.ones_like(pooled), model.alpha) if CONDITIONS[condition][1] else torch.ones_like(pooled)
                expected = (weighted.sum(-1) * slope[..., None]).transpose(-1, -2)
                torch.testing.assert_close(part.direct, expected[..., :2], rtol=0, atol=0)
                torch.testing.assert_close(part.broad, expected[..., 2:], rtol=0, atol=0)
                pre_input = transformed(dev.cone_drive[:1], model.alpha, CONDITIONS[condition][0])
                torch.testing.assert_close(part.h1.modulated_cones, pre_input - part.h1.surround, rtol=0, atol=0)
                trace_error = None
                if cell == "67#6":
                    local[condition], trace_error = trace_development(model, dev, condition, logits)
                assert unchanged(model, before)
                model64 = frozen_model(cp, condition, dtype=torch.float64, device="cuda")
                before64 = state(model64)
                harmonics.extend(f2_rows(model64, cell, condition, x_f2, time_s, cases))
                assert unchanged(model64, before64)
                cell_checks[condition] = {"all_parameters_frozen": True, "state_unchanged": True,
                                          "BC_location_semantics_exact": True, "pre_H1_semantics_exact": True,
                                          "trace_forward_max_abs_logit_error": trace_error}
            for first, second in (("A", "D"), ("B", "C")):
                torch.testing.assert_close(parts[first].h1.surround, parts[second].h1.surround, rtol=0, atol=0)
                torch.testing.assert_close(parts[first].h1.state, parts[second].h1.state, rtol=0, atol=0)
            row = {"cell": cell, "seed": SEED, "alpha": float(cp["model"]["theta"].exp()),
                   **{"NLL_" + key: value for key, value in nll.items()}, **effects(nll)}
            assert abs(row["early_addition_effect"] + row["bc_removal_given_early"] - row["relocation_total_B_minus_A"]) < 1e-14
            assert abs(row["bc_removal_without_early"] + row["early_addition_without_bc"] - row["relocation_total_B_minus_A"]) < 1e-14
            predictions.append(row)
            checks.append({"cell": cell, "development_identity_exact": True, "H1_A_equals_D_B_equals_C": True,
                           "conditions": cell_checks, "original_checkpoint_unchanged": sha(ROOT / entry["path"]) == entry["sha256"]})
            if local:
                for condition, fields in local.items():
                    for field, value in fields.items():
                        base = local["A"][field]
                        rms, base_rms = float(np.sqrt(np.mean(value ** 2))), float(np.sqrt(np.mean(base ** 2)))
                        traces.append({"cell": cell, "seed": SEED, "condition": condition, "trace": field,
                                       "component_values": int(value.size), "mean": float(value.mean()), "RMS": rms,
                                       "centered_SD": float(value.std()), "negative_fraction": float(np.mean(value < 0)),
                                       "mean_shift_vs_A": float((value - base).mean()), "RMS_ratio_vs_A": rms / base_rms,
                                       "difference_RMS_vs_A": float(np.sqrt(np.mean((value - base) ** 2)))})
            print(f"{cell}: original checkpoint; A/B/C/D scored; two-case F2 sanity complete", flush=True)
    mean = {key: float(np.mean([row[key] for row in predictions])) for key in predictions[0] if key not in ("cell", "seed", "alpha")}
    early = (mean["early_addition_effect"], mean["early_addition_without_bc"])
    removal = (mean["bc_removal_given_early"], mean["bc_removal_without_early"])
    early_consistency = sum(row["early_addition_effect"] > 0 and row["early_addition_without_bc"] > 0 for row in predictions)
    removal_consistency = sum(row["bc_removal_given_early"] > 0 and row["bc_removal_without_early"] > 0 for row in predictions)
    verdict = ("PENALTY_MAINLY_EARLY_ADDITION" if min(early) > max(map(abs, removal)) and early_consistency >= 3 else
               "PENALTY_MAINLY_BC_REMOVAL" if min(removal) > max(map(abs, early)) and removal_consistency >= 3 else
               "STRONG_INTERACTION_OR_MIXED")
    save_csv(OUT / "per_cell_nll.csv", predictions)
    save_csv(OUT / "minimal_f2.csv", harmonics)
    save_csv(OUT / "trace_summary.csv", traces)
    assert sha(OUT / "protocol.json") == protocol_hash
    assert all(sha(ROOT / name) == value for name, value in protocol["source_sha256"].items())
    assert sha(F2_DIR / "stimuli.npz") == protocol["F2"]["stimuli_sha256"]
    save_json(OUT / "summary.json", {"verdict": verdict, "equal_cell_mean": mean, "per_cell": predictions,
              "correctness": checks, "early_positive_in_both_contexts_cells": early_consistency,
              "removal_positive_in_both_contexts_cells": removal_consistency,
              "source_files_unchanged": True, "stimuli_unchanged": True, "new_checkpoints": 0,
              "training_updates": 0, "f2_case_count": 2, "trace_summary_cell": "67#6",
              "artifact_sha256": {name: sha(OUT / name) for name in ("protocol.json", "per_cell_nll.csv", "minimal_f2.csv", "trace_summary.csv")}})
    print(f"COMPLETE: {verdict}; mean early {early}; mean removal {removal}; interaction {mean['factorial_interaction']:+.8f}", flush=True)


if __name__ == "__main__":
    run()
