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
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.mechanistic_retina.mechanism_observation import (
    CANONICAL_TRACE_VARIABLE_SPECS, Intervention, InterventionSpec, observe_mechanism,
)
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.retipath_canonical_gain import CANONICAL_GAIN_SCHEMA, CanonicalGainRetiPath

OUT = ROOT / "output/evaluations/retipath_nonlinear_spatial_integration_pilot_20260916"
MANIFEST = ROOT / "output/evaluations/retipath_gain_canonicalization_20260915/migration_manifest.json"
CELLS = ("67#6", "67#7")
SEED = 2026091301
C0 = 0.9340826124
FRACTIONS = (0.25, 0.5, 1.0)
EPS = 0.01
REPAIR = sys.argv[1:] == ["--finalize-saved"]
FIELDS = (
    "h1_graph_drive", "h1_state", "h1_feedback", "h1_modulated_input",
    "bc_direct_effective_drive", "bc_broad_effective_drive", "ac_states", "ac_inhibitory_drive",
    "normalized_drive_E", "normalized_drive_I", "effective_drive_E", "effective_drive_I",
    "gE", "gI", "V1", "V2", "membrane_readout", "adaptation_state", "adaptation_term", "logit",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                         default=lambda item: item.item())
    mode = "x"
    if REPAIR and Path(path).name == "summary.json" and Path(path).exists():
        try:
            json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            mode = "w"
        else:
            raise FileExistsError("A valid summary already exists")
    with Path(path).open(mode, encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def save_npz(path, values):
    if REPAIR and Path(path).exists():
        with np.load(path, allow_pickle=False) as saved:
            assert set(saved.files) == set(values)
            assert all(np.allclose(saved[key], value, rtol=0, atol=1e-14) if np.issubdtype(value.dtype, np.floating)
                       else np.array_equal(saved[key], value) for key, value in values.items())
        return
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **values)


def load_model(entry):
    path = ROOT / entry["path"]
    assert sha(path) == entry["sha256"]
    cp = torch.load(path, map_location="cpu", weights_only=True)
    assert cp["cell_id"] == entry["cell"] and cp["seed"] == SEED
    assert cp["gain_schema"] == CANONICAL_GAIN_SCHEMA and cp["training_updates"] == 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = CanonicalGainRetiPath(
            MechanisticRetinaConfig(**cp["model_config"]), cp["cone_positions_degs"],
            cp["cell_positions_degs"], tuple(cp["cell_types"]), tuple(cp["polarities"]),
            rms_e=torch.tensor(cp["rms"]["e"]), rms_i=torch.tensor(cp["rms"]["i"]),
        )
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float64)
    assert cp["cone_positions_degs"].shape == (289, 2)
    assert abs(cp["model_config"]["dt_ms"] - 1000 / 150) < 1e-6
    return model, cp


@torch.inference_mode()
def evaluate(model, spatial, keep_traces=False):
    collected = {name: [] for name in (FIELDS if keep_traces else ("logit",))}
    for start in range(0, len(spatial), 32):
        patterns = torch.as_tensor(spatial[start:start + 32], device="cuda", dtype=torch.float64)
        x = patterns.new_zeros((len(patterns), 150, 289))
        x[:, 30:90] = patterns[:, None]
        history = patterns.new_zeros((len(patterns), 150, 1))
        trace = observe_mechanism(model, x, observed_counts=history,
                                  intervention=InterventionSpec(Intervention.FIX_HISTORY_ZERO))
        assert trace.schema_version == 2
        assert all(bool(torch.isfinite(v).all()) for v in trace.tensors.values())
        assert torch.count_nonzero(trace.tensors["history_state"]) == 0
        for name in collected:
            collected[name].append(trace.tensors[name].cpu().numpy())
    return {name: np.concatenate(values) for name, values in collected.items()}


def build_patterns(k, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    assert np.linalg.norm(k) > 0
    assert np.linalg.norm(k - k.mean()) > 1e-12 * np.linalg.norm(k)
    q, _ = np.linalg.qr(np.column_stack((np.ones(289), k / np.linalg.norm(k))))
    accepted, accepted_indices, rejected = [], [], []
    for candidate in range(1000):
        raw = rng.standard_normal(289)
        u = raw - q @ (q.T @ raw)
        u -= q @ (q.T @ u)
        rms = np.sqrt(np.mean(u ** 2))
        if rms < 1e-12:
            rejected.append({"candidate": candidate, "reason": "degenerate_projection"})
            continue
        u /= rms
        if np.max(np.abs(u)) > 4.0:
            rejected.append({"candidate": candidate, "reason": "max_abs_exceeds_4_RMS"})
            continue
        accepted.append(u)
        accepted_indices.append(candidate)
        if len(accepted) == 6:
            break
    assert len(accepted) == 6, "No response-dependent replacement or relaxed stimulus rule allowed"
    return np.stack(accepted), k / np.sqrt(np.mean(k ** 2)), accepted_indices, rejected


def cases_and_inputs(null, aligned):
    cases = [{"case": "zero", "kind": "zero", "pattern": -1, "fraction_C0": 0.0, "amplitude": 0.0, "sign": 0}]
    inputs = [np.zeros(289)]
    for fraction in FRACTIONS:
        for kind, patterns in (("null", null), ("aligned", aligned[None])):
            for index, pattern in enumerate(patterns):
                for sign in (1, -1):
                    cases.append({"case": f"{kind}_{index}_{fraction}_{sign}", "kind": kind,
                                  "pattern": index, "fraction_C0": fraction,
                                  "amplitude": fraction * C0, "sign": sign})
                    inputs.append(sign * fraction * C0 * pattern)
    return cases, np.stack(inputs)


def run():
    started = time.monotonic()
    if OUT.exists() and not REPAIR:
        raise FileExistsError("Refusing to overwrite or automatically repeat this pilot")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = [next(e for e in manifest["checkpoints"] if e["cell"] == cell and e["seed"] == SEED) for cell in CELLS]
    source_paths = [Path(__file__), ROOT / "evaluation/mechanistic_retina/mechanism_observation.py"]
    source_paths += sorted((ROOT / "models/mechanistic_retina").glob("*.py"))
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    protocol = {
        "task": "frozen nonlinear spatial-integration pilot", "locked_utc": datetime.now(timezone.utc).isoformat(),
        "locked_before_any_response": True, "cells": list(CELLS), "checkpoint_seed": SEED,
        "checkpoints": [{k: e[k] for k in ("cell", "seed", "path", "sha256")} for e in entries],
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST), "source_sha256": sources,
        "checkpoint_load_count": 2, "training_updates": 0, "dtype": "float64 inference copy", "device": "cuda",
        "bins": 150, "sampling_hz": 150, "stimulus_bins": [30, 90], "late_bins": [75, 90],
        "background_weber": 0, "history": "FIX_HISTORY_ZERO", "other_interventions": [],
        "initial_state": "unchanged default: V1/V2=V0=2/9; other dynamic deviations=0",
        "response": "mean_t(z(X)-z(0)), t in [75,90); fixed temporal envelope for every pixel",
        "kernel": "k_p=[R(+epsilon e_p)-R(-epsilon e_p)]/(2 epsilon)", "epsilon_weber": EPS,
        "patterns": {"count_per_cell": 6, "generator": "PCG64 standard_normal, dimension 289",
                     "seeds": {"67#6": 2026091606, "67#7": 2026091607},
                     "projection": "two passes of I-Q Q^T; Q=QR([ones, k/||k||]); no spatial resampling",
                     "spatial_RMS": 1.0, "rejection": "projected RMS<1e-12 or normalized max_abs>4; first 6 accepted of at most 1000",
                     "selection_uses_finite_contrast_response": False,
                     "conditioning": "projection uses only the mandated per-pixel epsilon kernel", "signs": [1, -1], "clipping": False},
        "controls": "zero; aligned v=k/RMS(k), both signs at the same three amplitudes; aligned is not reprojected",
        "C0_weber": C0, "C0_source": "user-provided previously frozen train-only scale; no train stimulus accessed",
        "amplitude_fractions": list(FRACTIONS), "amplitudes_weber_RMS": [f * C0 for f in FRACTIONS],
        "pixel_range_policy": "save actual extrema and count pixels below -1 Weber; no clipping or response-based exclusion",
        "linear_prediction": "dot(k,x)", "nonlinear_residual": "N=R_full-dot(k,x)",
        "primary": "mean(abs(N)) over 6 null patterns and both signs, separately per cell and amplitude",
        "odd": "[R(+u)-R(-u)]/2", "even": "[R(+u)+R(-u)]/2-R(0)",
        "trace_reduction": "time mean over [75,90), retain all spatial/mode/pathway axes; subtract matched zero trace; then Even/Odd elementwise",
        "trace_summary": "signed arithmetic mean and RMS over retained components, AFTER Even/Odd; all full traces and per-component late responses saved",
        "trace_order": list(FIELDS), "trace_schema_version": 2,
        "sanity_numeric_tolerance": {"max_null_to_aligned_dot_ratio": 1e-12, "max_unit_pattern_mean_abs": 1e-12,
                                      "max_RMS_error": 1e-12},
        "effect_pass_threshold": None,
        "verdict": "descriptive effect presence and numerical controls; no significance or biological validation threshold",
        "prohibited": ["training", "checkpoint writes", "other cells/seeds", "DoG/radius fitting", "external reference access",
                       "natural movies/spike targets", "pathway clamps", "alpha interventions", "architecture/FOV/support/preprocessing changes"],
    }
    if REPAIR:
        protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
        assert protocol["cells"] == list(CELLS) and protocol["epsilon_weber"] == EPS
        assert protocol["amplitude_fractions"] == list(FRACTIONS) and protocol["C0_weber"] == C0
        sources = protocol["source_sha256"]
        assert all(sha(ROOT / name) == value for name, value in sources.items() if name != str(Path(__file__).relative_to(ROOT)))
    else:
        OUT.mkdir(parents=True)
        save_json(OUT / "protocol.json", protocol)
    protocol_hash = sha(OUT / "protocol.json")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models, originals, cells = {}, {}, {}
    stimulus_archive = {"protocol_sha256": np.array(protocol_hash)}
    for entry in entries:
        cell = entry["cell"]
        key = cell.replace("#", "_")
        model, cp = load_model(entry)
        models[cell] = model
        originals[cell] = {name: value.clone() for name, value in model.state_dict().items()}
        if REPAIR:
            with np.load(OUT / "stimuli.npz", allow_pickle=False) as saved:
                k, null, aligned = (saved[f"{key}__{name}"] for name in ("kernel", "null_unit_RMS", "aligned_unit_RMS"))
                pixel_r = saved[f"{key}__kernel_pixel_R_plus_minus"].ravel()
                accepted = saved[f"{key}__accepted_candidates"].tolist()
                rejected = json.loads(str(saved[f"{key}__rejected_candidates_json"]))
        else:
            z0 = evaluate(model, np.zeros((1, 289)))["logit"]
            perturb = np.concatenate((EPS * np.eye(289), -EPS * np.eye(289)))
            pixel_z = evaluate(model, perturb)["logit"]
            pixel_r = (pixel_z[:, 75:90] - z0[:, 75:90]).mean((1, 2))
            k = (pixel_r[:289] - pixel_r[289:]) / (2 * EPS)
            null, aligned, accepted, rejected = build_patterns(k, protocol["patterns"]["seeds"][cell])
        cases, inputs = cases_and_inputs(null, aligned)
        cells[cell] = {"key": key, "k": k, "null": null, "aligned": aligned, "cases": cases, "inputs": inputs,
                       "accepted": accepted, "rejected": rejected,
                       "alpha": float(model.alpha), "cell_type": list(cp["cell_types"]), "polarity": list(cp["polarities"])}
        stimulus_archive.update({f"{key}__kernel": k, f"{key}__kernel_pixel_R_plus_minus": pixel_r.reshape(2, 289),
                                 f"{key}__null_unit_RMS": null, f"{key}__aligned_unit_RMS": aligned,
                                 f"{key}__inputs": inputs, f"{key}__cases_json": np.array(json.dumps(cases)),
                                 f"{key}__accepted_candidates": np.array(accepted),
                                 f"{key}__rejected_candidates_json": np.array(json.dumps(rejected)),
                                 f"{key}__cone_positions_deg": cp["cone_positions_degs"].numpy(),
                                 f"{key}__checkpoint_sha256": np.array(entry["sha256"])})
        print(f"{cell}: {'reusing saved' if REPAIR else 'froze'} kernel and six projected patterns", flush=True)
    save_npz(OUT / "stimuli.npz", stimulus_archive)
    stimuli_hash = sha(OUT / "stimuli.npz")
    trace_archive = {"protocol_sha256": np.array(protocol_hash), "stimuli_sha256": np.array(stimuli_hash)}
    rows, cell_summaries = [], []
    for entry in entries:
        cell = entry["cell"]
        data = cells[cell]
        key, cases, inputs, k = (data[name] for name in ("key", "cases", "inputs", "k"))
        traces = evaluate(models[cell], inputs, keep_traces=True)
        replay_errors = {}
        if REPAIR:
            with np.load(OUT / "traces.npz", allow_pickle=False) as saved:
                for name in FIELDS:
                    original = saved[f"{key}__{name}"]
                    replay_errors[name] = float(np.max(abs(traces[name] - original)))
                    assert replay_errors[name] < 1e-12, (cell, name, replay_errors[name])
                    traces[name] = original
        late = {name: value[:, 75:90].mean(1) for name, value in traces.items()}
        responses = {name: value - value[0:1] for name, value in late.items()}
        r = responses["logit"].ravel()
        r_lin = inputs @ k
        residual = r - r_lin
        even, odd = {}, {}
        for name, value in responses.items():
            e, o = np.zeros_like(value), np.zeros_like(value)
            for i, case in enumerate(cases):
                if case["sign"] == 1:
                    assert cases[i + 1]["sign"] == -1 and cases[i + 1]["kind"] == case["kind"]
                    e[i] = e[i + 1] = (value[i] + value[i + 1]) / 2
                    o[i] = o[i + 1] = (value[i] - value[i + 1]) / 2
            even[name], odd[name] = e, o
            trace_archive.update({f"{key}__{name}": traces[name], f"{key}__late_R__{name}": value,
                                  f"{key}__late_even__{name}": e, f"{key}__late_odd__{name}": o})
        trace_archive[f"{key}__cases_json"] = np.array(json.dumps(cases))
        metadata = {name: {"unit": CANONICAL_TRACE_VARIABLE_SPECS[name].normalization_unit,
                           "meaning": CANONICAL_TRACE_VARIABLE_SPECS[name].mathematical_meaning,
                           "shape": list(traces[name].shape)} for name in FIELDS}
        trace_archive[f"{key}__metadata_json"] = np.array(json.dumps(metadata))
        for i, case in enumerate(cases):
            rows.append({"cell": cell, "seed": SEED, **case, "R_full": r[i], "R_lin": r_lin[i],
                         "N": residual[i], "abs_N": abs(residual[i]), "Odd": float(odd["logit"][i, 0]),
                         "Even": float(even["logit"][i, 0]), "spatial_RMS": np.sqrt(np.mean(inputs[i] ** 2)),
                         "pixel_min_Weber": inputs[i].min(), "pixel_max_Weber": inputs[i].max(),
                         "pixels_below_minus_one": int(np.count_nonzero(inputs[i] < -1))})
        summaries, trace_summaries = [], []
        for fraction in FRACTIONS:
            indices = [i for i, case in enumerate(cases) if case["kind"] == "null" and case["fraction_C0"] == fraction]
            pairs = [i for i in indices if cases[i]["sign"] == 1]
            aligned = [i for i, case in enumerate(cases) if case["kind"] == "aligned" and case["fraction_C0"] == fraction]
            summaries.append({"fraction_C0": fraction, "amplitude_RMS": fraction * C0,
                              "mean_abs_N": float(np.mean(abs(residual[indices]))),
                              "min_abs_N": float(np.min(abs(residual[indices]))), "max_abs_N": float(np.max(abs(residual[indices]))),
                              "mean_R": float(np.mean(r[indices])), "max_abs_R_lin": float(np.max(abs(r_lin[indices]))),
                              "mean_abs_Even": float(np.mean(abs(even["logit"][pairs]))),
                              "mean_Even": float(np.mean(even["logit"][pairs])),
                              "mean_abs_Odd": float(np.mean(abs(odd["logit"][pairs]))),
                              "positive_R": int(np.sum(r[indices] > 0)), "negative_R": int(np.sum(r[indices] < 0)),
                              "aligned_R": r[aligned].tolist(), "aligned_R_lin": r_lin[aligned].tolist(),
                              "null_pixel_min": float(inputs[indices].min()), "null_pixel_max": float(inputs[indices].max()),
                              "null_pixel_fraction_below_minus_one": float(np.mean(inputs[indices] < -1))})
            for name in FIELDS:
                e, o = even[name][pairs].reshape(6, -1), odd[name][pairs].reshape(6, -1)
                trace_summaries.append({"fraction_C0": fraction, "trace": name,
                                        "mean_signed_Even": float(e.mean()), "mean_signed_Odd": float(o.mean()),
                                        "mean_component_RMS_Even": float(np.sqrt(np.mean(e ** 2, axis=1)).mean()),
                                        "mean_component_RMS_Odd": float(np.sqrt(np.mean(o ** 2, axis=1)).mean()),
                                        "max_abs_Even_component": float(abs(e).max()),
                                        "per_pattern_RMS_Even": np.sqrt(np.mean(e ** 2, axis=1)).tolist(),
                                        "per_pattern_RMS_Odd": np.sqrt(np.mean(o ** 2, axis=1)).tolist()})
        relative_null = float(np.max(abs(data["null"] @ k)) / abs(data["aligned"] @ k))
        rms_error = float(np.max(abs(np.sqrt(np.mean(data["null"] ** 2, axis=1)) - 1)))
        pair_rms_error = max(abs(np.sqrt(np.mean(inputs[i] ** 2)) - np.sqrt(np.mean(inputs[i + 1] ** 2)))
                             for i, case in enumerate(cases) if case["sign"] == 1)
        small_control = [i for i, case in enumerate(cases) if case["kind"] == "aligned" and case["fraction_C0"] == 0.25]
        checks = {"null_dot_relative_to_aligned": relative_null, "null_dot_pass": relative_null < 1e-12,
                  "max_abs_null_pattern_mean": float(abs(data["null"].mean(1)).max()),
                  "zero_mean_pass": bool(abs(data["null"].mean(1)).max() < 1e-12),
                  "unit_RMS_error": rms_error, "unit_RMS_pass": rms_error < 1e-12,
                  "plus_minus_RMS_difference": float(pair_rms_error), "plus_minus_RMS_pass": pair_rms_error < 1e-12,
                  "smallest_amplitude_aligned_direction_pass": bool(np.all(r[small_control] * r_lin[small_control] > 0)),
                  "all_forwards_finite": True, "history_state_zero": True,
                  "state_dict_unchanged": all(torch.equal(value, originals[cell][name]) for name, value in models[cell].state_dict().items()),
                  "checkpoint_hash_unchanged": sha(ROOT / entry["path"]) == entry["sha256"]}
        cell_summaries.append({"cell": cell, "seed": SEED, "alpha": data["alpha"], "cell_type": data["cell_type"],
                               "polarity": data["polarity"], "accepted_candidates": data["accepted"], "rejected_candidates": data["rejected"],
                               "zero_late_logit": float(late["logit"][0, 0]), "sanity": checks,
                               "amplitudes": summaries, "trace_summaries": trace_summaries,
                               "same_stimulus_state_check_replay_max_abs_errors": replay_errors})
        print(f"{cell}: finite-contrast inference complete; mean |N| = {[round(s['mean_abs_N'], 6) for s in summaries]}", flush=True)
    if REPAIR:
        with (OUT / "responses.csv").open(encoding="utf-8", newline="") as handle:
            saved_rows = list(csv.DictReader(handle))
        assert len(saved_rows) == len(rows)
        assert all(all(str(value) == saved[key] for key, value in row.items()) for row, saved in zip(rows, saved_rows))
    else:
        with (OUT / "responses.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_npz(OUT / "traces.npz", trace_archive)
    summary = {"protocol_sha256": protocol_hash, "stimuli_sha256": stimuli_hash, "cells": cell_summaries,
               "responses_rows": len(rows), "elapsed_seconds": time.monotonic() - started,
               "model_and_trace_source_files_unchanged": all(sha(ROOT / name) == value for name, value in sources.items()
                                                            if name != str(Path(__file__).relative_to(ROOT))),
               "export_repair": {"performed": REPAIR, "reason": "numpy bool JSON serialization; all stimulus/response/trace artifacts already saved",
                                 "script_sha256_now": sha(__file__), "kernel_recomputed": False,
                                 "new_stimuli": 0, "new_unique_checkpoints": 0,
                                 "checkpoint_loads": 6 if REPAIR else 2,
                                 "export_replay_attempts": 2 if REPAIR else 0,
                                 "scope": "replay identical saved stimuli solely to persist state integrity checks; statistics use original saved traces"},
               "manifest_unchanged": sha(MANIFEST) == protocol["manifest_sha256"],
               "protocol_unchanged": sha(OUT / "protocol.json") == protocol_hash,
               "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name()},
               "artifact_sha256": {name: sha(OUT / name) for name in ("protocol.json", "stimuli.npz", "responses.csv", "traces.npz")},
               "scope": "model-conditioned functional probe; no physiological validation or population inference"}
    save_json(OUT / "summary.json", summary)
    print(f"Saved the five requested artifacts; elapsed {summary['elapsed_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    run()
