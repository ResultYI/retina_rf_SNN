from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from retipath_nonlinear_spatial_integration_pilot import (
    ROOT, MANIFEST, CELLS, SEED, load_model, sha, np, torch,
    observe_mechanism, Intervention, InterventionSpec, CANONICAL_TRACE_VARIABLE_SPECS,
)

OUT = ROOT / "output/evaluations/retipath_f2_localization_pilot_20260916"
RATE = 150
FREQUENCY = 4
CONTRAST = 0.5
MEASURE = slice(300, 600)
FLOOR_MULTIPLIER = 4096
FIELDS = (
    "h1_graph_drive", "h1_state", "h1_feedback", "h1_modulated_input",
    "bc_direct_effective_drive", "bc_broad_effective_drive", "ac_states", "ac_inhibitory_drive",
    "effective_drive_E", "effective_drive_I", "gE", "gI", "V1", "V2",
    "membrane_readout", "adaptation_state", "adaptation_term", "logit", "probability",
)
PATH_NAMES = {
    "bc_direct_effective_drive": ("sustained", "transient"),
    "bc_broad_effective_drive": ("sustained", "transient"),
    "ac_states": ("local", "transient"),
    "ac_inhibitory_drive": ("local", "transient"),
}


def save_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=lambda v: v.item())
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def save_npz(path, values):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **values)


def build_stimuli():
    patterns, cases = [], []
    for width in (1, 2, 4):
        for phase in (0.0, 0.5):
            shift = width * phase
            bar_index = np.floor((np.arange(17, dtype=np.float64) - shift) / width).astype(int)
            raw = np.tile(np.where(bar_index % 2 == 0, 1.0, -1.0), (17, 1)).ravel()
            balanced = raw - raw.mean()
            balanced /= np.max(abs(balanced))
            patterns.append(balanced)
            cases.append({"case": f"grating_w{width}_phase{phase:g}", "kind": "grating",
                          "bar_width_pixels": width, "phase_bar_fraction": phase,
                          "shift_pixels": shift, "unbalanced_grid_mean": float(raw.mean())})
    patterns.append(np.ones(289))
    cases.append({"case": "uniform_full_field", "kind": "uniform", "bar_width_pixels": None,
                  "phase_bar_fraction": None, "shift_pixels": None, "unbalanced_grid_mean": None})
    time_s = np.arange(600, dtype=np.float64) / RATE
    temporal = np.sin(2 * np.pi * FREQUENCY * time_s)
    spatial = np.stack(patterns)
    stimulus = CONTRAST * temporal[None, :, None] * spatial[:, None, :]
    assert np.max(abs(spatial[:6].mean(-1))) < 1e-14
    assert np.max(abs(stimulus[:6].mean(-1))) < 1e-14
    assert np.max(abs(stimulus[:, MEASURE].mean(1))) < 1e-14
    assert float(stimulus.min()) >= -1 and float(stimulus.max()) <= CONTRAST
    assert np.array_equal(spatial[0], -spatial[1])
    return spatial, temporal, stimulus, time_s, cases


def project(values, time_s):
    measured = values[:, MEASURE]
    centered = measured - measured.mean(1, keepdims=True)
    phase = np.exp(-2j * np.pi * np.array([4.0, 8.0])[:, None] * time_s[MEASURE][None, :])
    coefficients = 2 / measured.shape[1] * np.einsum("btd,ht->bhd", centered.reshape(len(values), 300, -1), phase)
    return coefficients.reshape(len(values), 2, *values.shape[2:])


def harmonic_fixture(time_s):
    t = time_s
    known = 2 + 3 * np.sin(2 * np.pi * 4 * t) + 0.7 * np.cos(2 * np.pi * 8 * t)
    result = project(known[None, :, None], t)[0, :, 0]
    error = float(np.max(abs(result - np.array([-3j, 0.7]))))
    opposing = np.stack((np.sin(2 * np.pi * 4 * t), -np.sin(2 * np.pi * 4 * t)), -1)[None]
    c = project(opposing, t)
    f1 = float(np.sqrt(np.mean(abs(c[0, 0]) ** 2)))
    f2 = float(np.sqrt(np.mean(abs(c[0, 1]) ** 2)))
    assert error < 1e-12 and abs(f1 - 1) < 1e-12 and f2 < 1e-12
    return {"known_complex_coefficient_max_error": error, "opposing_components_F1_RMS": f1,
            "opposing_components_F2_RMS": f2, "no_abs_or_norm_before_projection": True}


def groups(field, coefficient):
    if field in PATH_NAMES:
        return [(field + "." + label, coefficient[..., index]) for index, label in enumerate(PATH_NAMES[field])]
    return [(field, coefficient)]


def rms(values):
    return float(np.sqrt(np.mean(abs(values) ** 2)))


def run():
    start = time.monotonic()
    if OUT.exists():
        raise FileExistsError("This frozen assay is not automatically repeated or overwritten")
    spatial, temporal, stimulus, time_s, cases = build_stimuli()
    fixture = harmonic_fixture(time_s)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = [next(e for e in manifest["checkpoints"] if e["cell"] == cell and e["seed"] == SEED) for cell in CELLS]
    source_paths = [Path(__file__), ROOT / "work/retipath_nonlinear_spatial_integration_pilot.py",
                    ROOT / "evaluation/mechanistic_retina/mechanism_observation.py", ROOT / "data/schottdorf_lee_catalog.py"]
    source_paths += sorted((ROOT / "models/mechanistic_retina").glob("*.py"))
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    protocol = {
        "task": "frozen contrast-reversing-grating F2 localization pilot",
        "locked_utc": datetime.now(timezone.utc).isoformat(), "locked_before_any_model_response": True,
        "cells": list(CELLS), "seed": SEED,
        "checkpoints": [{key: entry[key] for key in ("cell", "seed", "path", "sha256")} for entry in entries],
        "manifest_path": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": sha(MANIFEST),
        "source_sha256": source_hashes, "dtype": "float64 inference copies", "device": "cuda; TF32 disabled",
        "history_condition": "FIX_HISTORY_ZERO", "pathway_interventions": [],
        "initial_state": "unchanged default; V1/V2=V0=2/9, other dynamic deviations=0",
        "sampling_hz": RATE, "temporal_frequency_hz": FREQUENCY, "peak_Weber_contrast": CONTRAST,
        "total_cycles": 16, "pre_roll_cycles": 8, "measurement_cycles": 8,
        "total_bins": 600, "pre_roll_bins": [0, 300], "measurement_bins": [300, 600],
        "time_definition": "t_n=n/150 seconds; X_n,p=0.5*q_p*sin(2*pi*4*t_n); no endpoint duplication or rounded period",
        "spatial_definition": "vertical square bars sampled on existing pixel columns j=0..16; b_j=+1 if floor((j-phase*w)/w) even else -1; half-open edges",
        "balance_definition": "q=(b-mean_over_289_pixels(b))/max_abs(b-mean(b)); no clipping, added aperture, padding, support masking or response-dependent positioning",
        "contrast_semantics": "0.5 is peak absolute contrast about the global zero-Weber mean, not RMS or Michelson contrast; finite-grid balancing produces levels +/-1 and -/+8/9",
        "bar_widths_pixels": [1, 2, 4], "phase_bar_fractions": [0, 0.5],
        "phase_origin": "leftmost input pixel center; identical spatial array for both cells, not centered by response",
        "sampling_limit": "w=1 is at spatial Nyquist; the two rasterized phases are exact sign inverses, not independent spatial samples",
        "uniform_control": "q=ones(289); identical 0.5 contrast, 4 Hz waveform, full FOV and duration; temporal mean zero",
        "cases": cases, "trace_fields": list(FIELDS), "trace_schema_version": 2,
        "Fourier": "C_h,p=(2/300)*sum_n[(Y_n,p-mean_n Y_n,p)*exp(-i*2*pi*h*4*t_n)], n=300..599, h=1,2",
        "component_amplitude": "F_h,p=abs(C_h,p), only AFTER Fourier projection of the signed raw component",
        "primary_reduction": "A_h=sqrt(mean_p(abs(C_h,p)^2)); primary=A_2(grating)/A_1(matched uniform), same cell/trace/component group",
        "groups": "H1: all 289 pixels; BC/AC: sustained/transient or local/transient separately over N,K; effective E/I and gE/gI: N,K; V1/V2/membrane/adaptation/logit/probability: N",
        "raw_reporting": "per-component complex coefficients, amplitudes, matching component uniform denominator; plus separately labeled aggregate RMS rows",
        "roundoff_reference": "component floor=4096*float64_eps*max_abs_raw_trace_in_measurement; group floor=RMS(component floors); a conservative numerical reference, not biological effect threshold or statistical significance",
        "normalization_guard": "uniform F1 must exceed its own roundoff reference; otherwise ratio=null and UNRESOLVED_DENOMINATOR",
        "finite_window_limit": "8-cycle pre-roll is fixed, not an automatic guarantee of steady state; finite-window transient leakage cannot establish outer-retinal nonlinearity",
        "verdicts": {
            "OUTER_RETINA_F2_PRESENT": "resolved H1-like F2 attributable to nonlinear processing, rather than finite-window linear transient or input/projection contamination",
            "F2_FIRST_APPEARS_AT_BC": "H1-like F2 remains within numerical floor; BC F2 clearly exceeds floor with valid uniform F1",
            "MIXED": "inconsistent localization or unresolved temporal/numerical/normalization confound",
        },
        "external_context": {
            "Raval_2026": {"title": "Origin and functional impact of early nonlinearities in primate retina",
                            "doi": "10.64898/2026.03.19.713068", "status": "preprint v1, 2026-03-23",
                            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13041967/",
                            "eccentricity_deg": [20, 50], "comparison": "candidate qualitative mismatch only; not protocol replication"},
            "Crook_2008": {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2778053/", "relevance": "primate parasol F2 observable; 4 Hz and 50% contrast described"},
            "Yu_2022": {"url": "https://elifesciences.org/articles/70611/figures", "relevance": "grating F2 normalized to uniform spot F1"},
            "catalog_eccentricity_deg": {"67#6": 6.72, "67#7": 7.36},
        },
        "prohibited": ["training", "model/checkpoint changes", "FOV/support/preprocessing changes", "other cells/seeds",
                       "DoG/radius fitting", "pathway clamps", "natural movie/spike targets", "contrast/width/phase selection by response"],
    }
    OUT.mkdir(parents=True)
    save_json(OUT / "protocol.json", protocol)
    protocol_hash = sha(OUT / "protocol.json")
    save_npz(OUT / "stimuli.npz", {"stimulus_weber": stimulus, "spatial_patterns": spatial, "temporal_sine": temporal,
                                  "time_s": time_s, "cases_json": np.array(json.dumps(cases)), "protocol_sha256": np.array(protocol_hash)})
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    archives = {"protocol_sha256": np.array(protocol_hash), "stimuli_sha256": np.array(sha(OUT / "stimuli.npz")),
                "cases_json": np.array(json.dumps(cases)), "time_s": time_s}
    all_rows, cell_summaries = [], []
    for entry in entries:
        model, cp = load_model(entry)
        cell, key = entry["cell"], entry["cell"].replace("#", "_")
        xy = cp["cone_positions_degs"].numpy().reshape(17, 17, 2)
        assert np.all(np.diff(xy[0, :, 0]) > 0) and np.all(xy[:, :, 0] == xy[0:1, :, 0])
        assert np.all(xy[:, :, 1] == xy[:, 0:1, 1])
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.inference_mode():
            x = torch.as_tensor(stimulus, device="cuda", dtype=torch.float64)
            y = torch.zeros((7, 600, 1), device="cuda", dtype=torch.float64)
            trace = observe_mechanism(model, x, observed_counts=y,
                                      intervention=InterventionSpec(Intervention.FIX_HISTORY_ZERO))
        assert all(bool(torch.isfinite(value).all()) for value in trace.tensors.values())
        assert torch.count_nonzero(trace.tensors["history_state"]) == 0
        assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
        assert sha(ROOT / entry["path"]) == entry["sha256"]
        aggregates = []
        for field in FIELDS:
            values = trace.tensors[field].cpu().numpy()
            coefficients = project(values, time_s)
            floors = FLOOR_MULTIPLIER * np.finfo(np.float64).eps * np.max(abs(values[:, MEASURE]), axis=1)
            archives[f"{key}__{field}"] = values
            archives[f"{key}__C1_C2__{field}"] = coefficients
            for group, coefficient in groups(field, coefficients):
                component_floors = dict(groups(field, floors))[group]
                shape = coefficient.shape[2:]
                c = coefficient.reshape(7, 2, -1)
                floors_flat = component_floors.reshape(7, -1)
                denominator = abs(c[-1, 0])
                denom_group = rms(c[-1, 0])
                denom_floor = rms(floors_flat[-1])
                for index, case in enumerate(cases):
                    common = {"cell": cell, "seed": SEED, **case, "trace": field, "group": group,
                              "unit": CANONICAL_TRACE_VARIABLE_SPECS[field].normalization_unit}
                    for component in range(c.shape[-1]):
                        f1, f2 = abs(c[index, :, component])
                        valid = bool(denominator[component] > floors_flat[-1, component])
                        all_rows.append({**common, "reduction": "raw_component", "component": str(np.unravel_index(component, shape)),
                                         "n_components": 1, "C1_real": float(c[index, 0, component].real),
                                         "C1_imag": float(c[index, 0, component].imag), "C2_real": float(c[index, 1, component].real),
                                         "C2_imag": float(c[index, 1, component].imag), "F1": float(f1), "F2": float(f2),
                                         "uniform_F1": float(denominator[component]),
                                         "F2_over_uniform_F1": float(f2 / denominator[component]) if valid else None,
                                         "roundoff_reference": float(floors_flat[index, component]),
                                         "uniform_roundoff_reference": float(floors_flat[-1, component]),
                                         "F2_above_roundoff": bool(f2 > floors_flat[index, component]),
                                         "normalization_status": "VALID" if valid else "UNRESOLVED_DENOMINATOR"})
                    f1, f2 = rms(c[index, 0]), rms(c[index, 1])
                    floor = rms(floors_flat[index])
                    valid = denom_group > denom_floor
                    row = {**common, "reduction": "RMS_after_component_Fourier", "component": "ALL_IN_GROUP",
                           "n_components": c.shape[-1], "C1_real": None, "C1_imag": None, "C2_real": None, "C2_imag": None,
                           "F1": f1, "F2": f2, "uniform_F1": denom_group,
                           "F2_over_uniform_F1": f2 / denom_group if valid else None,
                           "roundoff_reference": floor, "uniform_roundoff_reference": denom_floor,
                           "F2_above_roundoff": f2 > floor, "normalization_status": "VALID" if valid else "UNRESOLVED_DENOMINATOR"}
                    aggregates.append(row)
                    all_rows.append(row)
        h1 = [row for row in aggregates if row["kind"] == "grating" and row["trace"].startswith("h1_")]
        bc = [row for row in aggregates if row["kind"] == "grating" and row["trace"].startswith("bc_")]
        first_at_bc = all(not row["F2_above_roundoff"] for row in h1) and all(row["F2_above_roundoff"] and row["normalization_status"] == "VALID" for row in bc)
        cell_summaries.append({"cell": cell, "seed": SEED, "polarity": cp["polarities"], "cell_type": cp["cell_types"],
                               "alpha": float(model.alpha), "h1_tau_ms": float(model.h1.tau_ms),
                               "h1_delay_ms": float(model.h1.delay_ms),
                               "inference_state_unchanged": True, "checkpoint_sha256_unchanged": True,
                               "all_forwards_finite": True, "history_state_zero": True,
                               "verdict": "F2_FIRST_APPEARS_AT_BC" if first_at_bc else "MIXED", "aggregates": aggregates})
        archives[f"{key}__cone_positions_degs"] = cp["cone_positions_degs"].numpy()
        archives[f"{key}__trace_metadata_json"] = np.array(json.dumps(trace.metadata(), ensure_ascii=False))
        print(f"{cell}: seven frozen sequences complete; {cell_summaries[-1]['verdict']}", flush=True)
        del model, trace, before
    save_npz(OUT / "traces.npz", archives)
    with (OUT / "harmonics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    verdict = "F2_FIRST_APPEARS_AT_BC" if all(c["verdict"] == "F2_FIRST_APPEARS_AT_BC" for c in cell_summaries) else "MIXED"
    summary = {"verdict": verdict, "protocol_sha256": protocol_hash, "cells": cell_summaries,
               "harmonic_fixture": fixture, "physical_range_weber": [float(stimulus.min()), float(stimulus.max())],
               "grating_max_abs_spatial_mean": float(np.max(abs(stimulus[:6].mean(-1)))),
               "max_abs_measurement_temporal_mean": float(np.max(abs(stimulus[:, MEASURE].mean(1)))),
               "stimulus_F2_RMS_by_case": [rms(c[1]) for c in project(stimulus, time_s)],
               "source_files_unchanged": all(sha(ROOT / name) == value for name, value in source_hashes.items()),
               "manifest_unchanged": sha(MANIFEST) == protocol["manifest_sha256"],
               "protocol_unchanged": sha(OUT / "protocol.json") == protocol_hash,
               "checkpoint_load_count": 2, "forward_sequence_count": 14, "training_updates": 0,
               "csv_rows": len(all_rows), "elapsed_seconds": time.monotonic() - start,
               "artifact_sha256": {name: sha(OUT / name) for name in ("protocol.json", "stimuli.npz", "traces.npz", "harmonics.csv")},
               "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name()},
               "interpretation_limit": "Only localizes the nonlinear response stage in this model; no biological-contribution ranking or physiological validation."}
    save_json(OUT / "summary.json", summary)
    print(f"Saved all five outputs; {verdict}; elapsed {summary['elapsed_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    run()
