"""R0: frozen real train data, default parameters, forward assertions only."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig, _LIVE_START_FRAME
from data.schottdorf_lee_catalog import mc_pc_recordings
from experiments.retipath_multiscale_v0.circuit import area_integral_weights
from experiments.retipath_population_v0_1.circuit import PopulationRetipath
from experiments.retipath_population_v0_1.real_data import (
    history_events, load_recording_train, load_training_movie,
    physical_stimulus, recorded_probability, recording_mapping,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/real_data/schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829/results.json"
REPOSITORY = ROOT / "data/real/schottdorf_lee_2021_repository"
MOVIE = ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"
OUT = ROOT / "output/real_data/retipath_population_r0_preflight"
ATOL = 1e-10
Q_ATOL = 2e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(model: PopulationRetipath) -> str:
    return hashlib.sha256(json.dumps({k: tensor_hash(v) for k, v in model.state_dict().items()}, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def error(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    return float((a - b).abs().max())


def trace_finite(trace) -> bool:
    return all(bool(torch.isfinite(value).all()) for group in (trace.inputs, trace.states, trace.outputs, trace.initial_state) for value in group.values())


def causal_checks(model, stimulus, events, mapping) -> dict:
    base = model(stimulus, observed_events=events)
    cut = stimulus.values.shape[1] // 2
    changed_x = stimulus.values.clone()
    changed_x[:, cut + 1:] = changed_x[:, cut + 1:].flip(1)
    future = model(replace(stimulus, values=changed_x), observed_events=events)
    input_error = max(error(values[:, :cut + 1], getattr(future, group)[name][:, :cut + 1])
                      for group in ("inputs", "states", "outputs")
                      for name, values in getattr(base, group).items() if values.ndim >= 3)
    changed_events = events.clone()
    changed_events[:, cut:, mapping.output_index] = 1 - changed_events[:, cut:, mapping.output_index]
    event_change = model(stimulus, observed_events=changed_events)
    event_error = error(base.outputs["logit"][:, :cut + 1], event_change.outputs["logit"][:, :cut + 1])
    past_event_effect = error(base.outputs["logit"][:, cut + 1:], event_change.outputs["logit"][:, cut + 1:])
    irrelevant_events = events.clone()
    irrelevant_events[..., 1 - mapping.output_index] = 1 - irrelevant_events[..., 1 - mapping.output_index]
    irrelevant = model(stimulus, observed_events=irrelevant_events)
    irrelevant_error = error(recorded_probability(base, mapping), recorded_probability(irrelevant, mapping))
    repeated = model(stimulus, observed_events=events)
    reset_error = error(base.outputs["logit"], repeated.outputs["logit"])
    prefix_stimulus = replace(stimulus, values=stimulus.values[:, :cut + 1],
                              time_ms=stimulus.time_ms[:cut + 1], input_valid=stimulus.input_valid[:, :cut + 1])
    prefix = model(prefix_stimulus, observed_events=events[:, :cut + 1])
    prefix_error = error(base.outputs["logit"][:, :cut + 1], prefix.outputs["logit"])
    single_stimulus = replace(stimulus, values=stimulus.values[:1], input_valid=stimulus.input_valid[:1])
    single = model(single_stimulus, observed_events=events[:1])
    batch_error = error(base.outputs["logit"][:1], single.outputs["logit"])
    initial_ok = all(torch.equal(value, torch.full_like(value, 2 / 9 if name == "V" else 0))
                     for name, value in base.initial_state.items())
    history_zero = bool((base.states["history"][:, 0] == 0).all())
    try:
        model(stimulus, observed_events=events, reset="carry")
    except ValueError:
        carry_rejected = True
    else:
        carry_rejected = False
    result = {
        "cut_bin": cut, "future_stimulus_prefix_max_abs": input_error,
        "current_and_future_event_logit_prefix_max_abs": event_error,
        "strictly_past_event_later_effect_max_abs": past_event_effect,
        "unobserved_output_history_selected_probability_max_abs": irrelevant_error,
        "reset_replay_max_abs": reset_error, "truncated_prefix_max_abs": prefix_error,
        "batch_isolation_max_abs": batch_error, "initial_states_match_baseline": initial_ok,
        "history_at_bin_zero_is_zero": history_zero, "carry_rejected": carry_rejected,
        "atol": ATOL,
    }
    result["passed"] = max(input_error, event_error, irrelevant_error, reset_error, prefix_error, batch_error) <= ATOL and past_event_effect > 0 and initial_ok and history_zero and carry_rejected
    assert result["passed"], result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight artifacts: {OUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    config = SchottdorfAdapterConfig(**source["adapter_config"])
    assert asdict(config) == dict(train_sequence_count=16, validation_sequence_count=4,
                                 sequence_steps=150, warmup_steps=30, crop_pixels=51, pool_factor=3)
    assert (source["cell_count"], source["recording_count"], _LIVE_START_FRAME) == (22, 37, 751)
    recordings = mc_pc_recordings(REPOSITORY / "data")
    expected_cells = {c["cell_id"]: c for c in source["cells"]}
    assert {r.cell_id for r in recordings} == set(expected_cells) and len(recordings) == 37
    for cid, cell in expected_cells.items():
        selected = tuple(r for r in recordings if r.cell_id == cid)
        assert tuple(r.recording_id for r in selected) == tuple(cell["recording_ids"])
        assert all((r.retinal_class, r.canonical_cell_type, r.polarity) ==
                   (cell["retinal_class"], cell["canonical_cell_type"], cell["polarity"]) for r in selected)
    code_paths = [
        "AGENTS.md", "docs/RETIPATH_POPULATION_ARCHITECTURE_V0_1.md",
        "docs/RETIPATH_REAL_DATA_INTEGRATION_V0.md", "docs/NEXT_TASK.md",
        "data/schottdorf_lee_2021.py", "data/schottdorf_lee_multirecording.py",
        "data/schottdorf_lee_catalog.py", "data/schottdorf_lee_spikes.py", "data/retinal_recording.py",
        "experiments/retipath_multiscale_v0/circuit.py", "experiments/retipath_multiscale_v0/contracts.py",
        "experiments/retipath_population_v0_1/circuit.py",
        "experiments/retipath_population_v0_1/real_data.py",
        "experiments/retipath_population_v0_1/run_real_r0_preflight.py",
        "models/mechanistic_retina/state.py", "models/mechanistic_retina/retipath_spatial_ei.py",
    ]
    # Freeze selection and intended accesses before opening any movie or spike payload.
    OUT.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema": "retipath_population_real_r0_v1", "phase": "R0_forward_only",
        "frozen_before_payload": datetime.now(timezone.utc).isoformat(),
        "source_cohort_manifest": str(SOURCE.relative_to(ROOT)),
        "adapter_config_unchanged": asdict(config), "source_native_grid": [256, 256],
        "source_full_fov_deg": [4.6, 4.6], "native_deg_per_pixel": 4.6 / 256,
        "source_crop_native_pixels": [51, 51], "source_pooled_grid": [17, 17],
        "pooled_deg_per_pixel": 3 * 4.6 / 256, "crop_fov_deg": [51 * 4.6 / 256] * 2,
        "axis_convention": "legacy row-major centers: x increases by column, y decreases by row; source cell origin=(0,0)",
        "calibration": {
            "source": "data.schottdorf_lee_2021._pooled_lm_signal/_load_calibrated_lm_drive",
            "rgb_divisor": 256.0,
            "gamma": {"R": [.01451, .9855, 2.3122], "G": [.005123, .9949, 2.2752], "B": [.02612, .9739, 2.2818]},
            "L_weights": [2.74, 3.4, 1.34], "M_weights": [1.06, 3.58, 2.07], "M_scale": 1.21,
            "pooling": "mean of each 3x3 native L+M block; unchanged float32 preprocessing",
            "background": "per-pooled-pixel mean of decoded frames [0,751), exactly existing loader",
            "weber": "(live L+M - background L+M) / background L+M",
            "units": "dimensionless relative L+M Weber drive; absolute R*/cone/s unknown",
        },
        "time": {
            "fps": 150, "dt_ms": 1000 / 150, "spike_resolution_ms": .1,
            "live_to_decoded_frame_offset": 751, "absolute_frame_zero_status": "UNRESOLVED_EXTERNAL_EVIDENCE_REQUIRED",
            "reinterpret_alignment": False, "train_live_frames_half_open": [0, 2400],
            "train_seconds_half_open": [0, 16], "validation_frames_not_materialized": [2400, 3000],
            "sequence_relative_midpoints_ms": "(bin+0.5)*1000/150; source segment offset stored separately",
            "spike_binning": "floor(corrected_live_relative_time_ms * 150/1000); count>0 occupancy",
            "subtract_video_start_again": False,
            "repeated_trials": "existing parser: first six columns, each live-relative; seventh maintained-activity column excluded",
            "repeat_movie": "existing pipeline uses first minute of 1x10 movie; no new clock or continuous inter-repeat timeline",
        },
        "history": {
            "sequence_bins": 150, "warmup_bins": 30, "scored_bins_per_sequence": 120,
            "external_preroll_bins": 0, "prepend_blank_to_forward": False,
            "reset": "independent baseline reset for every one-second sequence, no hidden carry",
            "physiological_initial_state_known": False,
            "warmup_is_exact_IIR_history": False,
            "history_events": "recorded occupancy only; fixed strictly-past filter; other output history is masked placeholder",
        },
        "spatial": {
            "Q": "unchanged full-aperture area intersection; no resizing or degree rescaling",
            "pixel_edges": "shared edges (pooled_edge_index*3-51/2)*4.6/256; x ascending, y descending",
            "legacy_center_agreement": "checked within float32 storage precision; no gap-prone independent center plus/minus halfwidth construction",
            "node_aperture_side_deg": .1, "template_ancestor_bounding_box_deg": [[-.35, .35], [-.35, .35]],
            "coverage": "numerical coverage checked against all structural ancestors, not physiological RF registration",
            "lost_pooling_information_reconstructed": False,
            "physical_origin": "reuse existing canonical single-cell origin, not new measured cross-cell geometry",
        },
        "instances": [asdict(recording_mapping(r)) for r in recordings],
        "sharing": "one separate model per recording; same initialization values, no shared parameter objects; architecture/type rules only",
        "families": {"H1": 25, "BC_ON": 25, "BC_OFF": 25, "AC_local_ON": 9, "AC_local_OFF": 9,
                     "AC_broad_ON": 9, "AC_broad_OFF": 9, "RGC_ON_OFF": 2, "effective_modes_per_RGC": 2},
        "PC_scope": "polarity-only structural forward; no midget family or chromatic prediction claim; not a compatible biological training port",
        "observed_port": "recorded single RGC spike only; no H1/BC/AC/EI real losses",
        "model": {"initialization": "unaltered PopulationRetipath constructor", "BC_output": "LegacyPReLU", "dtype": "float64", "device": "cpu", "parameters_require_grad": False},
        "checks": {"full_forward": "all 16 existing train segments of all retained trials in all 37 recordings",
                   "causality_reset_subset": "first up to four train sequences per recording, fixed before access",
                   "batch_size": 4, "causality_atol": ATOL, "Q_coverage_atol": Q_ATOL},
        "access_boundary": "legacy spike parser reads complete files including out-of-window timestamps; only [0,16s) is binned/materialized/forwarded; no validation/test metrics or parameter choices",
        "training_steps": 0, "optimizer_created": False, "checkpoint_loaded_or_written": False,
    }
    write_json(OUT / "dataset_contract.json", contract)
    tracked = {str(SOURCE.relative_to(ROOT)): sha256(SOURCE)}
    tracked.update({name: sha256(ROOT / name) for name in code_paths})
    expected_sources = {MOVIE: source["source_sha256"][MOVIE.name]}
    for name in ("README.md", "data/Cell List.docx", "data/CellsList.docx", "stimuli/1x10_256.mpg"):
        expected_sources[REPOSITORY / name] = source["source_sha256"][name]
    for recording in recordings:
        expected_sources[recording.path] = expected_cells[recording.cell_id]["source_sha256"][recording.path.name]
    for path, expected in expected_sources.items():
        actual = sha256(path)
        assert actual == expected, f"Source lineage mismatch: {path}"
        tracked[str(path.relative_to(ROOT))] = actual
    write_json(OUT / "manifest.json", {
        "schema": "retipath_population_r0_source_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT), "dataset_contract_sha256": sha256(OUT / "dataset_contract.json"),
        "source_sha256": tracked, "all_legacy_asset_hashes_match": True,
        "cell_count": 22, "recording_count": 37, "source_files_replaced": False,
        "old_checkpoints_accessed": False, "test_scientific_evaluation_consumed": False,
    })
    torch.set_num_threads(2)
    movie = load_training_movie(MOVIE, config)
    assert movie.sequences.shape == (16, 150, 289) and np.isfinite(movie.sequences).all()
    records = []
    models = []
    all_parameter_objects = set()
    with torch.inference_mode():
        for recording in recordings:
            mapping = recording_mapping(recording)
            split, parsed = load_recording_train(recording, movie, config)
            model = PopulationRetipath().requires_grad_(False).eval()
            objects = {id(p) for p in model.parameters()}
            assert not objects & all_parameter_objects
            all_parameter_objects.update(objects)
            models.append(model)
            initial_hash = state_hash(model)
            assert torch.equal(model.rgc_polarity, torch.tensor([1., -1.], dtype=torch.float64))
            n, steps, pixels = split.cone_drive.shape
            assert (steps, pixels) == (150, 289) and n == 16 * parsed["biological_trials"]
            assert split.spike_counts.shape == split.spike_events.shape == split.valid_mask.shape == (n, 150, 1)
            assert torch.equal(split.spike_events, (split.spike_counts > 0).float())
            assert bool((split.spike_counts >= 0).all()) and bool(torch.isfinite(split.cone_drive).all())
            assert not bool(split.valid_mask[:, :30].any()) and bool(split.valid_mask[:, 30:].all())
            expected_ids = tuple(f"{recording.recording_id}-live-frames-{segment*150:06d}-{(segment+1)*150-1:06d}-trial-{trial+1}"
                                 for segment in range(16) for trial in range(parsed["biological_trials"]))
            assert split.source_image_ids == expected_ids
            minimum_probability, maximum_probability, q_error = 1., 0., 0.
            probabilities_digest = hashlib.sha256()
            for start in range(0, n, 4):
                stimulus = physical_stimulus(split.cone_drive[start:start+4], movie.cone_positions_degs, config)
                events = history_events(split.spike_events[start:start+4], mapping, dtype=torch.float64)
                trace = model(stimulus, observed_events=events)
                assert trace_finite(trace)
                output = recorded_probability(trace, mapping)
                assert output.shape == split.spike_events[start:start+4].shape
                assert trace.inputs["q"].shape == (*output.shape[:2], 25)
                assert trace.outputs["probability"].shape == (*output.shape[:2], 2)
                assert bool(((output >= 0) & (output <= 1)).all())
                assert bool((trace.outputs["gE"] > 0).all() & (trace.outputs["gI"] > 0).all())
                weights = area_integral_weights(stimulus.pixel_bounds, model.input_xy)
                row_error = float((weights.sum(-1) - 1).abs().max())
                assert row_error <= Q_ATOL and bool((weights >= 0).all())
                # Independent numpy reduction verifies the physical Q output, including flatten order.
                bounds, xy = stimulus.pixel_bounds.numpy(), model.input_xy.numpy()
                overlap = np.maximum(0, np.minimum(bounds[None, ..., 1], xy[:, None] + .05)
                                     - np.maximum(bounds[None, ..., 0], xy[:, None] - .05))
                q_numpy = stimulus.values.numpy() @ (np.prod(overlap, axis=-1) / .01).T
                q_error = max(q_error, float(np.max(np.abs(q_numpy - trace.inputs["q"].numpy()))))
                assert q_error <= Q_ATOL
                minimum_probability = min(minimum_probability, float(output.min()))
                maximum_probability = max(maximum_probability, float(output.max()))
                probabilities_digest.update(output.contiguous().numpy().tobytes())
            first_stimulus = physical_stimulus(split.cone_drive[:4], movie.cone_positions_degs, config)
            first_events = history_events(split.spike_events[:4], mapping, dtype=torch.float64)
            checks = causal_checks(model, first_stimulus, first_events, mapping)
            assert state_hash(model) == initial_hash and all(p.grad is None and not p.requires_grad for p in model.parameters())
            support = model.pathway_support()
            bounds = first_stimulus.pixel_bounds
            source_extent = torch.stack((bounds[..., 0].min(0).values, bounds[..., 1].max(0).values), -1)
            masks = {name: int(mask.reshape(-1, 25).any(0).sum()) for name, mask in support.items()}
            records.append({
                **asdict(mapping), "recording_kind": recording.recording_kind.value, "parsed_metadata": parsed,
                "input_shape": [n, 150, 289], "Q_shape": [n, 150, 25], "target_shape": [n, 150, 1],
                "recorded_probability_shape": [n, 150, 1], "full_probability_shape": [n, 150, 2],
                "source_ids": list(split.source_image_ids), "train_sequences": n, "time_bins_per_sequence": 150,
                "forward_time_bins": n * 150, "warmup_bins_per_sequence": 30, "preroll_bins": 0,
                "valid_target_bins": int(split.valid_mask.sum()), "target_counts_sha256": tensor_hash(split.spike_counts),
                "input_tensor_sha256": tensor_hash(split.cone_drive), "probability_sha256": probabilities_digest.hexdigest(),
                "forward_finite": True, "probability_range": [minimum_probability, maximum_probability],
                "physical_fov_deg": [51 * 4.6 / 256] * 2, "source_extent_deg": source_extent.tolist(),
                "support_apertures_by_port": masks, "coverage": "FULL_TEMPLATE_ANCESTOR_SUPPORT",
                "Q_row_sum_max_abs_error": row_error, "Q_numpy_max_abs_error": q_error,
                "checks": checks, "state_hash_before_after": initial_hash, "parameters_unchanged": True,
                "alignment": "LEGACY_FRAME_751_PRESERVED_ABSOLUTE_ALIGNMENT_UNVERIFIED",
                "readiness_blockers": ["ABSOLUTE_FRAME_ZERO_UNRESOLVED", "BASELINE_RESET_NOT_MEASURED_PHYSIOLOGICAL_INITIAL_STATE"]
                                      + ([mapping.blocker] if mapping.blocker else []),
                "internal_real_supervised_ports": [], "spike_loss_implemented": False,
            })
            print(f"PASS {len(records):02d}/37 {recording.cell_id} {recording.recording_id} {mapping.recorded_type} {n} sequences", flush=True)
    per_cell = []
    for cid, expected in expected_cells.items():
        subset = [r for r in records if r["cell_id"] == cid]
        total = sum(r["train_sequences"] for r in subset)
        assert total == expected["train_sequences"]
        assert sum(r["valid_target_bins"] for r in subset) == expected["train_valid_bins"]
        per_cell.append({
            "cell_id": cid, "type": f"{expected['retinal_class']} {expected['polarity']}",
            "recordings": ";".join(r["recording_id"] for r in subset), "recording_count": len(subset),
            "independent_instance_count": len(subset), "input_shape": f"[{total},150,289]",
            "Q_shape": f"[{total},150,25]", "FOV_deg": "0.91640625 x 0.91640625",
            "pooled_degree_per_pixel": 3 * 4.6 / 256, "coverage": "FULL_TEMPLATE_ANCESTOR_SUPPORT",
            "unique_train_time_bins": 2400, "sequence_time_bins": 150, "total_forward_time_bins": total * 150,
            "warmup_bins_per_sequence": 30, "external_preroll_bins": 0,
            "target_shape": f"[{total},150,1]", "probability_shape": f"[{total},150,1]",
            "full_probability_shape": f"[{total},150,2]", "forward_finite": True,
            "RGC_port": subset[0]["population_port"], "type_compatible": subset[0]["type_compatible"],
            "causality_reset_verified": all(r["checks"]["passed"] for r in subset),
            "parameters_unchanged": True,
            "blocking_items": ";".join(subset[0]["readiness_blockers"]),
        })
    with (OUT / "per_cell_preflight.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_cell[0]))
        writer.writeheader()
        writer.writerows(per_cell)
    unchanged = {name: sha256(ROOT / name) == digest for name, digest in tracked.items()}
    assert all(unchanged.values()), "source changed during preflight"
    result = {
        "status": "VERIFIED_ENGINEERING_PREFLIGHT_WITH_DECLARED_SCIENTIFIC_BLOCKERS",
        "cell_count": len(per_cell), "recording_count": len(records),
        "forward_sequence_count": sum(r["train_sequences"] for r in records),
        "biological_trial_count": sum(r["parsed_metadata"]["biological_trials"] for r in records),
        "all_forward_finite": True, "all_causality_reset_checks_passed": True,
        "all_sources_unchanged": all(unchanged.values()), "all_legacy_input_asset_hashes_match": True,
        "all_model_parameters_unchanged": True, "all_model_instances_parameter_disjoint": True,
        "MC_native_family_cell_count": sum(r["type_compatible"] for r in per_cell),
        "PC_structural_proxy_cell_count": sum(not r["type_compatible"] for r in per_cell),
        "input_movie_live_frames_decoded": [0, 2400], "validation_test_tensors_materialized": False,
        "legacy_parser_whole_file_access": True, "validation_test_forward_or_metric": False,
        "source_metadata_only_validation_split_definition_read": True,
        "training_steps": 0, "backward_calls": 0, "optimizer_steps": 0, "loss_computed": False,
        "old_checkpoint_accessed": False, "new_checkpoint_written": False,
        "all_ready_for_real_training": False,
        "reporting_scope": "engineering adapter and default forward only; no fit, prediction score, physiological validation or new alignment decision",
        "source_unchanged_checks": unchanged, "recordings": records,
        "output_sha256": {name: sha256(OUT / name) for name in ("manifest.json", "dataset_contract.json", "per_cell_preflight.csv")},
    }
    write_json(OUT / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"recordings", "source_unchanged_checks", "output_sha256"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
