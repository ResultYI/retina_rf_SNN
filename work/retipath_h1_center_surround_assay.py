from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/evaluations/retipath_h1_center_surround_assay_20260915"
TEMP = Path("C:/Users/win11-pc/AppData/Local/Temp/retipath-h1-assay-20260915")
sys.path.insert(0, str(ROOT))
from evaluation.mechanistic_retina.mechanism_observation import Intervention, InterventionSpec, observe_mechanism
from models.mechanistic_retina.contracts import PathwayClamp

FIELDS = (
    "h1_graph_drive", "h1_state", "h1_feedback", "h1_modulated_input",
    "bc_direct_effective_drive", "bc_broad_effective_drive", "ac_states", "ac_inhibitory_drive",
    "normalized_drive_E", "normalized_drive_I", "effective_drive_E", "effective_drive_I",
    "gE", "gI", "V1", "V2", "membrane_readout", "adaptation_state", "logit", "probability",
)
PRIMARY = ("effective_drive_E", "effective_drive_I", "membrane_readout", "logit")
CONDITIONS = ("NORMAL", "BLOCK_H1_FEEDBACK")
MEASURES = ("R_center_NORMAL", "R_center_BLOCK", "R_large_NORMAL", "R_large_BLOCK", "R_surround_NORMAL", "R_surround_BLOCK",
            "S_NORMAL", "S_BLOCK", "Delta_S_H1", "Delta_center", "Delta_large",
            "S_large_NORMAL", "S_large_BLOCK", "Delta_S_large_H1", "gain_only_residual")
REPRESENTATIVES = ("69#4", "67#6", "68#4", "67#4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def load_definition():
    definition = json.loads((OUT / "stimulus_definition.json").read_text(encoding="utf-8"))
    assert sha(OUT / "PROTOCOL.md") == definition["protocol_sha256"]
    assert sha(definition["manifest_path"]) == definition["manifest_sha256"]
    assert definition["locked_before_any_assay_response"]
    return definition


def loader():
    path = ROOT / "output/evaluations/retipath_gain_canonicalization_20260915/checkpoint_migration.py"
    spec = importlib.util.spec_from_file_location("canonical_checkpoint_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_canonical_checkpoint


def stimuli(cell, definition):
    cases = [{"name": "BACKGROUND", "sign": 0, "radius_deg": 0.0, "pixel_count": 0}]
    values = [np.zeros((150, 289), dtype=np.float32)]
    for sign in (1, -1):
        for mask in cell["masks"]:
            x = np.zeros((150, 289), dtype=np.float32)
            x[30:90, mask["indices"]] = sign * definition["C"]
            cases.append({"name": mask["name"], "sign": sign, "radius_deg": mask["radius_deg"], "pixel_count": mask["pixel_count"]})
            values.append(x)
    return torch.from_numpy(np.stack(values)), cases


def scalar_trace(array, field, center_indices):
    if field.startswith("h1_"):
        return array[..., center_indices].mean(-1, dtype=np.float64)
    if field in ("bc_direct_effective_drive", "bc_broad_effective_drive", "ac_states", "ac_inhibitory_drive"):
        array = array.sum(-1, dtype=np.float64)
    return array.reshape(*array.shape[:2], -1).mean(-1, dtype=np.float64)


def window_response(response, window, direction):
    left, right = window
    if (left, right) == (30, 60):
        index = left + int(np.argmax(direction * response[left:right]))
        return float(response[index]), index
    return float(response[left:right].mean()), None


def run():
    if (OUT / "correctness.json").exists():
        raise FileExistsError("Assay already completed; refusing to rerun responses")
    definition = load_definition()
    load_model = loader()
    torch.set_num_threads(1)
    trace_dir = OUT / "figures/trace_data"
    trace_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [ROOT / "models/mechanistic_retina/retipath_canonical_gain.py",
                    ROOT / "models/mechanistic_retina/retipath_spatial_ei.py",
                    ROOT / "models/mechanistic_retina/h1_pathway.py",
                    ROOT / "evaluation/mechanistic_retina/mechanism_observation.py", Path(__file__)]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    started = datetime.now(timezone.utc).isoformat()
    checks, trace_rows = [], []
    start = time.monotonic()
    for cell in definition["cells"]:
        x, cases = stimuli(cell, definition)
        y = torch.zeros(x.shape[0], 150, 1)
        center_indices = next(m["indices"] for m in cell["masks"] if m["name"] == "CENTER")
        polarity_sign = 1 if cell["polarity"] == "ON" else -1
        for checkpoint in cell["checkpoints"]:
            assert sha(ROOT / checkpoint["path"]) == checkpoint["sha256"]
            model, _ = load_model(ROOT / checkpoint["path"])
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with torch.no_grad():
                normal = observe_mechanism(model, x, observed_counts=y)
                blocked = observe_mechanism(model, x, observed_counts=y,
                                            intervention=InterventionSpec(Intervention.BLOCK_H1_FEEDBACK))
            for field in ("h1_graph_drive", "h1_state", "h1_graph_edge_index", "h1_graph_edge_weight"):
                assert torch.equal(normal.tensors[field], blocked.tensors[field]), field
            assert torch.count_nonzero(blocked.tensors["h1_feedback"]) == 0
            assert torch.equal(blocked.tensors["h1_modulated_input"], x)
            for trace in (normal, blocked):
                assert all(torch.isfinite(value).all() for value in trace.tensors.values())
                assert torch.count_nonzero(trace.tensors["history_state"]) == 0
                assert torch.count_nonzero(trace.tensors["history_term"]) == 0
                assert (trace.tensors["gE"] >= 0).all() and (trace.tensors["gI"] >= 0).all()
                assert ((trace.probability >= 0) & (trace.probability <= 1)).all()
            archive = {field: np.stack((normal.tensors[field].numpy(), blocked.tensors[field].numpy())) for field in FIELDS}
            name = f"{cell['cell'].replace('#', '_')}_{checkpoint['seed']}.npz"
            path = trace_dir / name
            if path.exists():
                raise FileExistsError(path)
            np.savez_compressed(path, **archive, stimulus=x.numpy(), observed_occupancy=y.numpy(),
                                conditions=np.array(CONDITIONS), cases_json=np.array(json.dumps(cases)),
                                checkpoint_sha256=np.array(checkpoint["sha256"]),
                                history_condition=np.array("FIX_HISTORY_ZERO"))
            with np.load(path, allow_pickle=False) as saved:
                assert all(np.array_equal(saved[field], archive[field]) for field in FIELDS)
            effects = {}
            for i, case in enumerate(cases):
                effects[str(i)] = {}
                for field in FIELDS:
                    a, b = archive[field][:, i]
                    diff = np.abs(b - a).reshape(150, -1).max(-1)
                    threshold = float(8 * np.finfo(np.float32).eps * max(1.0, float(np.abs(a).max()), float(np.abs(b).max())))
                    nonzero, resolved = np.flatnonzero(diff > 0), np.flatnonzero(diff > threshold)
                    effects[str(i)][field] = {"first_nonzero_bin": int(nonzero[0]) if len(nonzero) else None,
                                             "first_resolved_bin": int(resolved[0]) if len(resolved) else None,
                                             "threshold": threshold, "max_abs_change": float(diff.max())}
            reduced = {field: np.stack([scalar_trace(archive[field][c], field, center_indices) for c in range(2)]) for field in FIELDS}
            for condition_index, condition in enumerate(CONDITIONS):
                for i, case in enumerate(cases):
                    for window_name, window in definition["windows"].items():
                        row = {"cell": cell["cell"], "seed": checkpoint["seed"], "polarity": cell["polarity"],
                               "cell_type": cell["cell_type"], "condition": condition, "stimulus": case["name"],
                               "contrast_sign": case["sign"], "contrast": case["sign"] * definition["C"], "window": window_name,
                               "history_condition": "FIX_HISTORY_ZERO", "trace_file": str(path.relative_to(OUT))}
                        peaks = {}
                        for field in FIELDS:
                            response = reduced[field][condition_index, i] - reduced[field][condition_index, 0]
                            direction = case["sign"] if field.startswith("h1_") else case["sign"] * polarity_sign
                            value, peak = window_response(response, window, direction or 1)
                            row[field] = value
                            if peak is not None:
                                peaks[field] = peak
                        row["onset_peak_bins_json"] = json.dumps(peaks)
                        row["block_effect_first_bins_json"] = json.dumps({field: effect["first_resolved_bin"] for field, effect in effects[str(i)].items()}) if condition_index else "{}"
                        trace_rows.append(row)
            extra = {}
            if cell["cell"] in REPRESENTATIVES and checkpoint["seed"] == definition["seeds"][0]:
                with torch.no_grad():
                    official = model.forward_sequence(x, observed_counts=y)
                    assert torch.equal(official.logits, normal.logits) and torch.equal(official.spike_probability, normal.probability)
                    nonzero_y = torch.zeros_like(y)
                    nonzero_y[:, 5::13] = 1
                    observed = observe_mechanism(model, x, observed_counts=nonzero_y)
                    fixed = observe_mechanism(model, x, observed_counts=nonzero_y,
                                              intervention=InterventionSpec(Intervention.FIX_HISTORY_ZERO))
                    allowed = {"history_input", "history_state", "history_term", "logit", "probability"}
                    changed = [field for field in fixed.tensors if not torch.equal(fixed.tensors[field], observed.tensors[field])]
                    assert set(changed) <= allowed
                    assert torch.equal(fixed.logits, normal.logits) and torch.equal(fixed.probability, normal.probability)
                    for field in fixed.tensors:
                        assert torch.equal(fixed.tensors[field], normal.tensors[field]), field
                    parts = model.spatial_components(x, frozenset({PathwayClamp.H1}))
                    for field, value in (("bc_direct_effective_drive", parts.direct), ("bc_broad_effective_drive", parts.broad),
                                         ("ac_states", parts.ac_states), ("effective_drive_E", parts.effective_e), ("effective_drive_I", parts.effective_i)):
                        assert torch.equal(blocked.tensors[field], value), field
                    future_x = x.clone()
                    future_x[:, 61:] = -future_x[:, 61:] + 0.25
                    for kind, original in ((Intervention.NORMAL, normal), (Intervention.BLOCK_H1_FEEDBACK, blocked)):
                        future = observe_mechanism(model, future_x, observed_counts=y, intervention=InterventionSpec(kind))
                        for field in FIELDS:
                            assert torch.equal(original.tensors[field][:, :61], future.tensors[field][:, :61]), field
                extra = {"canonical_forward_identity": True, "zero_history_equivalence": True,
                         "history_only_changed_fields": changed, "independent_downstream_recompute_identity": True,
                         "future_stimulus_invariance_both_conditions": True}
            assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
            checks.append({"cell": cell["cell"], "seed": checkpoint["seed"], "checkpoint_sha256": checkpoint["sha256"],
                           "stimulus_sequences": len(cases), "all_finite": True, "h1_graph_state_unchanged": True,
                           "feedback_strict_zero": True, "history_state_term_zero": True, "parameters_buffers_unchanged": True,
                           "trace_archive": str(path.relative_to(OUT)), "trace_sha256": sha(path),
                           "lossless_archive_roundtrip": True, "effects": effects, "extra_correctness": extra})
        print(f"Frozen assay completed {cell['cell']}: 3 seeds, both contrasts and conditions", flush=True)
    assert source_hashes == {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    assert all(sha(ROOT / ref["path"]) == ref["sha256"] for cell in definition["cells"] for ref in cell["checkpoints"])
    save_json(TEMP / "trace_rows.json", trace_rows)
    save_json(OUT / "correctness.json", {
        "status": "PASS", "protocol_sha256": sha(OUT / "PROTOCOL.md"), "stimulus_definition_sha256": sha(OUT / "stimulus_definition.json"),
        "response_execution_started_utc": started, "lock_preceded_response_execution": definition["locked_at_utc"] < started,
        "applied_contrast_float32": float(np.float32(definition["C"])), "checkpoint_count": len(checks),
        "model_condition_stimulus_sequences": sum(r["stimulus_sequences"] for r in checks) * 2,
        "source_sha256": source_hashes, "frozen_sources_checkpoints_unchanged": True, "checks": checks,
        "pathway_specs": [InterventionSpec(Intervention(c)).to_record() for c in CONDITIONS],
        "history_spec": InterventionSpec(Intervention.FIX_HISTORY_ZERO).to_record(),
        "training_updates": 0, "new_natural_movie_blocks": [], "natural_spike_target_analysis": False, "mach_sbc_runs": 0,
        "trace_schema_version": 2, "physiological_status_upgrades": 0, "device": "cpu", "dtype": "float32", "torch": torch.__version__,
        "elapsed_seconds": time.monotonic() - start,
    })


def stats(values, rng):
    values = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return {"n_cells": 0, "mean": None, "median": None, "positive": 0, "negative": 0, "zero": 0, "ci_low": None, "ci_high": None}
    index = rng.integers(0, len(values), (10000, len(values)))
    ci = np.quantile(values[index].mean(1), [.025, .975])
    return {"n_cells": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
            "ci_low": float(ci[0]), "ci_high": float(ci[1])}


def analyze():
    definition = load_definition()
    rows = json.loads((TEMP / "trace_rows.json").read_text(encoding="utf-8"))
    index = {(r["cell"], r["seed"], r["condition"], r["stimulus"], r["contrast_sign"], r["window"]): r for r in rows}
    per_cell, preferences = [], {}
    for cell in definition["cells"]:
        polarity = 1 if cell["polarity"] == "ON" else -1
        for sign in (1, -1):
            direction = sign * polarity
            candidates = ("SPOT_SMALL", "SPOT_MID", "CENTER")
            preferred = max(candidates, key=lambda spot: direction * np.mean([
                index[(cell["cell"], seed, "NORMAL", spot, sign, "late")]["logit"] for seed in definition["seeds"]]))
            preferences[(cell["cell"], sign)] = preferred
            for window in definition["windows"]:
                for variable in PRIMARY:
                    seed_records = []
                    for seed in definition["seeds"]:
                        def get(condition, stimulus):
                            return index[(cell["cell"], seed, condition, stimulus, sign, window)][variable]
                        cn, cb = get("NORMAL", "CENTER"), get("BLOCK_H1_FEEDBACK", "CENTER")
                        ln, lb = get("NORMAL", "LARGE"), get("BLOCK_H1_FEEDBACK", "LARGE")
                        pn, pb = get("NORMAL", preferred), get("BLOCK_H1_FEEDBACK", preferred)
                        record = {"cell": cell["cell"], "seed": str(seed), "polarity": cell["polarity"], "cell_type": cell["cell_type"],
                                  "contrast_sign": sign, "contrast_class": "preferred" if direction > 0 else "opposite", "window": window,
                                  "variable": variable, "preferred_center": preferred,
                                  "R_center_NORMAL": cn, "R_center_BLOCK": cb, "R_large_NORMAL": ln, "R_large_BLOCK": lb,
                                  "R_surround_NORMAL": get("NORMAL", "SURROUND"), "R_surround_BLOCK": get("BLOCK_H1_FEEDBACK", "SURROUND"),
                                  "S_NORMAL": cn - ln, "S_BLOCK": cb - lb, "Delta_S_H1": (cn - ln) - (cb - lb),
                                  "Delta_center": cb - cn, "Delta_large": lb - ln,
                                  "S_large_NORMAL": pn - ln, "S_large_BLOCK": pb - lb,
                                  "Delta_S_large_H1": (pn - ln) - (pb - lb),
                                  "gain_only_residual": lb - cb / cn * ln if cn > 0 and cb > 0 else None}
                        assert abs(record["Delta_S_H1"] - (record["Delta_large"] - record["Delta_center"])) < 1e-12
                        seed_records.append(record)
                    aggregate = {**seed_records[0], "seed": "aggregate"}
                    for measure in MEASURES:
                        if measure != "gain_only_residual":
                            aggregate[measure] = float(np.mean([r[measure] for r in seed_records]))
                    cn, cb = aggregate["R_center_NORMAL"], aggregate["R_center_BLOCK"]
                    aggregate["gain_only_residual"] = aggregate["R_large_BLOCK"] - cb / cn * aggregate["R_large_NORMAL"] if cn > 0 and cb > 0 else None
                    per_cell.extend([*seed_records, aggregate])
    groups = {"ALL": lambda r: True, "ON": lambda r: r["polarity"] == "ON", "OFF": lambda r: r["polarity"] == "OFF",
              "midget": lambda r: r["cell_type"] == "midget", "parasol": lambda r: r["cell_type"] == "parasol"}
    for polarity in ("ON", "OFF"):
        for cell_type in ("midget", "parasol"):
            groups[polarity + "_" + cell_type] = lambda r, p=polarity, t=cell_type: r["polarity"] == p and r["cell_type"] == t
    rng = np.random.default_rng(definition["bootstrap_seed"])
    grouped = defaultdict(list)
    for r in per_cell:
        grouped[(r["seed"], r["contrast_class"], r["window"], r["variable"])].append(r)
    summary = []
    for key in sorted(grouped):
        for group, predicate in groups.items():
            selected = [r for r in grouped[key] if predicate(r)]
            for measure in MEASURES:
                summary.append({"seed": key[0], "contrast_class": key[1], "window": key[2], "variable": key[3],
                                "group": group, "measure": measure, **stats([r[measure] for r in selected], rng)})
    def lookup(variable, measure, group="ALL", seed="aggregate"):
        return next(r for r in summary if r["variable"] == variable and r["measure"] == measure and r["group"] == group
                    and r["seed"] == seed and r["window"] == "late" and r["contrast_class"] == "preferred")
    sn, ds = lookup("logit", "S_NORMAL"), lookup("logit", "Delta_S_H1")
    gain = lookup("logit", "gain_only_residual")
    consistent = (sn["ci_low"] > 0 and ds["ci_low"] > 0 and ds["positive"] > 11
                  and all(lookup("logit", "Delta_S_H1", seed=str(seed))["mean"] > 0 for seed in definition["seeds"])
                  and all(lookup("logit", "Delta_S_H1", group)["mean"] > 0 for group in ("ON", "OFF", "midget", "parasol"))
                  and lookup("effective_drive_E", "Delta_S_H1")["mean"] > 0
                  and gain["n_cells"] > 11 and gain["ci_low"] > 0)
    mixed = any(lookup(v, "S_NORMAL")["ci_low"] > 0 and lookup(v, "Delta_S_H1")["mean"] > 0 for v in ("effective_drive_E", "logit"))
    verdict = "FUNCTIONALLY_CONSISTENT" if consistent else "MIXED" if mixed else "NOT_SUPPORTED"
    payload = {"per_cell": per_cell, "summary": summary, "trace_summary": rows, "verdict": verdict,
               "preferences": [{"cell": c, "sign": sign, "preferred_center": p} for (c, sign), p in preferences.items()]}
    save_json(TEMP / "analysis.json", payload)
    plot_results(definition, payload)
    print(json.dumps({"verdict": verdict, "S_NORMAL_logit": sn, "Delta_S_H1_logit": ds, "gain_residual": gain}, ensure_ascii=False), flush=True)


def plot_results(definition, payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    normal_color, block_color = "#146c94", "#cc6b32"
    rows = payload["trace_summary"]
    cell_by_id = {c["cell"]: c for c in definition["cells"]}
    def selected_row(r):
        p = 1 if r["polarity"] == "ON" else -1
        return r["contrast_sign"] == p and r["window"] == "late"
    late = [r for r in rows if selected_row(r)]
    spots = ("SPOT_SMALL", "SPOT_MID", "CENTER", "SPOT_EXPAND_1", "SPOT_EXPAND_2", "LARGE")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    for ax, variable in zip(axes, PRIMARY):
        for condition, color in zip(CONDITIONS, (normal_color, block_color)):
            values = []
            for spot in spots:
                by_cell = [[r[variable] for r in late if r["cell"] == cell and r["condition"] == condition and r["stimulus"] == spot] for cell in cell_by_id]
                values.append(np.array([np.mean(v) for v in by_cell]))
            matrix = np.array(values)
            mean = matrix.mean(1)
            sem = matrix.std(1, ddof=1) / np.sqrt(22)
            ax.plot(range(6), mean, "o-", color=color, label=condition)
            ax.fill_between(range(6), mean - sem, mean + sem, color=color, alpha=.12)
        ax.set_xticks(range(6), ["Small", "Mid", "Center", "Expand 1", "Expand 2", "Large"], rotation=35)
        ax.set_title(variable.replace("_drive", "").replace("_readout", ""))
        ax.set_ylabel("Baseline-subtracted response")
        ax.axhline(0, color="#aaaaaa", lw=.6)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen size tuning | preferred contrast, late window | cell mean ± SEM (22 cells)")
    fig.savefig(OUT / "figures/size_tuning.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    for ax, (polarity, cell_type) in zip(axes, (("ON", "midget"), ("OFF", "midget"), ("ON", "parasol"), ("OFF", "parasol"))):
        for condition, color in zip(CONDITIONS, (normal_color, block_color)):
            cells = [c["cell"] for c in definition["cells"] if c["polarity"] == polarity and c["cell_type"] == cell_type]
            matrix = np.array([[np.mean([r["logit"] for r in late if r["cell"] == cell and r["condition"] == condition and r["stimulus"] == spot]) for spot in ("CENTER", "LARGE")] for cell in cells])
            for line in matrix:
                ax.plot([0, 1], line, color=color, alpha=.2, lw=.8)
            ax.plot([0, 1], matrix.mean(0), "o-", color=color, lw=2.3, label=condition)
        ax.set_xticks([0, 1], ["Center", "Center + surround"])
        ax.set_title(f"{polarity} {cell_type} (n={len(cells)})")
        ax.set_ylabel("Late logit response")
        ax.axhline(0, color="#aaaaaa", lw=.6)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed-center surround comparison | three seeds averaged within each cell")
    fig.savefig(OUT / "figures/center_surround.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 5, figsize=(18, 11), constrained_layout=True)
    columns = ("h1_feedback", "bc_direct_effective_drive", "effective_drive_E", "effective_drive_I", "membrane_readout")
    time_ms = (np.arange(150) - 30) * 1000 / 150
    for row_index, cell_id in enumerate(REPRESENTATIVES):
        cell = cell_by_id[cell_id]
        center = next(m["indices"] for m in cell["masks"] if m["name"] == "CENTER")
        data = {field: [] for field in columns}
        for seed in definition["seeds"]:
            with np.load(OUT / f"figures/trace_data/{cell_id.replace('#', '_')}_{seed}.npz", allow_pickle=False) as archive:
                cases = json.loads(str(archive["cases_json"]))
                sign = 1 if cell["polarity"] == "ON" else -1
                i = next(i for i, c in enumerate(cases) if c["name"] == "LARGE" and c["sign"] == sign)
                for field in columns:
                    lines = np.stack([scalar_trace(archive[field][c], field, center) for c in range(2)])
                    data[field].append(lines[:, i] - lines[:, 0])
        for column, field in enumerate(columns):
            ax = axes[row_index, column]
            mean = np.mean(data[field], axis=0)
            for c, color in enumerate((normal_color, block_color)):
                ax.plot(time_ms, mean[c], color=color, label=CONDITIONS[c])
            ax.axvspan(0, 400, color="#e8e8e8", alpha=.6, zorder=0)
            if row_index == 0:
                ax.set_title(field.replace("_drive", "").replace("_readout", ""))
            if column == 0:
                ax.set_ylabel(f"{cell_id} {cell['polarity']}\n{cell['cell_type']}")
            if row_index == 3:
                ax.set_xlabel("Time from onset (ms)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Preselected representative traces | LARGE, preferred contrast | seed mean; gray = stimulus")
    fig.savefig(OUT / "figures/representative_traces.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(18, 7), constrained_layout=True)
    order = sorted(cell_by_id, key=lambda c: (cell_by_id[c]["polarity"], cell_by_id[c]["cell_type"], c))
    for ax, variable in zip(axes, PRIMARY):
        relevant = [r for r in payload["per_cell"] if r["variable"] == variable and r["window"] == "late" and r["contrast_class"] == "preferred"]
        for y_index, cell in enumerate(order):
            values = [r["Delta_S_H1"] for r in relevant if r["cell"] == cell and r["seed"] != "aggregate"]
            aggregate = next(r["Delta_S_H1"] for r in relevant if r["cell"] == cell and r["seed"] == "aggregate")
            ax.plot(values, [y_index] * 3, ".", color="#a7a7a7", markersize=5)
            ax.plot(aggregate, y_index, "o", color=normal_color if aggregate > 0 else block_color, markersize=5)
        ax.axvline(0, color="#444444", lw=.8)
        ax.set_yticks(range(len(order)), [f"{c} {cell_by_id[c]['polarity']} {cell_by_id[c]['cell_type']}" for c in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(variable.replace("_drive", "").replace("_readout", ""))
        ax.set_xlabel("Delta S H1 (positive = less suppression after block)")
    fig.suptitle("Per-cell H1 interaction | late, preferred contrast | color = seed mean; gray = individual seeds")
    fig.savefig(OUT / "figures/per_cell_delta_s.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run", "analyze"))
    parsed = parser.parse_args()
    run() if parsed.phase == "run" else analyze()
