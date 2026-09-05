from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CORRECTION = OUT / "correction_69_6_a0_4"
sys.path.insert(0, str(ROOT))

from baselines.center_surround_ln import LNError
from baselines.spatial_contrast_adapted import (
    CenterFilter, FittingData, FloatArray, OutputFit, bernoulli_objective,
    features_for_sequences, fit_output, reported_nll,
)
from data.schottdorf_lee_multirecording import SchottdorfCellwiseData, load_schottdorf_cell
from evaluation.mechanistic_retina.spatial_contrast_source import Sources, load_center_source, load_sources, verify_split
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file

NAMES = ("sc_adapted", "matched_w0")
METRICS = (*[f"{name}_nll" for name in NAMES], "center_surround_ln_nll", "canonical_v1_nll")


def prepare(sources: Sources, cell_id: str) -> tuple[SchottdorfCellwiseData, FittingData, FloatArray, FloatArray]:
    cell = next(c for c in sources.population.cells if c.cell_id == cell_id)
    selected = tuple(r for r in sources.recordings if r.cell_id == cell_id)
    if tuple(r.recording_id for r in selected) != cell.recording_ids:
        raise LNError("STOP: recording order changed")
    data = load_schottdorf_cell(selected, sources.movie, sources.population.adapter_config)
    model, _ = load_center_source(sources, cell)
    center = CenterFilter.from_ln(model)
    path = OUT / "filters" / f"{cell_id.replace('#', '_')}.npz"
    with np.load(path, allow_pickle=False) as saved:
        for key, value in (("spatial", center.spatial), ("unit_temporal", center.temporal),
                           ("gaussian", center.gaussian), ("amplitude", center.amplitude)):
            if not np.array_equal(saved[key], value):
                raise LNError("STOP: frozen center changed")
    prior = json.loads((OUT / "preflight.json").read_text())
    check = next(c for c in prior["checks"] if c["cell_id"] == cell_id)
    features = []
    for label, split, bins in (("train", data.train, cell.train_valid_bins),
                               ("validation", data.validation, cell.validation_valid_bins)):
        verify_split(split, bins)
        for field, value in (("input", split.cone_drive), ("target", split.spike_events), ("mask", split.valid_mask)):
            if hashlib.sha256(value.numpy().tobytes()).hexdigest() != check[f"{label}_{field}_sha256"]:
                raise LNError(f"STOP: {cell_id} {label} {field} changed")
        features.append(features_for_sequences(center, split.cone_drive)[0])
    fitting = FittingData.from_arrays(features[0], data.train.spike_counts.numpy(), data.train.valid_mask.numpy())
    for name in NAMES:
        saved = json.loads((OUT / "cells" / cell_id.replace('#', '_') / f"{name}.json").read_text())["fit"]
        if not np.array_equal(fitting.mean, saved["mean"]) or not np.array_equal(fitting.std, saved["std"]):
            raise LNError("STOP: fitting Z-score changed")
    return data, fitting, features[0], features[1]


def main() -> None:
    CORRECTION.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    other_files = [p for p in (OUT / "cells").glob("*/*") if p.is_file() and p.parent.name != "69_6"]
    other_hashes = {str(p.relative_to(OUT)): sha256_file(p) for p in other_files}
    archived = [*list((OUT / "cells/69_6").glob("*")),
                *[OUT / p for p in ("results.json", "summary.md", "per_cell.csv", "group_summary.csv",
                                   "verification.json", "output_verification.json", "initialization_count_check.json")]]
    for path in archived:
        target = CORRECTION / "before" / path.relative_to(OUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    before_hashes = {str(p.relative_to(OUT)): sha256_file(p) for p in archived}
    current_sources = {p: sha256_file(Path(p)) for p in json.loads((OUT / "preflight.json").read_text())["source_sha256"]}
    current_sources[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    record = dict(status="IN_PROGRESS", authorized_fits=[f"69#6/{n}" for n in NAMES],
                  read_only_cell="69#21", other_fit_count=42, other_fit_sha256_before=other_hashes,
                  replaced_and_summary_sha256_before=before_hashes, source_sha256=current_sources,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  git_status=subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True),
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__)),
                  protocol_changes="only 69#6 initial a0: 3 -> official 4; all other settings unchanged",
                  tests_added_or_run=0, fit_executions_this_correction=0)
    manifest = CORRECTION / "correction.json"
    manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    sources = load_sources(ROOT)
    data, fitting, _, validation = prepare(sources, "69#6")
    if fitting.maximum_count != 4:
        raise LNError("STOP: expected official a0=4")
    changes = []
    for name in NAMES:
        path = OUT / "cells/69_6" / f"{name}.json"
        payload = json.loads(path.read_text())
        old = payload["fit"]
        fit = fit_output(fitting, spatial_contrast=name == "sc_adapted")
        rate = fit.expected_counts(validation)
        nll = reported_nll(rate, data.validation.spike_events, data.validation.valid_mask)
        payload.update(fit=asdict(fit), validation_nll=nll)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(path.with_name(f"{name}-validation.npz"), expected_counts=rate,
                            probabilities=-np.expm1(-rate), target=data.validation.spike_events.numpy(),
                            valid_mask=data.validation.valid_mask.numpy(),
                            source_image_ids=np.asarray(data.validation.source_image_ids), trial_indices=data.validation.trial_indices)
        before = json.loads((CORRECTION / "before/cells/69_6" / f"{name}.json").read_text())
        changes.append(dict(model=name, initial_before=old["initial_parameters"], initial_after=fit.initial_parameters,
                            train_nll_before=old["train_nll"], train_nll_after=fit.train_nll,
                            validation_nll_before=before["validation_nll"], validation_nll_after=nll,
                            optimizer_success=fit.optimizer_success, optimizer_status=fit.optimizer_status,
                            iterations=fit.iterations, final_objective=bernoulli_objective(np.asarray(fit.parameters), fitting)[0]))
        record.update(changes_69_6=changes, fit_executions_this_correction=len(changes))
        manifest.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        print(f"CORRECTED 69#6 {name}: {before['validation_nll']:.12f} -> {nll:.12f}; success={fit.optimizer_success}", flush=True)
    data, fitting, training, validation = prepare(sources, "69#21")
    endpoint_checks = []
    for name in NAMES:
        path = OUT / "cells/69_21" / f"{name}.json"
        saved = json.loads(path.read_text())
        fit = OutputFit(**saved["fit"])
        objective, gradient = bernoulli_objective(np.asarray(fit.parameters), fitting)
        train_rate, val_rate = fit.expected_counts(training), fit.expected_counts(validation)
        train_nll = reported_nll(train_rate, data.train.spike_events, data.train.valid_mask)
        val_nll = reported_nll(val_rate, data.validation.spike_events, data.validation.valid_mask)
        with np.load(path.with_name(f"{name}-validation.npz"), allow_pickle=False) as output:
            output_exact = np.array_equal(val_rate, output["expected_counts"])
        finite = all(bool(np.isfinite(x).all()) for x in (objective, gradient, fit.parameters, train_rate,
                                                        val_rate, -np.expm1(-train_rate), -np.expm1(-val_rate)))
        endpoint_checks.append(dict(model=name, optimizer_success=fit.optimizer_success, optimizer_status=fit.optimizer_status,
                                    optimizer_message=fit.optimizer_message, iterations=fit.iterations,
                                    final_objective=objective, train_nll=train_nll, validation_nll=val_nll,
                                    train_nll_abs_error=abs(train_nll-fit.train_nll),
                                    validation_nll_abs_error=abs(val_nll-saved["validation_nll"]),
                                    max_abs_gradient=float(np.max(np.abs(gradient))),
                                    all_endpoint_outputs_finite=finite, saved_outputs_exact=output_exact,
                                    retained_unchanged=True))
        print(f"READ ONLY 69#21 {name}: success={fit.optimizer_success}, status={fit.optimizer_status}, objective={objective:.12f}, finite={finite}", flush=True)
    (CORRECTION / "69_21_read_only.json").write_text(json.dumps(endpoint_checks, indent=2, allow_nan=False), encoding="utf-8")
    if any(sha256_file(OUT / p) != h for p, h in other_hashes.items()):
        raise LNError("STOP: an unrequested fit artifact changed")
    if any(sha256_file(Path(p)) != h for p, h in current_sources.items()):
        raise LNError("STOP: a source changed during correction")
    results = json.loads((OUT / "results.json").read_text())
    row = next(r for r in results["cells"] if r["cell_id"] == "69#6")
    for change in changes:
        name = change["model"]
        row.update({f"{name}_nll": change["validation_nll_after"], f"{name}_success": change["optimizer_success"],
                    f"{name}_iterations": change["iterations"]})
    row["frozen_initialization_match"] = True
    summaries = []
    for group in ("overall", "MC_ON", "MC_OFF", "PC_ON", "PC_OFF"):
        selected = results["cells"] if group == "overall" else [r for r in results["cells"] if r["group"] == group]
        summaries.append(dict(group=group, cells=len(selected), **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}))
    results.update(status="COMPLETED_CORRECTED", summary=summaries, frozen_benchmark_accepted=True,
                   correction_evidence="correction_69_6_a0_4/correction.json")
    results["contract"]["initialization"] = "[max(entire training response raw counts), -2, 1, 0]; matched control omits fixed last zero"
    results["completion"].update(refits_after_correction=2, exact_initialization_cells=22, total_fit_executions_including_superseded=46)
    results["initialization_deviation"].update(numerical_results_recomputed_after_correction=True, remaining_blocker=None,
                                               resolution="69#6 both fits replaced with authorized a0=4 execution")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    for filename, rows in (("per_cell.csv", results["cells"]), ("group_summary.csv", summaries)):
        with (OUT / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    verification = json.loads((OUT / "output_verification.json").read_text())
    for item in verification["rows"]:
        if item["cell_id"] == "69#6":
            fit = json.loads((OUT / "cells/69_6" / f"{item['model']}.json").read_text())["fit"]
            item.update(nll_recomputed=row[f"{item['model']}_nll"], max_abs_gradient=max(abs(x) for x in fit["gradient"]),
                        optimizer_message=fit["optimizer_message"], evidence="correction_69_6_a0_4/correction.json")
    verification["correction"] = "69#6 replaced; other 42 fit JSON/prediction artifacts unchanged by SHA256"
    (OUT / "output_verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False), encoding="utf-8")
    record.update(status="COMPLETED_CORRECTED", other_42_fits_unchanged=True, source_hashes_unchanged_during_correction=True,
                  summary=summaries, endpoint_69_21=endpoint_checks,
                  replacement_sha256={str(p.relative_to(OUT)): sha256_file(p) for p in (OUT / "cells/69_6").glob("*")})
    manifest.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
