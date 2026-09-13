# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    initial = json.loads((OUT / "evidence_manifest.initial.json").read_text())
    protocol = json.loads((OUT / "protocol_hash.json").read_text())
    assert sha(OUT / "TEST_PROTOCOL.md") == protocol["sha256"]
    assert sha(OUT / "data_usage_gate.json") == protocol["data_usage_gate_sha256"]
    assert sha(OUT / "data_usage_inventory.csv") == protocol["data_usage_inventory_sha256"]
    changes = []
    for path, expected in initial["input_sha256"].items():
        current = sha(ROOT / path)
        if current != expected:
            changes.append(dict(path=path, expected=expected, observed=current))
    assert not changes, changes
    required = ["DATA_USAGE_REPORT.md", "data_usage_inventory.csv", "TEST_PROTOCOL.md", "protocol_hash.json",
        "preflight.json", "per_cell_test_nll.csv", "paired_model_comparisons.csv", "alignment_heldout_relation.csv",
        "development_vs_heldout.csv", "pathway_per_cell.csv", "pathway_population.csv", "pathway_logit_vs_nll.csv",
        "TEST_CONSUMED.md", "REPORT.md"]
    assert all((OUT / name).is_file() for name in required)
    preflight = json.loads((OUT / "preflight.json").read_text())
    status = json.loads((OUT / "evaluation_status.json").read_text())
    assert preflight["all_passed"] and len(preflight["cells"]) == 88
    assert status["status"] == "COMPLETE" and status["completed_cells"] == 22
    with (OUT / "model_artifact_inventory.csv").open(encoding="utf-8") as stream:
        models = list(csv.DictReader(stream))
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert branch == initial["branch"] and head == initial["HEAD"]
    git_status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True)
    prior = set(initial["git_status_at_audit_start"].splitlines())
    current = set(git_status.splitlines())
    outputs = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob("*"))
               if p.is_file() and p.name != "evidence_manifest.json" and "__pycache__" not in p.parts}
    dependencies = {
        "data_usage_inventory.csv": ["repository_search_inventory.csv", "scan_escalated/", "executed_adapter_configs.csv", "binary_source_ids_default.csv", "binary_source_ids_escalated.csv", "archive_members.csv", "production_source_ranges.csv", "recording_trial_inventory.csv"],
        "per_cell_test_nll.csv": ["TEST_PROTOCOL.md", "preflight.json", "test_source_ranges.csv", "test_identity_checks.json", "model_artifact_inventory.csv", "cells/*-test-predictions.pt"],
        "paired_model_comparisons.csv": ["per_cell_test_nll.csv", "summarize_heldout.py", "TEST_PROTOCOL.md"],
        "alignment_heldout_relation.csv": ["per_cell_test_nll.csv", initial["lineages"]["fixed_alignment_audit"] + "/per_cell_results.csv"],
        "development_vs_heldout.csv": [initial["lineages"]["fixed_alignment_audit"] + "/summary.json", "paired_model_comparisons.csv", "alignment_heldout_relation.csv"],
        "pathway_per_cell.csv": ["cells/*-test-predictions.pt", "test_source_ranges.csv", "model_artifact_inventory.csv", "evaluate_heldout.py", "TEST_PROTOCOL.md"],
        "pathway_population.csv": ["pathway_per_cell.csv", "summarize_heldout.py", "TEST_PROTOCOL.md"],
        "pathway_logit_vs_nll.csv": ["pathway_per_cell.csv"],
        "pathway_logit_nll_correlations.csv": ["pathway_logit_vs_nll.csv", "summarize_heldout.py"],
    }
    manifest = dict(status="COMPLETE", generated_utc=datetime.now(timezone.utc).isoformat(), branch=branch, HEAD=head,
        protocol_sha256=protocol["sha256"], protocol_unchanged=True,
        range=dict(live_seconds=[20, 60], live_frames=[3000, 9000], decoded_frames=[3751, 9751],
            bounds="zero-based half-open", cells=22, recordings=37, trials=137, sequences=5480, scored_bins=657600),
        input_sha256=initial["input_sha256"], frozen_input_count=len(initial["input_sha256"]),
        frozen_input_hash_changes=changes, production_source_and_data_modifications=0,
        checkpoint_identity=[dict(cell_id=r["cell_id"], model=r["model"], path=r["path"], sha256=initial["input_sha256"][r["path"]]) for r in models],
        checkpoint_parameter_modifications=0, new_model_training_runs=0, new_model_seeds=0, hyperparameter_tuning=0,
        checkpoint_selection=0, new_training_checkpoints=0, frozen_model_prediction_arrays=22*7,
        statistical_bootstrap=dict(seed=2026090501, draws=100000, unit="paired biological cell", equal_cell_weight=True,
            index_sha256=json.loads((OUT / "analysis_summary.json").read_text())["bootstrap_index_sha256"]),
        p_values=0, protocol_deviations="No data/model/statistical-definition changes. Audit runner startup and compiled-rank dependency corrections are documented in startup_check_record.md; no repeated held-out forward passes.",
        derived_table_dependencies=dependencies, output_sha256=outputs,
        git_status_at_finish=git_status, git_status_added_entries=sorted(current-prior), git_status_removed_entries=sorted(prior-current),
        external_writes_or_publication=False,
        reproduction="Read-only verification may recompute NLL/statistics from saved predictions. Do not rerun held-out producer scripts as a new confirmatory experiment; range is consumed. Initial source/data hashes and immutable protocol identify the completed run.")
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(frozen_inputs_verified=len(initial["input_sha256"]), frozen_input_changes=len(changes),
        output_files_hashed=len(outputs), checkpoints=len(models), new_status_entries=len(current-prior),
        removed_status_entries=len(prior-current), protocol_unchanged=True), indent=2))


if __name__ == "__main__":
    main()
