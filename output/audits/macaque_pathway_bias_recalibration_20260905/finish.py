# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

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
    changed = [path for path, digest in initial["input_sha256"].items() if sha(ROOT / path) != digest]
    assert not changed, changed
    protocol = json.loads((OUT / "protocol_hash.json").read_text())
    fit_lock = json.loads((OUT / "bias_fit_lock.json").read_text())
    gate = json.loads((OUT / "preflight.json").read_text())
    assert sha(OUT / "PROTOCOL.md") == protocol["sha256"]
    assert sha(OUT / "bias_fit_per_cell.csv") == fit_lock["bias_fit_per_cell_sha256"]
    assert sha(OUT / "training_only_logits.pt") == gate["training_only_logits_sha256"]
    assert gate["all_passed"] and len(gate["cells"]) == 22
    required = ["PROTOCOL.md", "protocol_hash.json", "preflight.json", "bias_fit_per_cell.csv", "normal_bias_sanity.csv",
        "pathway_bias_recalibration_per_cell.csv", "pathway_bias_recalibration_population.csv",
        "stimulus_dependent_logit_components.csv", "REPORT.md"]
    assert all((OUT / name).is_file() for name in required)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert branch == initial["branch"] and head == initial["HEAD"]
    status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True)
    prior = set(initial["git_status_at_start"].splitlines())
    current = set(status.splitlines())
    outputs = {path.relative_to(OUT).as_posix(): sha(path) for path in sorted(OUT.rglob("*"))
               if path.is_file() and path.name != "evidence_manifest.json" and "__pycache__" not in path.parts}
    dependencies = {
        "bias_fit_per_cell.csv": ["training_only_logits.pt", "preflight.json", "PROTOCOL.md", "control.py"],
        "normal_bias_sanity.csv": ["bias_fit_per_cell.csv", "bias_fit_lock.json", "prior cells/*-test-predictions.pt", "analyze.py"],
        "pathway_bias_recalibration_per_cell.csv": ["bias_fit_per_cell.csv", "bias_fit_lock.json", "prior cells/*-test-predictions.pt", "analyze.py"],
        "pathway_bias_recalibration_population.csv": ["pathway_bias_recalibration_per_cell.csv", "PROTOCOL.md", "analyze.py"],
        "stimulus_dependent_logit_components.csv": ["prior cells/*-test-predictions.pt", "analyze.py"],
    }
    value = dict(status="COMPLETE", timestamp_utc=datetime.now(timezone.utc).isoformat(), branch=branch, HEAD=head,
        protocol_sha256=protocol["sha256"], protocol_unchanged=True, bias_fit_lock=fit_lock,
        original_training_live_seconds=[0, 16], original_training_live_frames=[0, 2400], training_scored_bins=263040,
        evaluation_live_seconds=[20, 60], evaluation_live_frames=[3000, 9000], evaluation_scored_bins=657600,
        range_bounds="zero-based half-open", cells=22, recordings=37, recording_local_trials=137,
        evidence_level="predefined follow-up analysis on an already consumed confirmatory set",
        original_normal_is_primary=True, heldout_targets_used_for_calibration=False,
        fitted_scalar_biases=88, diagnostic_normal_scalars=22, model_training=0, new_model_seeds=0,
        optimizer_steps=0, architecture_modifications=0, production_source_model_data_modifications=0,
        pathway_gain_center_temporal_history_refits=0, initial_input_count=len(initial["input_sha256"]),
        changed_initial_inputs=changed, input_sha256=initial["input_sha256"], aligned_checkpoint_identity=initial["aligned_checkpoint_identity"],
        output_sha256=outputs, derived_dependencies=dependencies,
        prior_output_root="output/audits/macaque_aligned_heldout_pathway_evaluation_20260905",
        solver=dict(method="deterministic bracketed bisection", dtype="float64", mean_rate_tolerance=1e-12,
            bracket_width_tolerance=1e-12, maximum_iterations=200),
        bootstrap=dict(seed=2026090502, draws=100000, unit="paired biological cell", equal_cell_weight=True,
            percentile_method="linear", index_sha256=json.loads((OUT / "analysis_summary.json").read_text())["bootstrap_index_sha256"]),
        git_status_at_finish=status, added_status_entries=sorted(current-prior), removed_status_entries=sorted(prior-current),
        new_training_checkpoints=0, additional_experiments=0,
        reproduction="control.py freeze/preflight/fit stages document the single completed run; analyze.py applies hash-frozen training-only scalars to prior saved heldout logits. Verify derived values read-only from these hashed inputs; do not describe reuse as an untouched test.")
    (OUT / "evidence_manifest.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(inputs_verified=len(initial["input_sha256"]), changed_inputs=len(changed),
        outputs_hashed=len(outputs), protocol_unchanged=True, bias_fit_unchanged=True), indent=2))


if __name__ == "__main__":
    main()
