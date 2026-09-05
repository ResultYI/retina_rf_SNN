from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / ".omo/evidence/rf-identifiability-reachability-audit"
REQUIRED = (
    "rf-identifiability-results.json",
    "stimulus-subspace-audit.json",
    "teacher-estimand-audit.json",
    "oracle-recovery-results.json",
    "stagewise-rf-recovery.json",
    "rf-increment-decomposition.json",
    "gradient-reachability-results.json",
    "numerical-validation.json",
    "identity-manifest.json",
    "per-cell-metrics.csv",
    "per-seed-summary.csv",
    "decision-report.md",
    "decision-report-zh.md",
    "config-seed19.yaml",
    "config-seed20.yaml",
    "config-seed21.yaml",
)


def _load(name: str):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _assert_finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_finite(item)
        return
    if isinstance(value, float):
        assert math.isfinite(value)


def test_required_audit_artifacts_exist_and_json_is_strict() -> None:
    assert all((EVIDENCE / name).is_file() for name in REQUIRED)
    for path in EVIDENCE.glob("*.json"):
        _assert_finite(json.loads(path.read_text(encoding="utf-8")))


def test_case_ii_follows_predeclared_oracle_gates() -> None:
    decision = _load("rf-identifiability-results.json")
    gates = decision["predeclared_gates"]
    assert decision["case"] == "II"
    assert gates["teacher_dynamic_supported_energy_at_least_0_9"] is True
    assert gates["noise_free_full_cosine_at_least_0_8"] is True
    assert gates["noise_free_dynamic_cosine_at_least_0_8"] is True
    assert gates["sampled_full_cosine_at_least_0_8"] is False
    assert gates["sampled_dynamic_cosine_at_least_0_8"] is False


def test_teacher_estimand_and_stage_reconstruction_pass() -> None:
    teacher = _load("teacher-estimand-audit.json")
    stagewise = _load("stagewise-rf-recovery.json")
    assert teacher["finite_difference"]["passed"] is True
    assert teacher["history_contract"]["teacher_rf_depends_on_observed_history"] is False
    assert stagewise["raw_stage0_rf_exactly_equal"] is True
    assert all(
        value["official_stage05_tensor_exact"] is True
        for value in stagewise["reconstruction"].values()
    )


def test_checkpoints_are_unchanged_and_final_test_is_identity_only() -> None:
    identity = _load("identity-manifest.json")
    assert identity["final_test_boundary"] == [
        "TEST_SPLIT_ACCESSED_FOR_IDENTITY_ONLY",
        "TEST_EXAMPLES_NOT_USED_FOR_INFERENCE_METRICS_MODEL_SELECTION_OR_CONCLUSIONS",
        "FINAL_TEST_SCIENTIFIC_EVALUATION_NOT_CONSUMED",
    ]
    for seed in identity["checkpoints"].values():
        assert all(checkpoint["unchanged"] is True for checkpoint in seed.values())
        assert all(checkpoint["v2_readout_keys_complete"] is True for checkpoint in seed.values())


def test_reused_numerical_controls_remain_passed() -> None:
    numerical = _load("numerical-validation.json")
    prior = numerical["closure_numerical_validation_reused_not_rerun"]
    assert numerical["teacher_analytic_finite_difference"]["passed"] is True
    assert numerical["checkpoint_hashes_unchanged"] is True
    assert prior["all_seed_controls_passed"] is True
    assert prior["history_causality"]["focused_test_status"] == "PASS"
    assert prior["static_negative_control"]["focused_test_status"] == "PASS"


def test_config_snapshots_match_official_runs() -> None:
    runs = ROOT / ".omo/evidence/canonical-v2-stage05-checkpoint-rerun/runs"
    for seed in (19, 20, 21):
        snapshot = EVIDENCE / f"config-seed{seed}.yaml"
        official = runs / f"seed-{seed}-type_blind/config.yaml"
        assert snapshot.read_text(encoding="utf-8") == official.read_text(encoding="utf-8")


def test_manifest_source_hashes_match_current_files() -> None:
    identity = _load("identity-manifest.json")
    for relative_path, expected in identity["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected, relative_path
