from __future__ import annotations

from scripts.run_calibration_audit import calibration_decisions, parse_args


def test_calibration_audit_cli_has_no_final_test_surface() -> None:
    # Given / When
    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "audit.json",
        ]
    )

    # Then
    assert not hasattr(args, "final_test")


def test_calibration_decisions_apply_requested_thresholds() -> None:
    # Given / When
    decisions = calibration_decisions(
        constant_rate_nll=0.20,
        bias_only_nll=0.1995,
        full_glm_nll=0.1990,
        intercept_nll=0.2005,
        affine_nll=0.1980,
        threshold_passed=True,
        tolerance=0.001,
    )

    # Then
    assert decisions.glm == "GLM_CALIBRATION_ONLY"
    assert decisions.frozen_snn == "AFFINE_EXCEEDS_RATE_BASELINE"
    assert decisions.threshold == "THRESHOLD_BASELINE_NESTED"
    assert decisions.long_train == "NO_GO"
    assert decisions.final_test == "NOT_CONSUMED"
