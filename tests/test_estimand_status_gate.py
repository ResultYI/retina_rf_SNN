from __future__ import annotations

from math import inf

from scripts.run_estimand_alignment_experiments import _parameter_recovery_status


def test_status_gate_missing_controlled_history_is_not_identifiable() -> None:
    # Given
    controlled = _controlled_history_results()
    del controlled["standard_train_rate"]

    # When
    status = _parameter_recovery_status(
        controlled,
        per_cell_kernel_correlation=(0.5, 0.6),
        per_cell_kernel_error=(0.1, 0.2),
    )

    # Then
    assert status == "not_identifiable"


def test_status_gate_nonfinite_controlled_history_is_not_identifiable() -> None:
    # Given
    controlled = _controlled_history_results()
    controlled["matched_observed"]["probability_rf_gain"] = (0.1, inf)

    # When
    status = _parameter_recovery_status(
        controlled,
        per_cell_kernel_correlation=(0.5, 0.6),
        per_cell_kernel_error=(0.1, 0.2),
    )

    # Then
    assert status == "not_identifiable"


def test_status_gate_failed_controlled_history_is_not_supported() -> None:
    # Given
    controlled = _controlled_history_results()
    controlled["standard_train_rate"]["conditional_logit_rf_gain"] = (0.0, 0.0)

    # When
    status = _parameter_recovery_status(
        controlled,
        per_cell_kernel_correlation=(0.5, 0.6),
        per_cell_kernel_error=(0.1, 0.2),
    )

    # Then
    assert status == "not_supported"


def test_status_gate_exploratory_sign_only_mutation_stays_supported() -> None:
    # Given
    controlled = _controlled_history_results()
    controlled["zero"]["teacher_signed_gains"] = (-1.0, 1.0)

    # When
    status = _parameter_recovery_status(
        controlled,
        per_cell_kernel_correlation=(0.5, 0.6),
        per_cell_kernel_error=(0.1, 0.2),
    )

    # Then
    assert status == "supported"


def test_status_gate_bad_finite_self_fit_recovery_is_not_supported() -> None:
    # Given
    controlled = _controlled_history_results()

    # When
    status = _parameter_recovery_status(
        controlled,
        per_cell_kernel_correlation=(-1.0, -1.0),
        per_cell_kernel_error=(1e12, 1e12),
    )

    # Then
    assert status == "not_supported"


def _controlled_history_results() -> dict[str, dict[str, tuple[float, ...] | str]]:
    return {
        key: {
            "history_contract": key,
            "history_source": "fixture",
            "conditional_logit_rf_gain": (0.1, -0.2),
            "probability_rf_gain": (0.01, -0.02),
            "response_gain": (0.001, -0.002),
        }
        for key in ("zero", "matched_observed", "standard_train_rate")
    }
