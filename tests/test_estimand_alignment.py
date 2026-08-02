from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import h5py
import numpy as np

from benchmarks.point_process_teacher import TeacherPopulationConfig, generate_teacher_responses
from data.rgc_response_export import write_rgc_response
from data.synthetic_teacher import fit_teacher_input_normalization
from evaluation.history_estimands import (
    HistoryEstimandRequest,
    audit_history_estimands,
)
from evaluation.teacher_self_fit import (
    TeacherSelfFitRequest,
    audit_teacher_self_fit,
)
from evaluation.teacher_self_fit_model import history_trace
from scripts.run_estimand_alignment_experiments import _parameter_recovery_status


def test_parameter_recovery_status_fails_when_controlled_history_fails() -> None:
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


def test_parameter_recovery_status_ignores_exploratory_sign_only_mutation() -> None:
    # Given
    controlled = _controlled_history_results()
    controlled["zero"]["teacher_signed_gains"] = (float("-1"), float("1"))

    # When
    status = _parameter_recovery_status(
        controlled,
        per_cell_kernel_correlation=(0.5, 0.6),
        per_cell_kernel_error=(0.1, 0.2),
    )

    # Then
    assert status == "supported"


def test_history_trace_uses_only_past_events() -> None:
    # Given
    spikes = np.asarray([[[[1.0], [0.0], [1.0]]]], dtype=np.float32)

    # When
    trace = history_trace(spikes, decay=0.5)

    # Then
    assert np.allclose(trace.reshape(-1), (0.0, 1.0, 0.5))


def test_history_estimands_preserve_diagnostic_decomposition(tmp_path: Path) -> None:
    # Given
    path = _teacher_path(tmp_path)

    # When
    audit = audit_history_estimands(
        HistoryEstimandRequest(path, probe_steps=32, endogenous_trials=64, seed=23)
    )

    # Then
    assert set(audit.controlled_by_history) == {
        "zero",
        "matched_observed",
        "standard_train_rate",
    }
    assert tuple(mode.history_contract for mode in audit.modes) == (
        "zero",
        "matched_observed",
        "standard_train_rate",
        "observed_history",
        "endogenous_history",
    )
    assert np.allclose(
        np.asarray(audit.direct_response_gain)
        + np.asarray(audit.history_mediated_gain),
        audit.marginal_response_gain,
    )


def test_teacher_self_fit_smoke_reports_requested_trial_count(tmp_path: Path) -> None:
    # Given
    path = _teacher_path(tmp_path)

    # When
    audit = audit_teacher_self_fit(
        TeacherSelfFitRequest(
            path,
            trial_counts=(2,),
            monte_carlo_seeds=2,
            max_iterations=1,
            probe_steps=32,
            seed=29,
        )
    )

    # Then
    assert tuple(point.trial_count for point in audit.points) == (2,)
    assert len(audit.points[0].per_cell_direction_recovery_rate) == 4
    assert np.isfinite(audit.points[0].heldout_nll_mean)


def test_teacher_self_fit_reports_arbitrary_cell_group_recovery(
    tmp_path: Path,
) -> None:
    # Given
    path = _teacher_path(tmp_path, cells_per_type_polarity=4)

    # When
    audit = audit_teacher_self_fit(
        TeacherSelfFitRequest(
            path,
            trial_counts=(2,),
            monte_carlo_seeds=1,
            max_iterations=1,
            probe_steps=32,
            seed=31,
        )
    )

    # Then
    point = audit.points[0]
    assert len(point.per_cell_kernel_error) == 16
    assert len(point.per_cell_kernel_correlation) == 16
    assert {effect.effect for effect in point.group_effect_recovery} == {
        "type",
        "polarity",
        "interaction",
    }


def test_estimand_cli_separates_recovery_and_observational_status(
    tmp_path: Path,
) -> None:
    # Given
    path = _teacher_path(tmp_path, cells_per_type_polarity=4)
    output = tmp_path / "estimands.json"

    # When
    subprocess.run(
        [
            sys.executable,
            "scripts/run_estimand_alignment_experiments.py",
            "--validation",
            str(path),
            "--output",
            str(output),
            "--smoke",
            "--monte-carlo-seeds",
            "2",
            "--self-fit-iterations",
            "1",
            "--convergence-check-seeds",
            "2",
            "--convergence-check-iterations",
            "1",
        ],
        check=True,
    )

    # Then
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report) == {
        "parameter_recovery",
        "observational_response",
        "exploratory_type_effects",
        "status",
    }
    assert report["parameter_recovery"]["status"] in {"supported", "not_supported"}
    assert set(report["parameter_recovery"]["controlled_history_results"]) == {
        "zero",
        "matched_observed",
        "standard_train_rate",
    }
    assert all(
        len(result["conditional_logit_rf_gain"]) == 16
        for result in report["parameter_recovery"][
            "controlled_history_results"
        ].values()
    )
    assert report["observational_response"]["status"] == "supported"
    assert report["parameter_recovery"]["self_fit_power"]["monte_carlo_seeds"] == 2
    assert "teacher_signed_gains" in report["exploratory_type_effects"]


def test_estimand_cli_missing_teacher_keeps_observational_response(
    tmp_path: Path,
) -> None:
    # Given
    path = _teacher_path(tmp_path, include_teacher=False)
    output = tmp_path / "missing-teacher.json"

    # When
    subprocess.run(
        [
            sys.executable,
            "scripts/run_estimand_alignment_experiments.py",
            "--validation",
            str(path),
            "--output",
            str(output),
            "--smoke",
        ],
        check=True,
    )

    # Then
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parameter_recovery"]["status"] == "not_identifiable"
    assert "Synthetic teacher" in report["parameter_recovery"]["reason"]
    assert report["observational_response"]["status"] == "supported"


def test_estimand_cli_missing_teacher_normalization_keeps_observational_response(
    tmp_path: Path,
) -> None:
    # Given
    path = _teacher_path(tmp_path, cells_per_type_polarity=4)
    with h5py.File(path, "a") as handle:
        del handle["teacher/input_mean"]
    output = tmp_path / "partial-teacher.json"

    # When
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_estimand_alignment_experiments.py",
            "--validation",
            str(path),
            "--output",
            str(output),
            "--smoke",
        ],
        check=False,
    )

    # Then
    assert completed.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parameter_recovery"]["status"] == "not_identifiable"
    assert "teacher normalization" in report["parameter_recovery"]["reason"]
    assert report["observational_response"]["status"] == "supported"


def _teacher_path(
    tmp_path: Path,
    *,
    cells_per_type_polarity: int = 1,
    include_teacher: bool = True,
) -> Path:
    rng = np.random.default_rng(17)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    normalization = fit_teacher_input_normalization(cones)
    generated = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        np.arange(80) * 0.005,
        trials=2,
        seed=19,
        adaptive=True,
        teacher_normalization=normalization,
        population_config=TeacherPopulationConfig(
            cells_per_type_polarity=cells_per_type_polarity
        ),
    )
    path = tmp_path / "teacher.h5"
    if include_teacher:
        write_rgc_response(
            path,
            generated.session,
            teacher_kernels=generated.kernels,
            teacher_normalization=normalization,
        )
    else:
        write_rgc_response(path, generated.session)
    return path


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
