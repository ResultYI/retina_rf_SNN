from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from evaluation.type_prior_comparison import (
    TypePriorComparisonError,
    compare_type_prior_runs,
    overall_status,
)
from tests.type_prior_fixture import (
    mutate_artifact as _mutate_artifact,
    mutate_manifest as _mutate_manifest,
    remove_history as _remove_history,
    set_source_pairs as _set_source_pairs,
    write_grid as _write_grid,
)


def test_cli_reports_type_prior_value_when_validation_runs_are_comparable(
    tmp_path: Path,
) -> None:
    # Given
    runs = _write_grid(tmp_path)
    output = tmp_path / "comparison.json"

    # When
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_type_prior_variants.py",
            "--output",
            str(output),
            *[item for run in runs for item in ("--run", str(run))],
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    assert tuple(report) == (
        "run_contract",
        "comparisons",
        "predictive_value",
        "rf_stability_value",
        "data_efficiency_value",
        "status",
    )
    assert report["status"] == "supported"
    assert report["predictive_value"]["status"] == "supported"
    assert report["rf_stability_value"]["status"] == "supported"
    assert report["data_efficiency_value"]["trial_budget"] == 1
    assert report["run_contract"]["evaluation_split"] == "validation"
    assert len(report["run_contract"]["input_runs"]) == len(runs)
    assert report["comparisons"]["type_aware_vs_type_blind"]["paired_sample_count"] == 4
    assert report["comparisons"]["type_aware_vs_shuffled_type"]["paired_sample_count"] == 4
    assert report["comparisons"]["type_aware_vs_cell_only"]["reference_only"] is True
    assert "glm_test" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "foreign_fingerprint",
            lambda run: _mutate_manifest(run, ("dataset_fingerprint",), "foreign"),
            "dataset fingerprint",
        ),
        (
            "cell_order",
            lambda run: _mutate_artifact(run, "cell_ids", ("cell-b", "cell-a")),
            "cell identity",
        ),
        (
            "cone_identity",
            lambda run: _mutate_artifact(
                run,
                "cone_positions_degs",
                torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
            ),
            "cone identity",
        ),
        (
            "lag_order",
            lambda run: _mutate_artifact(run, "lag_order", "current_to_oldest"),
            "lag order",
        ),
        (
            "history_contract",
            lambda run: _remove_history(run, "zero"),
            "history contract",
        ),
        (
            "source_pair_count",
            lambda run: _set_source_pairs(run, 2),
            "source-pair count",
        ),
        (
            "training_budget",
            lambda run: _mutate_manifest(
                run,
                ("config", "training", "max_optimizer_steps"),
                9,
            ),
            "complete variant grid",
        ),
        (
            "test_split",
            lambda run: _mutate_manifest(run, ("evaluation_split",), "test"),
            "validation split",
        ),
    ],
)
def test_comparator_rejects_incomparable_runs(
    tmp_path: Path,
    name: str,
    mutate: Callable[[Path], None],
    message: str,
) -> None:
    # Given
    runs = _write_grid(tmp_path)
    mutate(tmp_path / "type_blind-s1-b1")

    # When / Then
    with pytest.raises(TypePriorComparisonError, match=message):
        compare_type_prior_runs(runs)


def test_cli_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    # Given
    runs = _write_grid(tmp_path)
    _mutate_manifest(tmp_path / "type_blind-s1-b1", ("dataset_fingerprint",), "foreign")
    output = tmp_path / "failed.json"

    # When
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/compare_type_prior_variants.py",
            "--output",
            str(output),
            *[item for run in runs for item in ("--run", str(run))],
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then
    assert completed.returncode != 0
    assert not output.exists()


def test_one_seed_grid_is_not_identifiable(tmp_path: Path) -> None:
    # Given
    runs = _write_grid(tmp_path, seeds=(1,), budgets=(4,))

    # When
    report = compare_type_prior_runs(runs)

    # Then
    assert report["status"] == "not_identifiable"
    assert report["predictive_value"]["status"] == "not_identifiable"
    assert report["rf_stability_value"]["status"] == "not_identifiable"


def test_comparator_reports_predictive_and_rf_results_per_cell(
    tmp_path: Path,
) -> None:
    # Given
    runs = _write_grid(tmp_path)

    # When
    report = compare_type_prior_runs(runs)

    # Then
    predictive = report["comparisons"]["type_aware_vs_type_blind"]
    assert [item["cell_id"] for item in predictive["per_cell_nll_delta"]] == [
        "cell-a",
        "cell-b",
    ]
    assert all(item["paired_sample_count"] == 4 for item in predictive["per_cell_nll_delta"])
    stability = report["comparisons"]["rf_stability"]
    assert [item["cell_id"] for item in stability["by_variant"]["type_aware"]["per_cell"]] == [
        "cell-a",
        "cell-b",
    ]
    assert [item["cell_id"] for item in stability["type_aware_vs_type_blind"]["per_cell_cosine_delta"]] == [
        "cell-a",
        "cell-b",
    ]


def test_matched_comparator_reports_learning_gain_and_learned_rf_delta(
    tmp_path: Path,
) -> None:
    # Given
    runs = _write_grid(
        tmp_path,
        seeds=(1, 2, 3),
        budgets=(20,),
        shuffle_mode="balanced_shuffled_type",
        matched_initialization=True,
    )

    # When
    report = compare_type_prior_runs(runs)

    # Then
    matched = report["comparisons"]["matched_initialization"]
    assert matched["passed"] is True
    assert matched["max_effective_parameter_difference"] < 1e-7
    assert matched["max_step0_nll_difference"] < 1e-7
    learning = report["comparisons"]["learning_gain"]
    assert learning["type_aware_vs_type_blind"]["learning_gain_delta"] > 0
    assert learning["type_aware_vs_balanced_shuffled_type"]["learning_gain_delta"] > 0
    assert learning["type_aware_vs_type_blind"]["seed_direction_consistent"] is True
    assert (
        learning["type_aware_vs_type_blind"]["factorial_contrasts"]["identifiable"]
        is False
    )
    learned_rf = report["comparisons"]["learned_delta_rf_stability"]
    assert "by_variant" in learned_rf
    assert (
        learned_rf["by_variant"]["type_aware"][
            "mean_trained_initialized_cosine_distance"
        ]
        > 0
    )
    assert report["data_efficiency_value"]["estimand"] == "learning_gain"


def test_cell_only_reference_is_optional(tmp_path: Path) -> None:
    # Given
    runs = [
        run
        for run in _write_grid(tmp_path)
        if "cell_only" not in run.name
    ]

    # When
    report = compare_type_prior_runs(runs)

    # Then
    assert report["status"] == "supported"
    assert "type_aware_vs_cell_only" not in report["comparisons"]
    assert "cell_only" not in report["comparisons"]["rf_stability"]["by_variant"]


def test_comparator_rejects_seed_budget_mismatch(tmp_path: Path) -> None:
    # Given
    runs = [
        run
        for run in _write_grid(tmp_path)
        if "-s2-b1" not in run.name
    ]

    # When / Then
    with pytest.raises(TypePriorComparisonError, match="training budget"):
        compare_type_prior_runs(runs)


@pytest.mark.parametrize(
    ("predictive", "stability", "efficiency", "expected"),
    [
        ("supported", "not_supported", "not_supported", "supported"),
        ("supported", "significant_disadvantage", "not_supported", "mixed"),
        ("not_supported", "not_supported", "not_supported", "not_supported"),
        ("not_identifiable", "supported", "not_supported", "not_identifiable"),
    ],
)
def test_overall_status_rule_combinations(
    predictive: str,
    stability: str,
    efficiency: str,
    expected: str,
) -> None:
    # Given / When / Then
    assert overall_status((predictive, stability, efficiency)) == expected
