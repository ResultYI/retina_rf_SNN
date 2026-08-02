from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from evaluation.type_prior_comparison import TypePriorComparisonError
from evaluation.type_prior_comparison_io import load_comparable_grid
from tests.type_prior_fixture import (
    mutate_artifact as _mutate_artifact,
    mutate_artifact_free_kernel as _mutate_artifact_free_kernel,
    mutate_artifact_history_kernel as _mutate_artifact_history_kernel,
    remove_artifact_key as _remove_artifact_key,
    write_run as _write_run,
)


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "stale_schema",
            lambda run: _mutate_artifact(run, "schema", "retina-rf-artifacts-v1"),
            "schema",
        ),
        (
            "extra_top_level_key",
            lambda run: _mutate_artifact(run, "poison", 1),
            "top-level keys",
        ),
        (
            "missing_top_level_key",
            lambda run: _remove_artifact_key(run, "free_running"),
            "top-level keys",
        ),
        (
            "missing_dynamic_kernel",
            lambda run: _mutate_artifact(
                run,
                "conditional_dynamic_by_history",
                {
                    "zero": {"trained_low": torch.ones(2, 1, 1)},
                    "matched_observed": {"trained_low": torch.ones(2, 1, 1)},
                    "standard_train_rate": {"trained_low": torch.ones(2, 1, 1)},
                },
            ),
            "dynamic history",
        ),
        (
            "malformed_cell_ids",
            lambda run: _mutate_artifact(run, "cell_ids", ("cell-a", 7)),
            "cell_ids",
        ),
        (
            "malformed_lag_order",
            lambda run: _mutate_artifact(run, "lag_order", 7),
            "lag order",
        ),
        (
            "missing_static_kernel",
            lambda run: _mutate_artifact_history_kernel(
                run,
                "conditional_static_by_history",
                "matched_observed",
                "initialized",
                None,
            ),
            "static history",
        ),
        (
            "extra_free_running_kernel",
            lambda run: _mutate_artifact_free_kernel(
                run,
                "poison",
                torch.ones(2, 1, 2),
            ),
            "free_running",
        ),
        (
            "missing_free_running_kernel",
            lambda run: _mutate_artifact_free_kernel(
                run,
                "dynamic_initialized_high",
                None,
            ),
            "free_running",
        ),
        (
            "identity_shape_mismatch",
            lambda run: _mutate_artifact_history_kernel(
                run,
                "conditional_dynamic_by_history",
                "matched_observed",
                "trained_low",
                torch.ones(1, 1, 1),
            ),
            "kernel shape",
        ),
        (
            "nonfinite_kernel",
            lambda run: _mutate_artifact_history_kernel(
                run,
                "conditional_dynamic_by_history",
                "matched_observed",
                "trained_low",
                torch.full((2, 1, 2), float("nan")),
            ),
            "finite",
        ),
    ],
)
def test_rf_artifact_loader_rejects_malformed_artifacts(
    tmp_path: Path,
    name: str,
    mutate: Callable[[Path], None],
    message: str,
) -> None:
    # Given
    run = tmp_path / name
    _write_run(run, mode="type_aware", seed=1, budget=1)
    mutate(run)

    # When / Then
    with pytest.raises(TypePriorComparisonError, match=message):
        load_comparable_grid([run])
