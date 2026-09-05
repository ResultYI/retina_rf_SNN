from __future__ import annotations

import json

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import SchottdorfRecording, mc_pc_recordings
from data.schottdorf_lee_multirecording import (
    load_schottdorf_movie_drive,
)
from evaluation.mechanistic_retina.schottdorf_baseline_cell import (
    evaluate_baseline_cell,
)
from evaluation.mechanistic_retina.schottdorf_baseline_reporting import (
    summarize_baselines,
    write_cell_csv,
)
from evaluation.mechanistic_retina.schottdorf_baseline_types import (
    SchottdorfBaselineRunConfig,
    SchottdorfBaselineRunError,
    SchottdorfBaselineRunResult,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    require_unchanged_source,
    sha256_file,
)


_HISTORY_LAGS = 4
_SOURCE_SCHEMA = "schottdorf_lee_2021_macaque_cellwise_canonical_v1"


def run_schottdorf_prediction_baselines(
    config: SchottdorfBaselineRunConfig,
) -> SchottdorfBaselineRunResult:
    _verify_output_boundary(config)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise SchottdorfBaselineRunError("baseline output directory must be empty")
    source_path = config.retinal_artifact_dir / "results.json"
    source_sha256 = sha256_file(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _verify_source_summary(source)
    adapter = SchottdorfAdapterConfig(**source["adapter_config"])
    available = mc_pc_recordings(config.repository_dir / "data")
    grouped = _group_recordings(available)
    source_cells = {cell["cell_id"]: cell for cell in source["cells"]}
    selected_ids = tuple(source_cells) if config.cell_ids is None else config.cell_ids
    if set(selected_ids) - set(source_cells):
        raise SchottdorfBaselineRunError("requested cell is absent from retinal artifact")
    movie_sha256 = sha256_file(config.movie_path)
    if movie_sha256 != source["source_sha256"][config.movie_path.name]:
        raise SchottdorfBaselineRunError("movie/source artifact hash mismatch")
    movie = load_schottdorf_movie_drive(config.movie_path, adapter)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cells = [
        evaluate_baseline_cell(
            config,
            adapter,
            movie,
            grouped[cell_id],
            source_cells[cell_id],
        )
        for cell_id in selected_ids
    ]
    require_unchanged_source(config.movie_path, movie_sha256)
    require_unchanged_source(source_path, source_sha256)
    summary = summarize_baselines(cells)
    payload = {
        "schema": "schottdorf_lee_2021_macaque_matched_prediction_baselines_v1",
        "dataset": "Schottdorf and Lee 2021, G-Node 10.12751/g-node.xage77",
        "source_retinal_artifact": str(config.retinal_artifact_dir.resolve()),
        "source_retinal_results_sha256": source_sha256,
        "cell_count": len(cells),
        "recording_count": sum(cell["recording_count"] for cell in cells),
        "input_representation": source["input_representation"],
        "likelihood": "Bernoulli event per native 150 Hz stimulus frame",
        "split": "identical held-out contiguous temporal segments",
        "feature_contract": {
            "stimulus": "current_and_past_l_plus_m_luminance",
            "stimulus_features": movie.cone_positions_degs.shape[0],
            "stimulus_lags": 16,
            "history": "strictly_past_same_cell_spike_events",
            "history_lags": _HISTORY_LAGS,
        },
        "glm_training_contract": {
            "solver": "LBFGS_strong_wolfe",
            "maximum_iterations": config.glm_max_iterations,
            "regularization": (
                "fixed_a_priori_L2_on_non_bias_stimulus_and_history_weights"
            ),
            "l2_penalty": 1e-4,
            "regularization_selected_using_validation": False,
            "validation_used_for_fit_or_selection": False,
            "convergence_definition": (
                "finite gradients and either max|gradient| <= 1e-4 or LBFGS "
                "tolerance termination before iteration/evaluation budgets; "
                "strict-gradient count is reported separately"
            ),
        },
        "comparison_scope": (
            "fixed-a-priori L2 causal GLM prediction comparison on identical "
            "data/evaluation; not matched capacity"
        ),
        "fairness_checks": {
            "same_cells_split_valid_bins_target_and_stimulus": True,
            "constant_rate_estimated_from_training_only": True,
            "glm_fit_uses_training_only": True,
            "glm_stimulus_is_current_or_past_only": True,
            "glm_spike_history_is_strictly_past_same_cell_only": True,
            "validation_used_for_hyperparameters_or_checkpoint_selection": False,
            "retinal_checkpoint_modified_or_retrained": False,
            "future_information_used": False,
            "glm_all_fits_converged": summary["glm_converged_cells"]
            == len(cells),
        },
        "fairness_anomalies": (
            []
            if summary["glm_converged_cells"] == len(cells)
            else [
                f"{len(cells) - int(summary['glm_converged_cells'])} GLM fits "
                "did not reach the declared gradient convergence threshold"
            ]
        ),
        **summary,
        "cells": cells,
    }
    (config.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_cell_csv(config.output_dir / "per_cell_results.csv", cells)
    means = payload["mean_validation_nll_across_cells"]
    return SchottdorfBaselineRunResult(
        artifact_dir=config.output_dir,
        cell_count=len(cells),
        constant_rate_nll=float(means["constant_rate"]),
        glm_nll=float(means["glm"]),
        retinal_nll=float(means["retinal"]),
    )


def _verify_source_summary(source) -> None:
    if (
        source["schema"] != _SOURCE_SCHEMA
        or source["cell_count"] != 22
        or source["recording_count"] != 37
        or source["input_representation"]
        != "macaque_experiment_calibrated_l_plus_m_weber_drive_v1"
    ):
        raise SchottdorfBaselineRunError("source retinal artifact contract mismatch")


def _verify_output_boundary(config: SchottdorfBaselineRunConfig) -> None:
    output = config.output_dir.resolve()
    source = config.retinal_artifact_dir.resolve()
    if output == source or source in output.parents:
        raise SchottdorfBaselineRunError(
            "baseline output must not overwrite or contain the source artifact"
        )


def _group_recordings(
    recordings: tuple[SchottdorfRecording, ...],
) -> dict[str, tuple[SchottdorfRecording, ...]]:
    grouped: dict[str, list[SchottdorfRecording]] = {}
    for recording in recordings:
        grouped.setdefault(recording.cell_id, []).append(recording)
    return {cell_id: tuple(items) for cell_id, items in grouped.items()}


__all__ = [
    "SchottdorfBaselineRunConfig",
    "SchottdorfBaselineRunError",
    "SchottdorfBaselineRunResult",
    "run_schottdorf_prediction_baselines",
]
