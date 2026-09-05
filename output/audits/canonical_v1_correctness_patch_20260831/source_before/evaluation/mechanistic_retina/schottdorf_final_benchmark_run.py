from __future__ import annotations

import json

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import SchottdorfRecording, mc_pc_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.schottdorf_final_benchmark_cell import (
    evaluate_final_benchmark_cell,
)
from evaluation.mechanistic_retina.schottdorf_final_benchmark_reporting import (
    summarize_final_benchmark,
    write_final_cell_csv,
)
from evaluation.mechanistic_retina.schottdorf_final_benchmark_types import (
    FinalBenchmarkConfig,
    FinalBenchmarkError,
    FinalBenchmarkResult,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    require_unchanged_source,
    sha256_file,
)
from evaluation.json_types import JsonValue


def run_final_prediction_benchmark(config: FinalBenchmarkConfig) -> FinalBenchmarkResult:
    _verify_output(config)
    retinal_path = config.retinal_artifact_dir / "results.json"
    glm_path = config.glm_artifact_dir / "results.json"
    retinal_sha = sha256_file(retinal_path)
    glm_sha = sha256_file(glm_path)
    retinal_source = _load_json(retinal_path)
    glm_source = _load_json(glm_path)
    _verify_sources(retinal_source, glm_source)
    adapter = SchottdorfAdapterConfig(**retinal_source["adapter_config"])
    grouped = _group_recordings(mc_pc_recordings(config.repository_dir / "data"))
    retinal_cells = {cell["cell_id"]: cell for cell in retinal_source["cells"]}
    glm_cells = {cell["cell_id"]: cell for cell in glm_source["cells"]}
    selected = tuple(retinal_cells) if config.cell_ids is None else config.cell_ids
    if set(selected) - set(retinal_cells) or set(selected) - set(glm_cells):
        raise FinalBenchmarkError("requested cell is absent from source artifacts")
    movie_sha = sha256_file(config.movie_path)
    if movie_sha != retinal_source["source_sha256"][config.movie_path.name]:
        raise FinalBenchmarkError("movie/source artifact hash mismatch")
    movie = load_schottdorf_movie_drive(config.movie_path, adapter)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for index, cell_id in enumerate(selected, start=1):
        print(f"[{index:02d}/{len(selected):02d}] {cell_id}", flush=True)
        cells.append(
            evaluate_final_benchmark_cell(
                config,
                adapter,
                movie,
                grouped[cell_id],
                retinal_cells[cell_id],
                glm_cells[cell_id],
            )
        )
    require_unchanged_source(config.movie_path, movie_sha)
    require_unchanged_source(retinal_path, retinal_sha)
    require_unchanged_source(glm_path, glm_sha)
    summary = summarize_final_benchmark(cells)
    finite = summary["neural_finite_gradient_cells"] == len(cells)
    payload = {
        "schema": "schottdorf_lee_2021_final_fair_prediction_benchmark_revision4_v1",
        "dataset": "Schottdorf and Lee 2021, G-Node 10.12751/g-node.xage77",
        "cell_count": len(cells),
        "recording_count": sum(cell["recording_count"] for cell in cells),
        "source_retinal_artifact": str(config.retinal_artifact_dir.resolve()),
        "source_retinal_results_sha256": retinal_sha,
        "source_glm_artifact": str(config.glm_artifact_dir.resolve()),
        "source_glm_results_sha256": glm_sha,
        "input_representation": retinal_source["input_representation"],
        "split": "identical held-out contiguous temporal segments",
        "likelihood": "Bernoulli event per native 150 Hz stimulus frame",
        "feature_contract": {
            "stimulus": "current_and_past_l_plus_m_luminance",
            "stimulus_features": movie.cone_positions_degs.shape[0],
            "stimulus_lags": 16,
            "history": "strictly_past_same_cell_spike_events",
            "history_lags": 4,
        },
        "models": {
            "constant": "per-cell training-segment event rate",
            "glm": "fixed-L2 causal per-cell stimulus-history GLM",
            "neural": "compact causal Graph-TCN, width chosen nearest frozen retinal total parameters",
            "retinal": "frozen Canonical V1 revision 4 checkpoint replay",
        },
        "neural_training_contract": {
            "optimizer": "AdamW",
            "maximum_steps": config.neural_maximum_steps,
            "patience": config.neural_patience,
            "learning_rate": config.neural_learning_rate,
            "weight_decay": config.neural_weight_decay,
            "selection_metric": "training Bernoulli NLL only",
            "validation_used_for_fit_or_selection": False,
        },
        "fairness_checks": {
            "same_cells_split_valid_bins_target_and_stimulus": True,
            "constant_rate_estimated_from_training_only": True,
            "glm_fit_uses_training_only": True,
            "neural_fit_uses_training_only": True,
            "stimulus_current_or_past_only": True,
            "spike_history_strictly_past_same_cell_only": True,
            "validation_used_for_fit_or_selection": False,
            "retinal_checkpoint_modified_or_retrained": False,
            "future_information_used": False,
        },
        "fairness_anomalies": [] if finite else ["non-finite neural gradient detected"],
        **summary,
        "cells": cells,
    }
    (config.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_final_cell_csv(config.output_dir / "per_cell_results.csv", cells)
    return FinalBenchmarkResult(
        artifact_dir=config.output_dir,
        cell_count=len(cells),
        mean_validation_nll={
            key: float(value)
            for key, value in payload["mean_validation_nll_across_cells"].items()
        },
    )


def _verify_sources(
    retinal: dict[str, JsonValue],
    glm: dict[str, JsonValue],
) -> None:
    if (
        retinal["schema"] != "schottdorf_lee_2021_macaque_cellwise_canonical_v1"
        or retinal["model_revision"] != 4
        or retinal["cell_count"] != 22
        or retinal["recording_count"] != 37
        or glm["schema"] != "schottdorf_lee_2021_macaque_matched_prediction_baselines_v1"
        or glm["cell_count"] != 22
        or glm["recording_count"] != 37
    ):
        raise FinalBenchmarkError("source artifact contract mismatch")


def _verify_output(config: FinalBenchmarkConfig) -> None:
    output = config.output_dir.resolve()
    sources = (
        config.retinal_artifact_dir.resolve(),
        config.glm_artifact_dir.resolve(),
    )
    if any(
        output == source or source in output.parents or output in source.parents
        for source in sources
    ):
        raise FinalBenchmarkError(
            "output must be outside frozen source artifact trees"
        )
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FinalBenchmarkError("final benchmark output directory must be empty")


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _group_recordings(recordings: tuple[SchottdorfRecording, ...]):
    grouped: dict[str, list[SchottdorfRecording]] = {}
    for recording in recordings:
        grouped.setdefault(recording.cell_id, []).append(recording)
    return {cell_id: tuple(items) for cell_id, items in grouped.items()}


__all__ = ["run_final_prediction_benchmark"]
