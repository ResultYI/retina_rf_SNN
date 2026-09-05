from __future__ import annotations

from dataclasses import asdict
import json
import re

from data.schottdorf_lee_catalog import SchottdorfRecording, mc_pc_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.schottdorf_multirecording_fit import (
    fit_schottdorf_cell,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    group_summary,
    md5_file,
    require_unchanged_source,
    sha256_file,
    stable_cell_ids,
    write_cell_csv,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_types import (
    CellFitRecord,
    SchottdorfMultiRunConfig,
    SchottdorfMultiRunError,
    SchottdorfMultiRunResult,
    SourceLineageError,
)
from models.mechanistic_retina.contracts import MECHANISTIC_MODEL_REVISION


def run_schottdorf_multirecording_training(
    config: SchottdorfMultiRunConfig,
) -> SchottdorfMultiRunResult:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise SchottdorfMultiRunError("multi-recording output directory must be empty")
    available = mc_pc_recordings(config.repository_dir / "data")
    recordings = available
    if config.recording_ids is not None:
        requested = set(config.recording_ids)
        recordings = tuple(item for item in available if item.recording_id in requested)
        found = {item.recording_id for item in recordings}
        if found != requested:
            raise SchottdorfMultiRunError(
                f"unknown or unsupported recording ids: {sorted(requested - found)}"
            )
    movie_sha256 = sha256_file(config.movie_path)
    annex_pointer_path = config.repository_dir / "stimuli" / "1x10_256.mpg"
    lineage_sources = {
        "data/Cell List.docx": config.repository_dir / "data" / "Cell List.docx",
        "data/CellsList.docx": config.repository_dir / "data" / "CellsList.docx",
        "README.md": config.repository_dir / "README.md",
        "stimuli/1x10_256.mpg": annex_pointer_path,
    }
    catalog_source_sha256 = {
        name: sha256_file(path) for name, path in lineage_sources.items()
    }
    annex_size, annex_md5 = _annex_contract(annex_pointer_path)
    movie_md5 = md5_file(config.movie_path)
    if config.movie_path.stat().st_size != annex_size or movie_md5 != annex_md5:
        raise SourceLineageError("movie does not match the public git-annex object")
    movie = load_schottdorf_movie_drive(config.movie_path, config.adapter)
    require_unchanged_source(config.movie_path, movie_sha256)
    for name, path in lineage_sources.items():
        require_unchanged_source(path, catalog_source_sha256[name])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cell_recordings = _group_recordings_by_cell(recordings)
    results: list[CellFitRecord] = [
        fit_schottdorf_cell(
            config,
            movie,
            grouped_recordings,
            index,
            movie_sha256=movie_sha256,
            catalog_source_sha256=catalog_source_sha256,
        )
        for index, grouped_recordings in enumerate(cell_recordings)
    ]
    stable_cells = stable_cell_ids(results)
    payload = {
        "schema": "schottdorf_lee_2021_macaque_cellwise_canonical_v1",
        "model_class": "physiology-constrained recurrent/state-space retinal point-process model",
        "model_revision": MECHANISTIC_MODEL_REVISION,
        "dataset": "Schottdorf and Lee 2021, G-Node 10.12751/g-node.xage77",
        "species": "Macaca fascicularis",
        "macaque_primary_biological_target": True,
        "recording_count": len(recordings),
        "cell_count": len(results),
        "population_locality_constructed": False,
        "cellwise_geometry": "single-cell origin only; no cross-recording geometry",
        "training_target": "measured_macaque_spike_events_only",
        "fresh_initialization_per_cell": True,
        "fresh_optimizer_per_cell": True,
        "held_out_validation": "contiguous live-movie temporal segments disjoint from training",
        "input_representation": (
            "macaque_experiment_calibrated_l_plus_m_weber_drive_v1"
        ),
        "pc_prediction_scope": (
            "luminance-drive prediction only; chromatic opponency is not resolved"
        ),
        "catalog_quality_filter": "none; all public MC/PC recordings retained",
        "catalog_lineage": {
            "recording_rows": "Cell List.docx plus supplemental rows in README.md",
            "recording_kind_override": (
                "CellsList.docx laboratory record identifies ER68U4/184 as 6x1min"
            ),
        },
        "movie_annex_contract": {
            "size_bytes": annex_size,
            "md5": annex_md5,
        },
        "adapter_config": asdict(config.adapter),
        "training_contract": {
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "base_seed": config.seed,
            "likelihood": "Bernoulli event per native 150 Hz stimulus frame",
        },
        "stable_prediction_definition": (
            "the pooled multi-recording cell fit has finite lower trained NLL and "
            "all major parameter groups actually updated"
        ),
        "stable_cell_ids": sorted(stable_cells),
        "stable_cell_count": len(stable_cells),
        "improved_cell_count": sum(result["prediction_improved"] for result in results),
        "group_summary": group_summary(results),
        "source_sha256": {config.movie_path.name: movie_sha256} | catalog_source_sha256,
        "cells": results,
    }
    (config.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_cell_csv(config.output_dir / "per_cell_results.csv", results)
    return SchottdorfMultiRunResult(
        artifact_dir=config.output_dir,
        recording_count=len(recordings),
        cell_count=payload["cell_count"],
        stable_cell_count=len(stable_cells),
        improved_cell_count=payload["improved_cell_count"],
    )


def _group_recordings_by_cell(
    recordings: tuple[SchottdorfRecording, ...],
) -> tuple[tuple[SchottdorfRecording, ...], ...]:
    grouped: dict[str, list[SchottdorfRecording]] = {}
    for recording in recordings:
        grouped.setdefault(recording.cell_id, []).append(recording)
    return tuple(tuple(items) for items in grouped.values())


def _annex_contract(pointer_path: Path) -> tuple[int, str]:
    pointer = pointer_path.read_text(encoding="utf-8").strip()
    match = re.search(r"MD5-s(\d+)--([0-9a-f]{32})", pointer)
    if match is None:
        raise SourceLineageError("public movie git-annex pointer is invalid")
    return int(match.group(1)), match.group(2)


__all__ = [
    "SchottdorfMultiRunConfig",
    "SchottdorfMultiRunResult",
    "run_schottdorf_multirecording_training",
]
