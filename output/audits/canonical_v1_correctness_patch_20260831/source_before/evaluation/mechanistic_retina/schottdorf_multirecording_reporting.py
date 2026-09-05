from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch

from evaluation.mechanistic_retina.schottdorf_multirecording_types import (
    CellFitRecord,
    GroupSummaryRow,
    SourceLineageError,
    TensorSummary,
)


def major_parameter_group_updates(updated: tuple[str, ...]) -> dict[str, bool]:
    names = set(updated)
    expected = {
        "pathway_weights": ("bipolar.raw_weights",),
        "pathway_gates": (
            "gates.raw_h1_amplitude",
            "gates.ac_local",
            "gates.ac_transient",
            "gates.history",
        ),
        "bounded_tau": ("h1.raw_tau", "feature_bank.raw_tau", "amacrine.raw_tau"),
        "bounded_explicit_delay": (
            "h1.raw_delay", "feature_bank.raw_delay", "amacrine.raw_delay"
        ),
        "aggregate_bc_ac_gains": ("cell_gains.log_bc", "cell_gains.log_ac"),
        "rgc_bias": ("rgc.response_bias",),
    }
    return {
        group: all(name in names for name in parameters)
        for group, parameters in expected.items()
    }


def stable_cell_ids(results: list[CellFitRecord]) -> set[str]:
    cell_ids = {str(result["cell_id"]) for result in results}
    return {
        cell_id
        for cell_id in cell_ids
        if all(
            bool(result["prediction_improved"]) and _training_is_stable(result)
            for result in results
            if result["cell_id"] == cell_id
        )
    }


def group_summary(
    results: list[CellFitRecord],
) -> dict[str, GroupSummaryRow]:
    summary: dict[str, GroupSummaryRow] = {}
    for retinal_class in ("MC", "PC"):
        for polarity in ("ON", "OFF"):
            group = [
                result
                for result in results
                if result["retinal_class"] == retinal_class
                and result["polarity"] == polarity
            ]
            key = f"{retinal_class}_{polarity}"
            if not group:
                summary[key] = _empty_group()
                continue
            summary[key] = {
                "recordings": sum(int(result["recording_count"]) for result in group),
                "cells": len(group),
                "mean_nll_raw": sum(
                    float(result["validation_nll_raw"]) for result in group
                )
                / len(group),
                "mean_nll_trained": sum(
                    float(result["validation_nll_trained"]) for result in group
                )
                / len(group),
                "improved_cells": sum(
                    bool(result["prediction_improved"]) for result in group
                ),
            }
    return summary


def write_cell_csv(path: Path, results: list[CellFitRecord]) -> None:
    fields = (
        "cell_id",
        "recording_count",
        "recording_ids",
        "recorded_cell_classes",
        "retinal_class",
        "canonical_cell_type",
        "polarity",
        "recording_kinds",
        "biological_trials",
        "train_sequences",
        "validation_sequences",
        "train_valid_bins",
        "validation_valid_bins",
        "native_dt_ms",
        "validation_nll_raw",
        "validation_nll_trained",
        "nll_improvement",
        "prediction_improved",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unchanged_source(path: Path, expected_sha256: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise SourceLineageError(f"source changed during run: {path}")


def tensor_summary(value: torch.Tensor) -> TensorSummary:
    flat = value.detach().flatten().float()
    return {
        "values": flat.tolist(),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
        "norm": float(torch.linalg.vector_norm(flat)),
    }


def _training_is_stable(result: CellFitRecord) -> bool:
    training = result["training"]
    return training["gradients_finite"] and all(
        training["major_parameter_groups_updated"].values()
    )


def _empty_group() -> GroupSummaryRow:
    return {
        "recordings": 0,
        "cells": 0,
        "mean_nll_raw": None,
        "mean_nll_trained": None,
        "improved_cells": 0,
    }


__all__ = [
    "group_summary",
    "major_parameter_group_updates",
    "md5_file",
    "require_unchanged_source",
    "sha256_file",
    "stable_cell_ids",
    "tensor_summary",
    "write_cell_csv",
]
