from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from collections.abc import Mapping

from evaluation.mechanistic_retina.direct_metrics import rf_payload
from evaluation.mechanistic_retina.mechanism_diagnosis import LegacyDiagnosis
from evaluation.mechanistic_retina.mechanism_run_types import MechanismRunEvidence
from evaluation.mechanistic_retina.metrics import JsonValue


def diagnosis_payload(diagnosis: LegacyDiagnosis) -> Mapping[str, JsonValue]:
    return {
        "exact_closure": dict(diagnosis.exact_closure),
        "retrained_ablations": [
            {
                "name": run.name.value,
                "validation_ce": run.validation_ce,
                "bias_ce": run.bias_ce,
                "replaceability_ratio": run.replaceability_ratio,
                "rf": dict(run.rf),
            }
            for run in diagnosis.ablations
        ],
        "subspace_overlap": {
            "output_count": diagnosis.overlap.output_count,
            "h1_unique_fraction": diagnosis.overlap.h1_unique_fraction,
            "bc_unique_fraction": diagnosis.overlap.bc_unique_fraction,
            "ac_unique_fraction": diagnosis.overlap.ac_unique_fraction,
            "pairs": [
                {
                    "first": pair.first,
                    "second": pair.second,
                    "maximum_canonical_correlation": pair.maximum_canonical_correlation,
                    "principal_angles_deg": list(pair.principal_angles_deg),
                }
                for pair in diagnosis.overlap.pairs
            ],
        },
        "change_decision": diagnosis.decision,
    }


def run_payload(run: MechanismRunEvidence) -> Mapping[str, JsonValue]:
    return {
        "teacher": run.teacher,
        "model": run.ablation.value,
        "seed": run.seed,
        "validation_ce": run.validation_ce,
        "bias_ce": run.bias_ce,
        "no_h1_ce": run.no_h1_ce,
        "no_ac_ce": run.no_ac_ce,
        "prediction_better_than_bias": run.validation_ce < run.bias_ce,
        "gates": {
            "h1": run.gates.h1,
            "ac_local": run.gates.ac_local,
            "ac_transient": run.gates.ac_transient,
            "history": run.gates.history,
        },
        "rf": dict(rf_payload(run.rf)),
        "pathway_norms": dict(run.pathway_norms),
        "pathway_cosines": dict(run.pathway_cosines),
        "pathway_sum_error": run.pathway_sum_error,
        "gradients_finite": run.training.gradients_finite,
        "checkpoints": [
            {
                "step": point.step,
                "train_ce": point.train_ce,
                "validation_ce": point.validation_ce,
            }
            for point in run.training.checkpoints
        ],
    }


def write_metric_csvs(
    output: Path,
    runs: tuple[MechanismRunEvidence, ...],
) -> None:
    pathway_fields = (
        "phase",
        "teacher",
        "model",
        "seed",
        "pathway",
        "norm",
        "teacher_cosine",
        "gate",
        "validation_ce",
    )
    with output.joinpath("per-pathway-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=pathway_fields)
        writer.writeheader()
        for run in runs:
            for pathway, norm in run.pathway_norms.items():
                writer.writerow(
                    {
                        "phase": run.phase,
                        "teacher": run.teacher,
                        "model": run.ablation.value,
                        "seed": run.seed,
                        "pathway": pathway,
                        "norm": norm,
                        "teacher_cosine": run.pathway_cosines[pathway],
                        "gate": _path_gate(run, pathway),
                        "validation_ce": run.validation_ce,
                    }
                )
    cell_fields = (
        "teacher",
        "model",
        "seed",
        "cell_id",
        "full_cosine",
        "temporal_cosine",
        "exact_resolved",
        "type_polarity_resolved",
    )
    with output.joinpath("per-cell-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields)
        writer.writeheader()
        for run in runs:
            for cell in run.rf.metric.cells:
                writer.writerow(
                    {
                        "teacher": run.teacher,
                        "model": run.ablation.value,
                        "seed": run.seed,
                        "cell_id": cell.cell_id,
                        "full_cosine": cell.full_cosine,
                        "temporal_cosine": cell.temporal_cosine,
                        "exact_resolved": cell.exact_resolved,
                        "type_polarity_resolved": cell.type_polarity_resolved,
                    }
                )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_gate(run: MechanismRunEvidence, pathway: str) -> float:
    values = {
        "H1": run.gates.h1,
        "AC": min(run.gates.ac_local, run.gates.ac_transient),
        "BC": 1.0,
    }
    return values[pathway]


__all__ = ["diagnosis_payload", "run_payload", "sha256_file", "write_metric_csvs"]
