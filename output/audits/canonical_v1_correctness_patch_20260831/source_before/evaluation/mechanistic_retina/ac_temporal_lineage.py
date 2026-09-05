from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from evaluation.mechanistic_retina.ac_circuit_inputs import CheckpointPayload
from evaluation.mechanistic_retina.ac_circuit_lineage import file_sha256
from evaluation.mechanistic_retina.ac_circuit_support import JsonValue
from evaluation.mechanistic_retina.ac_temporal_probe import TemporalProbe
from evaluation.mechanistic_retina.spike_banks import tensor_sha256
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


TEMPORAL_RF_ARTIFACT_SCHEMA: Final = "ac-temporal-circuit-perturbation-rf-v1"
TEMPORAL_RF_ESTIMAND: Final = (
    "final-bin conditional RGC-logit Jacobian over fixed RF lag window"
)
TEMPORAL_RF_DEFINITION: Final = "signed sum of final-bin global RF over cone dimension"


@dataclass(frozen=True, slots=True)
class TemporalRFLineage:
    identity: dict[str, JsonValue]
    source_sha256: Mapping[str, str]


def implementation_source_sha256() -> Mapping[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    analysis_names = (
        "ac_circuit_inputs.py",
        "ac_circuit_lineage.py",
        "ac_circuit_support.py",
        "ac_temporal_lineage.py",
        "ac_temporal_perturbation.py",
        "ac_temporal_probe.py",
        "ac_temporal_support.py",
        "rf_effective.py",
        "spike_banks.py",
    )
    analysis_paths = tuple(
        repo_root / "evaluation/mechanistic_retina" / name for name in analysis_names
    )
    model_paths = tuple(sorted((repo_root / "models/mechanistic_retina").glob("*.py")))
    return {
        path.relative_to(repo_root).as_posix(): file_sha256(path)
        for path in sorted((*analysis_paths, *model_paths))
    }


def temporal_rf_lineage(
    checkpoint: CheckpointPayload,
    checkpoint_path: Path,
    probe: TemporalProbe,
) -> TemporalRFLineage:
    source_hashes = implementation_source_sha256()
    lag_steps = int(checkpoint["model_config"]["lag_steps"])
    cell_count = checkpoint["cell_positions"].shape[0]
    cone_count = checkpoint["cone_positions"].shape[0]
    identity: dict[str, JsonValue] = {
        "rf_estimand": TEMPORAL_RF_ESTIMAND,
        "temporal_rf_definition": TEMPORAL_RF_DEFINITION,
        "lag_order": list(range(lag_steps)),
        "lag_order_semantics": "oldest_to_current",
        "probe_names": list(probe.names),
        "observed_history_context": "all-zero",
        "model_revision": checkpoint["model_revision"],
        "checkpoint_role": checkpoint["role"],
        "dt_ms": probe.dt_ms,
        "cell_order": list(range(cell_count)),
        "cell_types": list(checkpoint["cell_types"]),
        "polarities": list(checkpoint["polarities"]),
        "cell_positions_degs": checkpoint["cell_positions"].tolist(),
        "cone_order": list(range(cone_count)),
        "cone_positions_degs": checkpoint["cone_positions"].tolist(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "probe_sha256": tensor_sha256(probe.cone_response),
        "source_sha256": dict(source_hashes),
    }
    return TemporalRFLineage(identity, source_hashes)


def temporal_timing_contract(
    model: MechanisticGraphTemporalRetina,
    invariance: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return {
        "tau": {
            "meaning": "bounded-learnable decay/time constants in ms",
            "analysis_status": "trained checkpoint values held fixed",
            "unchanged": invariance["tau"],
        },
        "explicit_pathway_delay": {
            "meaning": "bounded-learnable fractional pathway delays in ms",
            "analysis_status": "trained checkpoint values held fixed",
            "unchanged": invariance["delay"],
        },
        "rf_lag_window": {
            "lag_steps": model.config.lag_steps,
            "dt_ms": model.config.dt_ms,
            "learnable": False,
        },
        "rgc_history_shift": {
            "shift_steps": 1,
            "shift_ms": model.config.dt_ms,
            "learnable": False,
        },
    }


__all__ = [
    "TEMPORAL_RF_ARTIFACT_SCHEMA",
    "TEMPORAL_RF_DEFINITION",
    "TEMPORAL_RF_ESTIMAND",
    "implementation_source_sha256",
    "temporal_rf_lineage",
    "temporal_timing_contract",
]
