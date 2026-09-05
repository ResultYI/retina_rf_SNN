from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Final

import torch

from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    sha256_file,
)
from evaluation.mechanistic_retina.temporal_center_surround import (
    CenterSurroundProbeConfig,
    TEMPORAL_OFFSETS_MS,
    build_center_surround_probe,
)
from evaluation.mechanistic_retina.temporal_center_surround_reporting import (
    CellTrace,
    MetricRow,
    condition_metric_rows,
    group_metric_rows,
    write_metric_tables,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import build_mechanistic_retina


_MODES: Final = ("normal", "H1_off", "AC_off")
_CLAMPS: Final = {
    "normal": frozenset(),
    "H1_off": frozenset({PathwayClamp.H1}),
    "AC_off": frozenset(
        {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
    ),
}


@dataclass(frozen=True, slots=True)
class FrozenExperimentError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def run_frozen_temporal_center_surround(
    training_dir: Path,
    output_dir: Path,
    probe_config: CenterSurroundProbeConfig = CenterSurroundProbeConfig(),
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FrozenExperimentError("output directory must be empty")
    source_path = training_dir / "results.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_cells = tuple(source["cells"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[MetricRow] = []
    traces: list[CellTrace] = []
    tensor_artifact: dict[str, dict[str, torch.Tensor]] = {}
    checkpoint_hashes: dict[str, str] = {}
    all_unchanged = True
    for source_cell in source_cells:
        cell_id = str(source_cell["cell_id"])
        checkpoint_path = (
            training_dir / "cells" / cell_id.replace("#", "_") / "model-trained.pt"
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config_payload = dict(checkpoint["model_config"])
        config_payload["architecture_mode"] = ArchitectureMode(
            config_payload["architecture_mode"]
        )
        config = MechanisticRetinaConfig(**config_payload)
        model = build_mechanistic_retina(
            config,
            checkpoint["cone_positions_degs"],
            checkpoint["cell_positions_degs"],
            (str(checkpoint["canonical_cell_type"]),),
            (str(checkpoint["polarity"]),),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        state_before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        polarity_sign = 1.0 if str(checkpoint["polarity"]) == "ON" else -1.0
        center_support = model.feature_bank.bc_support[0]
        surround_support = model.feature_bank.ac_support[0] * (1 - center_support)
        probe = build_center_surround_probe(
            center_support,
            surround_support,
            config.dt_ms,
            polarity_sign,
            probe_config,
        )
        history = torch.zeros(
            probe.cone_drive.shape[0], probe.cone_drive.shape[1], 1
        )
        baseline_cones = torch.zeros_like(probe.cone_drive[:1])
        baseline_history = torch.zeros_like(history[:1])
        mode_tensors: dict[str, torch.Tensor] = {}
        probability_by_mode: dict[str, torch.Tensor] = {}
        for mode in _MODES:
            clamps = _CLAMPS[mode]
            with torch.no_grad():
                baseline = model.forward_sequence(
                    baseline_cones, observed_counts=baseline_history, clamps=clamps
                )
                output = model.forward_sequence(
                    probe.cone_drive, observed_counts=history, clamps=clamps
                )
            logit_delta = output.logits[..., 0] - baseline.logits[0, :, 0]
            probability_delta = (
                output.spike_probability[..., 0]
                - baseline.spike_probability[0, :, 0]
            )
            probability_by_mode[mode] = probability_delta
            mode_tensors[f"{mode}_logit"] = output.logits[..., 0]
            mode_tensors[f"{mode}_probability"] = output.spike_probability[..., 0]
            mode_tensors[f"{mode}_logit_delta"] = logit_delta
            mode_tensors[f"{mode}_probability_delta"] = probability_delta
            rows.extend(
                condition_metric_rows(
                    cell_id,
                    f"{source_cell['retinal_class']}_{source_cell['polarity']}",
                    mode,
                    probe,
                    logit_delta,
                    probability_delta,
                    probe_config,
                )
            )
        tensor_artifact[cell_id] = mode_tensors | {
            "cone_drive": probe.cone_drive,
            "time_ms": probe.time_ms,
            "offset_ms": probe.offset_ms,
            "center_support": center_support,
            "surround_support": surround_support,
        }
        group = f"{source_cell['retinal_class']}_{source_cell['polarity']}"
        traces.append(
            CellTrace(cell_id, group, probe.time_ms, probe.names, probability_by_mode)
        )
        unchanged = all(
            torch.equal(state_before[name], value)
            for name, value in model.state_dict().items()
        ) and sha256_file(checkpoint_path) == checkpoint_hash
        all_unchanged = all_unchanged and unchanged
        checkpoint_hashes[cell_id] = checkpoint_hash
    group_rows = group_metric_rows(rows)
    write_metric_tables(output_dir, rows, group_rows)
    torch.save(tensor_artifact, output_dir / "response_tensors.pt")
    from evaluation.mechanistic_retina.temporal_center_surround_plots import save_figures

    save_figures(traces, output_dir / "figures", probe_config)
    payload = {
        "schema": "schottdorf_revision4_frozen_temporal_center_surround",
        "source_training_artifact": str(training_dir.resolve()),
        "source_results_sha256": sha256_file(source_path),
        "checkpoint_sha256": checkpoint_hashes,
        "execution": {
            "training_performed": False,
            "optimizer_created": False,
            "model_state_and_checkpoint_unchanged": all_unchanged,
        },
        "probe": asdict(probe_config)
        | {
            "dt_ms": float(source_cells[0]["native_dt_ms"]),
            "offsets_ms": list(TEMPORAL_OFFSETS_MS),
            "center_definition": "checkpoint feature_bank.bc_support",
            "surround_definition": "checkpoint feature_bank.ac_support excluding bc_support; stimulus annulus, not AC pathway support",
            "drive": "polarity-matched L+M Weber contrast",
            "fractional_event_sampling": "per-bin pulse-overlap fraction",
            "observed_history": "all-zero and identical across conditions",
        },
        "metric_contract": {
            "response": "condition minus same-clamp all-zero-input trace",
            "peak": "signed value at maximum absolute deviation",
            "integral": "signed response sum times dt in seconds",
            "suppression_facilitation": "condition minus center-only within same clamp",
            "onset_offset": "mean probability response in post-event window",
        },
        "cell_count": len(source_cells),
        "condition_count": 7,
        "modes": list(_MODES),
        "rows": [asdict(row) for row in rows],
        "group_summary": group_rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not all_unchanged:
        raise FrozenExperimentError("a frozen checkpoint or model state changed")
    return output_dir


__all__ = [
    "FrozenExperimentError",
    "run_frozen_temporal_center_surround",
]
