from __future__ import annotations

from collections.abc import Mapping
import glob
from pathlib import Path

import torch

from data.cone_response import DataContractError, load_cone_response
from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.pathway_decomposition import (
    effective_pathway_rf,
    pathway_output_sensitivity,
)
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.phase1_data import AssetRequest, prepare_phase1_assets
from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.stages import MechanisticSeedData


def run_natural_exploratory(
    model: MechanisticGraphTemporalRetina,
    data: MechanisticSeedData,
    pattern: str,
) -> Mapping[str, JsonValue]:
    paths = tuple(Path(value) for value in sorted(glob.glob(pattern, recursive=True)))
    selected = _compatible_path(paths, data)
    if selected is None:
        return {
            "status": "NATURAL_CONE_RESPONSE_DATA_NOT_AVAILABLE",
            "matched_files": len(paths),
            "reason": "no existing HDF5 matched frozen 29-cone geometry and >=16 time bins",
        }
    export = load_cone_response(selected)
    assets = prepare_phase1_assets(AssetRequest(240819, 56, 19, 91001, 16))
    normalized = (
        torch.from_numpy(export.response).float()
        - torch.from_numpy(assets.normalization.input_mean).float()
    ) / torch.from_numpy(assets.normalization.input_std).float()
    cones = normalized[:320].unsqueeze(0)
    history = torch.zeros((1, cones.shape[1], len(data.cell_ids)), dtype=cones.dtype)
    with torch.no_grad():
        output = model.forward_sequence(cones, observed_counts=history)
    sensitivities = pathway_output_sensitivity(
        model, cones, history, time_index=cones.shape[1] - 1
    )
    pathways = effective_pathway_rf(model, cones, history)
    total_rf = effective_rf(model, cones, history)
    h1_rf = total_rf - _clamped_rf(model, cones, history, PathwayClamp.H1)
    clamps = {
        clamp.value: float(
            (
                output.logits
                - model.forward_sequence(
                    cones, observed_counts=history, clamps=frozenset({clamp})
                ).logits
            )
            .abs()
            .mean()
        )
        for clamp in PathwayClamp
    }
    currents = {
        "BC-sustained": output.bc_sustained_current,
        "BC-transient": output.bc_transient_current,
        "AC-local": output.amacrine_local_current,
        "AC-transient": output.amacrine_transient_current,
    }
    per_pathway = {
        name: {
            "activation_or_current_mean_abs": float(current.abs().mean()),
            "current_contribution_mean_abs": float(current.abs().mean()),
            "output_sensitivity_mean_abs": float(sensitivities[name].abs().mean()),
            "causal_clamp_logit_mean_abs": clamps[_clamp_name(name)],
            "pathway_rf_norm": float(pathways[name].norm()),
        }
        for name, current in currents.items()
    }
    per_pathway["H1"] = {
        "activation_or_current_mean_abs": float(output.h1_state.abs().mean()),
        "current_contribution_mean_abs": float(
            (
                output.total_current
                - model.forward_sequence(
                    cones,
                    observed_counts=history,
                    clamps=frozenset({PathwayClamp.H1}),
                ).total_current
            ).abs().mean()
        ),
        "output_sensitivity_mean_abs": float(
            torch.stack(tuple(sensitivities.values())).abs().mean()
        ),
        "causal_clamp_logit_mean_abs": clamps[PathwayClamp.H1.value],
        "pathway_rf_norm": float(h1_rf.norm()),
    }
    return {
        "status": "EXPLORATORY_MODEL_ANALYSIS",
        "source_path": str(selected),
        "time_bins": cones.shape[1],
        "cone_count": cones.shape[2],
        "history_policy": "all-zero because natural cone HDF5 has no paired RGC spikes",
        "states": {
            "h1_mean_abs": float(output.h1_state.abs().mean()),
            "bc_sustained_mean_abs": float(output.on_sustained_state.abs().mean()),
            "bc_transient_mean_abs": float(output.on_transient_state.abs().mean()),
            "ac_local_mean_abs": float(output.amacrine_local_state.abs().mean()),
            "ac_transient_mean_abs": float(output.amacrine_transient_state.abs().mean()),
            "rgc_membrane_mean_abs": float(output.rgc_membrane.abs().mean()),
            "rgc_adaptation_mean_abs": float(output.rgc_adaptation.abs().mean()),
            "rgc_history_mean_abs": float(output.rgc_history_state.abs().mean()),
            "logit_mean": float(output.logits.mean()),
            "spike_probability_mean": float(output.spike_probability.mean()),
        },
        "per_pathway": per_pathway,
        "per_cell": [
            {
                "cell_id": cell_id,
                "total_current_mean_abs": float(output.total_current[..., index].abs().mean()),
                "membrane_mean_abs": float(output.rgc_membrane[..., index].abs().mean()),
                "logit_mean": float(output.logits[..., index].mean()),
                "spike_probability_mean": float(output.spike_probability[..., index].mean()),
            }
            for index, cell_id in enumerate(data.cell_ids)
        ],
        "interpretation_boundary": "model-internal exploratory analysis; not biological causal evidence",
    }


def _compatible_path(paths: tuple[Path, ...], data: MechanisticSeedData) -> Path | None:
    for path in paths:
        try:
            export = load_cone_response(path)
        except (DataContractError, KeyError, OSError, ValueError):
            continue
        if export.response.shape[0] < 16 or export.response.shape[1] != data.cone_positions.shape[0]:
            continue
        positions = torch.from_numpy(export.positions_degs).float()
        if torch.allclose(positions, data.cone_positions.float(), atol=1e-6, rtol=0):
            return path
    return None


def _clamped_rf(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    history: torch.Tensor,
    clamp: PathwayClamp,
) -> torch.Tensor:
    stimulus = cones.detach().clone().requires_grad_(True)
    logits = model.forward_sequence(
        stimulus, observed_counts=history, clamps=frozenset({clamp})
    ).logits[:, -1]
    values = []
    for cell in range(logits.shape[-1]):
        values.append(
            torch.autograd.grad(
                logits[:, cell].sum(), stimulus, retain_graph=cell + 1 < logits.shape[-1]
            )[0][:, -model.config.lag_steps :]
        )
    return torch.stack(values, dim=1).detach()


def _clamp_name(name: str) -> str:
    return {
        "BC-sustained": PathwayClamp.DIRECT_BC_SUSTAINED.value,
        "BC-transient": PathwayClamp.DIRECT_BC_TRANSIENT.value,
        "AC-local": PathwayClamp.AMACRINE_LOCAL.value,
        "AC-transient": PathwayClamp.AMACRINE_TRANSIENT.value,
    }[name]


__all__ = ["run_natural_exploratory"]
