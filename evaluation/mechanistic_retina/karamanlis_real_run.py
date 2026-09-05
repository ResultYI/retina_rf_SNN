from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from data.karamanlis_2024 import (
    KaramanlisAdapterConfig,
    load_marmoset_imagesequence,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.real_sampled import (
    RealSpikeTrainingRequest,
    fit_real_spike_model,
)


@dataclass(frozen=True, slots=True)
class KaramanlisRealRunConfig:
    session_dir: Path
    output_dir: Path
    steps: int = 50
    learning_rate: float = 0.03
    batch_size: int = 4
    seed: int = 202_603_01
    adapter: KaramanlisAdapterConfig = KaramanlisAdapterConfig()


@dataclass(frozen=True, slots=True)
class KaramanlisRealRunResult:
    artifact_dir: Path
    validation_nll_raw: float
    validation_nll_trained: float


def run_karamanlis_real_training(
    run_config: KaramanlisRealRunConfig,
) -> KaramanlisRealRunResult:
    if run_config.output_dir.exists() and any(run_config.output_dir.iterdir()):
        raise FileExistsError("real-data output directory must be empty")
    data = load_marmoset_imagesequence(run_config.session_dir, run_config.adapter)
    model_config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        dt_ms=data.dt_ms,
    )
    torch.manual_seed(run_config.seed)
    model = build_mechanistic_retina(
        model_config,
        data.cone_positions_degs,
        data.cell_positions_degs,
        data.cell_types,
        data.polarities,
    )
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_base = {
        "schema": "karamanlis_2024_marmoset_canonical_retina",
        "revision": MECHANISTIC_MODEL_REVISION,
        "session_id": data.session_id,
        "model_config": asdict(model_config)
        | {"architecture_mode": model_config.architecture_mode.value},
        "adapter_config": asdict(run_config.adapter)
        | {"cell_selection": run_config.adapter.cell_selection.value},
        "cell_ids": data.cell_ids,
        "cell_types": data.cell_types,
        "polarities": data.polarities,
        "cell_positions_degs": data.cell_positions_degs,
        "cone_positions_degs": data.cone_positions_degs,
    }
    torch.save(
        checkpoint_base | {"stage": "raw", "model": model.state_dict()},
        run_config.output_dir / "model-raw.pt",
    )
    training = fit_real_spike_model(
        RealSpikeTrainingRequest(
            model=model,
            train=data.train,
            validation=data.validation,
            steps=run_config.steps,
            learning_rate=run_config.learning_rate,
            batch_size=run_config.batch_size,
            seed=run_config.seed,
        )
    )
    torch.save(
        checkpoint_base | {"stage": "trained", "model": model.state_dict()},
        run_config.output_dir / "model-trained.pt",
    )
    exp_path = run_config.session_dir / "expdata.mat"
    stimulus_path = run_config.session_dir / "imagesequence_data.mat"
    per_cell = [
        {
            "id": cell_id,
            "type": cell_type,
            "polarity": polarity,
            "nll_raw": nll_raw,
            "nll_trained": nll_trained,
            "nll_improvement": nll_raw - nll_trained,
            "mean_probability_raw": probability_raw,
            "mean_probability_trained": probability_trained,
            "observed_event_rate": event_rate,
        }
        for (
            cell_id,
            cell_type,
            polarity,
            nll_raw,
            nll_trained,
            probability_raw,
            probability_trained,
            event_rate,
        ) in zip(
            data.cell_ids,
            data.cell_types,
            data.polarities,
            training.per_cell_nll_raw,
            training.per_cell_nll_trained,
            training.per_cell_probability_raw,
            training.per_cell_probability_trained,
            training.per_cell_event_rate,
            strict=True,
        )
    ]
    improvements = torch.tensor(
        tuple(row["nll_improvement"] for row in per_cell), dtype=torch.float64
    )
    by_cell_class = {}
    for class_name in tuple(
        dict.fromkeys(
            f"{polarity} {cell_type}"
            for polarity, cell_type in zip(
                data.polarities, data.cell_types, strict=True
            )
        )
    ):
        rows = [
            row
            for row in per_cell
            if f"{row['polarity']} {row['type']}" == class_name
        ]
        by_cell_class[class_name] = {
            "cell_count": len(rows),
            "mean_nll_raw": sum(row["nll_raw"] for row in rows) / len(rows),
            "mean_nll_trained": sum(row["nll_trained"] for row in rows) / len(rows),
            "mean_nll_improvement": sum(row["nll_improvement"] for row in rows)
            / len(rows),
            "improved_cell_fraction": sum(
                row["nll_improvement"] > 0 for row in rows
            )
            / len(rows),
        }
    payload = {
        "model_class": "physiology-constrained recurrent/state-space retinal point-process model",
        "model_revision": MECHANISTIC_MODEL_REVISION,
        "causal_contract": model.config.causal_contract,
        "topology": "Cone->H1->shared BC encoder; narrow BC->RGC; broad BC->AC->RGC; RGC->spike likelihood",
        "dataset": "G-Node 10.12751/g-node.ejk8kx / OpenRetina gollisch_lab/karamanlis_2024",
        "species": "marmoset",
        "session_id": data.session_id,
        "training_target": "measured_marmoset_spike_events_only",
        "fresh_initialization": True,
        "fresh_optimizer": True,
        "native_dt_ms": data.dt_ms,
        "recording_sampling_rate_hz": data.recording_sampling_rate_hz,
        "projector_rate_hz": data.projector_rate_hz,
        "rgc_history_shift_ms": data.dt_ms,
        "tau_units": "ms",
        "explicit_pathway_delay_units": "ms",
        "input_representation": data.input_representation,
        "input_contract": {
            "stimulus": "measured natural-image flash sequence",
            "display_to_drive": "retinally projected achromatic Weber contrast",
            "crop_pixels": data.crop_pixels,
            "pooled_grid": [data.pooled_grid_size, data.pooled_grid_size],
            "retinal_um_per_degree": 200.0,
        },
        "cells": [
            {"id": cell_id, "type": cell_type, "polarity": polarity}
            for cell_id, cell_type, polarity in zip(
                data.cell_ids, data.cell_types, data.polarities, strict=True
            )
        ],
        "splits": {
            "train_sequences": data.train.cone_drive.shape[0],
            "validation_sequences": data.validation.cone_drive.shape[0],
            "train_source_images": len(set(data.train.source_image_ids)),
            "validation_source_images": len(set(data.validation.source_image_ids)),
            "source_disjoint": set(data.train.source_image_ids).isdisjoint(
                data.validation.source_image_ids
            ),
            "time_steps": data.train.cone_drive.shape[1],
        },
        "spike_encoding": {
            "likelihood": "Bernoulli event per native projector frame",
            "observed_event_rate": training.observed_event_rate,
            "multi_spike_bin_fraction": float(
                (data.train.spike_counts > 1).float().mean()
            ),
        },
        "validation_prediction": {
            "nll_raw": training.validation_nll_raw,
            "nll_trained": training.validation_nll_trained,
            "mean_probability_raw": training.mean_probability_raw,
            "mean_probability_trained": training.mean_probability_trained,
            "per_cell": per_cell,
            "by_cell_class": by_cell_class,
            "nll_improvement_distribution": {
                "minimum": float(improvements.min()),
                "quartile_25": float(torch.quantile(improvements, 0.25)),
                "median": float(torch.quantile(improvements, 0.50)),
                "quartile_75": float(torch.quantile(improvements, 0.75)),
                "maximum": float(improvements.max()),
                "mean": float(improvements.mean()),
                "improved_cell_fraction": float((improvements > 0).float().mean()),
            },
        },
        "training": {
            "steps": run_config.steps,
            "learning_rate": run_config.learning_rate,
            "batch_size": run_config.batch_size,
            "seed": run_config.seed,
            "gradients_finite": training.gradients_finite,
            "actually_updated": training.actually_updated,
        },
        "source_sha256": {
            "expdata.mat": _sha256_file(exp_path),
            "imagesequence_data.mat": _sha256_file(stimulus_path),
        },
    }
    (run_config.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return KaramanlisRealRunResult(
        artifact_dir=run_config.output_dir,
        validation_nll_raw=training.validation_nll_raw,
        validation_nll_trained=training.validation_nll_trained,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "KaramanlisRealRunConfig",
    "KaramanlisRealRunResult",
    "run_karamanlis_real_training",
]
