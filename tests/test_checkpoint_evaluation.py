from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import h5py

from data.dataset import fit_log_cone_stats, save_log_cone_stats
from scripts.evaluate_checkpoint import main
from training.hybrid import TrainingStage
from training.stage1 import Stage1BuildConfig, build_stage1_components


def _write_export(path: Path, offset: float) -> np.ndarray:
    time_steps = 12
    cone_count = 8
    time = np.arange(time_steps, dtype=np.float32)[:, None]
    cone = np.arange(cone_count, dtype=np.float32)[None, :]
    response = 20.0 + offset + 0.2 * time + 0.1 * cone
    positions = np.column_stack(
        (np.arange(cone_count, dtype=np.float32) * 0.1, np.zeros(cone_count))
    ).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.attrs["eccentricity_deg"] = 0.0
        handle.create_dataset("cone_response_achromatic", data=response)
        handle.create_dataset("cone_xy_deg", data=positions)
        handle.create_dataset("cone_type", data=np.zeros(cone_count, dtype=np.uint8))
        handle.create_dataset(
            "time_axis_seconds",
            data=np.arange(time_steps, dtype=np.float64) * 0.05,
        )
        handle.create_dataset(
            "eye_movement_xy_deg",
            data=np.zeros((time_steps, 2), dtype=np.float32),
        )
        handle.create_dataset(
            "response_shape_time_cone",
            data=np.asarray((time_steps, cone_count), dtype=np.int64),
        )
        handle.create_dataset(
            "format_version",
            data=np.frombuffer(b"retina-snn-cone-response-v1", dtype=np.uint8),
        )
        handle.create_dataset(
            "response_units",
            data=np.frombuffer(b"isomerizations_per_integration_time", dtype=np.uint8),
        )
    return positions


def test_checkpoint_evaluation_cli_writes_one_evidence_bundle(tmp_path: Path) -> None:
    # Given
    train_h5 = tmp_path / "train.h5"
    eval_h5 = tmp_path / "eval.h5"
    positions = _write_export(train_h5, 0.0)
    _write_export(eval_h5, 0.3)
    mean, scale = fit_log_cone_stats((train_h5,))
    stats = tmp_path / "normalization_stats.npz"
    save_log_cone_stats(stats, mean, scale)
    components = build_stage1_components(
        positions,
        Stage1BuildConfig(dt_ms=50.0, horizon_count=1),
    )
    with torch.no_grad():
        for parameter in components.decoder.parameters():
            parameter.fill_(0.1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 2,
            "step": 11,
            "core": components.core.state_dict(),
            "decoder": components.decoder.state_dict(),
            "optimizer": {},
            "stage": TrainingStage.CORE_FINETUNE.value,
        },
        checkpoint,
    )
    output_dir = tmp_path / "evaluation"

    # When
    exit_code = main(
        (
            "--checkpoint",
            str(checkpoint),
            "--normalization-stats",
            str(stats),
            "--train-h5",
            str(train_h5),
            "--eval-h5",
            str(eval_h5),
            "--output-dir",
            str(output_dir),
            "--input-steps",
            "3",
            "--horizons",
            "1",
            "--batch-size",
            "2",
            "--rf-sample-count",
            "2",
            "--glm-max-steps",
            "1",
        )
    )
    payload = json.loads((output_dir / "evaluation_summary.json").read_text())

    # Then
    assert exit_code == 0
    assert payload["checkpoint"]["stage"] == "core_finetune"
    assert payload["checkpoint"]["epoch"] == 2
    assert set(payload["prediction"]) >= {
        "model_mse_fine",
        "model_mse_coarse",
        "best_baseline_mse_fine",
        "best_baseline_mse_coarse",
        "skill_fine",
        "skill_coarse",
    }
    assert len(payload["population_usage"]) == 3
    assert len(payload["population_ablation"]) == 3
    assert len(payload["temporal_probes"]) == 24
    assert len(payload["rf_probes"]) == 6
    assert payload["humret"]["status"] == "not_run"
    assert payload["parameter_audit"]
    assert (output_dir / "rf_probes.npz").is_file()
