from __future__ import annotations

import glob
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch

from baselines.point_process_glm import fit_point_process_glm
from evaluation.response_reporting import write_response_report
from evaluation.rf_dynamic import compare_dynamic_rf, evaluate_dynamic_rf
from evaluation.rf_static import compare_rf_kernels, extract_static_rf
from models.response_snn import ResponseRetinaModel
from training.response_config import ResponseExperimentConfig
from training.response_data import PreparedResponseData
from training.response_trainer import ResponseTrainer


def evaluate_and_report_response_experiment(
    output: Path,
    *,
    model: ResponseRetinaModel,
    initialized_model: ResponseRetinaModel,
    trainer: ResponseTrainer,
    data: PreparedResponseData,
    config: ResponseExperimentConfig,
    checkpoint: Path,
) -> None:
    conditional = trainer.evaluate(data.test)
    free_running = trainer.evaluate(data.test, free_running=True)
    glm = fit_point_process_glm(data, device=trainer.device)
    teacher_kernels = _teacher_kernels(config.data.test_glob)
    teacher_dynamic = (
        None
        if teacher_kernels is None
        else (
            torch.as_tensor(teacher_kernels[1], device=model.rgc.support_mask.device),
            torch.as_tensor(teacher_kernels[2], device=model.rgc.support_mask.device),
        )
    )
    probe = data.test.cone_response[0:1].to(trainer.device)
    static_rf = extract_static_rf(
        model,
        probe,
        lag_steps=config.evaluation.rf_lag_steps,
    )
    dynamic_rf = evaluate_dynamic_rf(
        model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
    )
    initialized_dynamic_rf = evaluate_dynamic_rf(
        initialized_model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
    )
    dynamic_comparison = compare_dynamic_rf(
        dynamic_rf,
        initialized_dynamic_rf,
        seed=config.seed,
    )
    static_reference = (
        None
        if teacher_kernels is None
        else compare_rf_kernels(
            static_rf.kernels,
            torch.as_tensor(
                teacher_kernels[0],
                device=static_rf.kernels.device,
                dtype=static_rf.kernels.dtype,
            ),
        )
    )
    write_response_report(
        output,
        conditional=conditional,
        free_running=free_running,
        glm=glm,
        static_rf=static_rf,
        static_reference=static_reference,
        dynamic_rf=dynamic_rf,
        initialized_dynamic_rf=initialized_dynamic_rf,
        dynamic_comparison=dynamic_comparison,
        synthetic=teacher_kernels is not None,
        checkpoint=str(checkpoint.resolve()),
    )
    torch.save(initialized_model.state_dict(), output / "initialized_model_state.pt")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "dataset_fingerprint": data.fingerprint,
                "target_kind": data.target_kind.value,
                "cell_count": len(data.cells.ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _teacher_kernels(
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    with h5py.File(paths[0], "r") as handle:
        keys = (
            "teacher/static_kernel",
            "teacher/context_kernel_low",
            "teacher/context_kernel_high",
        )
        if not all(key in handle for key in keys):
            return None
        return (
            np.asarray(handle[keys[0]][()], dtype=np.float32),
            np.asarray(handle[keys[1]][()], dtype=np.float32),
            np.asarray(handle[keys[2]][()], dtype=np.float32),
        )


__all__ = ["evaluate_and_report_response_experiment"]
