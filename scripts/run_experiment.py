from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.point_process_glm import fit_point_process_glm
from configs.physiology_profiles import human_macaque
from configs.rgc_type_priors import load_type_priors
from evaluation.response_reporting import write_response_report
from evaluation.rf_dynamic import evaluate_dynamic_rf
from evaluation.rf_static import compare_rf_kernels, extract_static_rf
from models.response_snn import build_response_retina_model
from training.response_checkpointing import load_response_checkpoint
from training.response_config import load_response_config
from training.response_data import prepare_response_data, sample_response_batch
from training.response_trainer import ResponseTrainer


class ResponseExperimentError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit known recorded-RGC responses with a stateful retinal SNN."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        _run(args, output)
    except (OSError, RuntimeError, ValueError) as exc:
        (output / "run_status.json").write_text(
            json.dumps(
                {"status": "FAILED_WITH_REPORT", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "final_report_zh.md").write_text(
            f"# RGC 响应拟合失败\n\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise


def _run(args: argparse.Namespace, output: Path) -> None:
    config = load_response_config(args.config)
    torch.manual_seed(config.seed)
    data = prepare_response_data(config.data)
    priors = load_type_priors(
        config.model.type_prior_path,
        required_type_ids=tuple(sorted(set(data.cells.type_ids))),
    )
    spacing = _cone_spacing(data.cone_positions_degs)
    profile = human_macaque(
        dt_ms=data.dt_ms,
        cone_spacing_deg=spacing,
        eccentricity_deg=float(np.mean(data.cells.eccentricities_deg)),
    )
    model = build_response_retina_model(
        torch.as_tensor(data.cone_positions_degs),
        data.cells,
        profile,
        priors,
        support_radius_degs=config.model.support_radius_degs,
        readout_rate_tau_ms=config.model.readout_rate_tau_ms,
        surrogate_slope=config.model.surrogate_slope,
    ).to(torch.device(args.device))
    initial_state = copy.deepcopy(model.state_dict())
    trainer = ResponseTrainer(model, config, data, torch.device(args.device))
    best_path = output / "checkpoint_best_nll.pt"
    last_path = output / "checkpoint_last.pt"
    if args.checkpoint:
        load_response_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=None,
            generator=None,
            fingerprint=data.fingerprint,
            target_kind=data.target_kind.value,
        )
    elif not args.diagnostics_only:
        _train(trainer, data, config, output, best_path, last_path, args.stop_after_steps)
    if not best_path.exists() and not args.checkpoint:
        raise ResponseExperimentError("No best-NLL checkpoint was produced")
    checkpoint = Path(args.checkpoint) if args.checkpoint else best_path
    load_response_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        generator=None,
        fingerprint=data.fingerprint,
        target_kind=data.target_kind.value,
    )
    conditional = trainer.evaluate(data.test)
    free_running = trainer.evaluate(data.test, free_running=True)
    glm = fit_point_process_glm(data, device=torch.device(args.device))
    probe = data.test.cone_response[0:1].to(torch.device(args.device))
    static_rf = extract_static_rf(
        model,
        probe,
        lag_steps=config.evaluation.rf_lag_steps,
    )
    dynamic_rf = evaluate_dynamic_rf(
        model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
    )
    initialized_model = build_response_retina_model(
        torch.as_tensor(data.cone_positions_degs),
        data.cells,
        profile,
        priors,
        support_radius_degs=config.model.support_radius_degs,
        readout_rate_tau_ms=config.model.readout_rate_tau_ms,
        surrogate_slope=config.model.surrogate_slope,
    ).to(torch.device(args.device))
    initialized_model.load_state_dict(initial_state)
    initialized_dynamic_rf = evaluate_dynamic_rf(
        initialized_model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
    )
    teacher_kernel = _teacher_kernel(config.data.test_glob)
    static_reference = (
        None
        if teacher_kernel is None
        else compare_rf_kernels(
            static_rf.kernels,
            torch.as_tensor(
                teacher_kernel,
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
        synthetic=_has_teacher_data(config.data.test_glob),
        checkpoint=str(checkpoint.resolve()),
    )
    torch.save(initial_state, output / "initialized_model_state.pt")
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


def _train(
    trainer: ResponseTrainer,
    data,
    config,
    output: Path,
    best_path: Path,
    last_path: Path,
    stop_after_steps: int | None,
) -> None:
    steps = config.training.max_optimizer_steps
    if stop_after_steps is not None:
        steps = min(steps, stop_after_steps)
    log_path = output / "training.jsonl"
    for _ in range(steps):
        batch = sample_response_batch(
            data.train,
            batch_size=config.training.batch_size,
            generator=trainer.sampling_generator,
            device=trainer.device,
        )
        result = trainer.train_step(*batch)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result)) + "\n")
        validate = (
            trainer.optimizer_step % config.training.validation_interval_steps == 0
            or trainer.optimizer_step == steps
        )
        if validate:
            metrics = trainer.evaluate(data.validation)
            trainer.save(last_path)
            if metrics.nll < trainer.best_nll:
                trainer.best_nll = metrics.nll
                trainer.save(best_path)


def _cone_spacing(positions: np.ndarray) -> float:
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    )
    distances[distances == 0] = np.inf
    return float(np.median(distances.min(axis=1)))


def _has_teacher_data(pattern: str) -> bool:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return False
    with h5py.File(paths[0], "r") as handle:
        return "teacher" in handle


def _teacher_kernel(pattern: str) -> np.ndarray | None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    with h5py.File(paths[0], "r") as handle:
        if "teacher/static_kernel" not in handle:
            return None
        return np.asarray(handle["teacher/static_kernel"][()], dtype=np.float32)


if __name__ == "__main__":
    main()
