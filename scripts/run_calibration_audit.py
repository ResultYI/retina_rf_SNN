# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "pyyaml", "torch"]
# ///
# How to run: python scripts/run_calibration_audit.py --help

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.point_process_glm import fit_point_process_glm
from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from data.input_identity import validate_experiment_input
from evaluation.response_calibration import (
    LogitCalibrationRequest,
    ThresholdCalibrationRequest,
    fit_logit_calibration,
    fit_threshold_calibration,
)
from evaluation.response_metrics import compute_response_metrics
from evaluation.response_predictions import (
    ResponsePredictionRequest,
    collect_response_predictions,
)
from models.response_snn import build_response_retina_model
from scripts.run_experiment import _build_trainer, _cone_spacing
from training.response_checkpointing import load_response_checkpoint
from training.response_config import load_response_config
from training.response_data import prepare_response_data
from training.response_trainer import ResponseTrainer


@dataclass(frozen=True, slots=True)
class CalibrationAuditDecisions:
    glm: str
    frozen_snn: str
    threshold: str
    long_train: str = "NO_GO"
    final_test: str = "NOT_CONSUMED"


def main() -> None:
    args = parse_args()
    run_audit(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-only GLM and SNN calibration diagnostics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--glm-steps", type=int, default=100)
    parser.add_argument("--calibration-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.001)
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> None:
    config = load_response_config(args.config)
    torch.manual_seed(config.seed)
    device = torch.device(args.device)
    data = prepare_response_data(config.data)
    validate_experiment_input(data.input_identity, data.dt_ms)
    priors = load_type_priors(
        config.model.type_prior_path,
        required_type_ids=tuple(sorted(set(data.cells.type_ids))),
    )
    model = build_response_retina_model(
        torch.as_tensor(data.cone_positions_degs),
        data.cells,
        macaque_photopic(
            dt_ms=data.dt_ms,
            cone_spacing_deg=_cone_spacing(data.cone_positions_degs),
            eccentricity_deg=float(np.mean(data.cells.eccentricities_deg)),
        ),
        priors,
        support_radius_degs=config.model.support_radius_degs,
        readout_rate_tau_ms=config.model.readout_rate_tau_ms,
        surrogate_slope=config.model.surrogate_slope,
        parameter_sharing_mode=config.model.parameter_sharing_mode,
        parameter_sharing_seed=config.seed,
        matched_initialization=config.model.matched_initialization,
        enable_response_bias=config.model.enable_response_bias,
        enable_synaptic_gain=config.model.enable_synaptic_gain,
        enable_direct_readout=config.model.enable_direct_readout,
        synaptic_gain_min=config.model.synaptic_gain_min,
        synaptic_gain_max=config.model.synaptic_gain_max,
        synaptic_gain_init=config.model.synaptic_gain_init,
    ).to(device)
    trainer = _build_trainer(model, config, data, device, run_stage0=False)
    checkpoint = load_response_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=None,
        generator=None,
        fingerprint=data.fingerprint,
        target_kind=data.target_kind.value,
        config=config,
    )
    train = collect_response_predictions(
        ResponsePredictionRequest(
            model,
            data.train,
            config.training.burn_in_steps,
            device,
        )
    )
    validation = collect_response_predictions(
        ResponsePredictionRequest(
            model,
            data.validation,
            config.training.burn_in_steps,
            device,
        )
    )
    uncalibrated = compute_response_metrics(
        validation.logits,
        validation.targets,
        validation.valid_mask,
        data.target_kind,
        trainer.baseline_rates,
    )
    glm = {
        mode: fit_point_process_glm(
            data,
            device=device,
            steps=args.glm_steps,
            burn_in_steps=config.training.burn_in_steps,
            evaluate_test=False,
            mode=mode,
        )
        for mode in ("bias_only", "bias_plus_history", "full_glm")
    }
    intercept = fit_logit_calibration(
        LogitCalibrationRequest(
            train,
            validation,
            data.target_kind,
            trainer.baseline_rates,
            "intercept",
            args.calibration_iterations,
        )
    )
    affine = fit_logit_calibration(
        LogitCalibrationRequest(
            train,
            validation,
            data.target_kind,
            trainer.baseline_rates,
            "affine",
            args.calibration_iterations,
        )
    )
    threshold = fit_threshold_calibration(
        ThresholdCalibrationRequest(
            model,
            train,
            validation,
            data.target_kind,
            trainer.baseline_rates,
            args.calibration_iterations,
            args.tolerance,
        )
    )
    bias_nll = glm["bias_only"].validation_metrics.nll
    full_nll = glm["full_glm"].validation_metrics.nll
    decisions = calibration_decisions(
        constant_rate_nll=uncalibrated.constant_rate_nll,
        bias_only_nll=bias_nll,
        full_glm_nll=full_nll,
        intercept_nll=intercept.validation_metrics.nll,
        affine_nll=affine.validation_metrics.nll,
        threshold_passed=threshold.passed,
        tolerance=args.tolerance,
    )
    result = {
        "contract": {
            "evaluation_split": "validation",
            "final_test": "NOT_CONSUMED",
            "snn_optimizer_steps": 0,
            "checkpoint_optimizer_step": checkpoint.optimizer_step,
            "parameter_sharing_mode": config.model.parameter_sharing_mode,
        },
        "constant_rate": {"validation_nll": uncalibrated.constant_rate_nll},
        "glm": {
            mode: {
                "validation_metrics": asdict(fit.validation_metrics),
                "best_step": fit.best_step,
            }
            for mode, fit in glm.items()
        }
        | {"full_minus_bias_only_nll_improvement": bias_nll - full_nll},
        "frozen_snn": {
            "uncalibrated_validation_metrics": asdict(uncalibrated),
            "intercept": asdict(intercept),
            "positive_affine": asdict(affine),
        },
        "threshold_only": {
            "diagnostic_role": "counterfactual_threshold_only",
            **asdict(threshold),
        },
        "decisions": asdict(decisions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def calibration_decisions(
    *,
    constant_rate_nll: float,
    bias_only_nll: float,
    full_glm_nll: float,
    intercept_nll: float,
    affine_nll: float,
    threshold_passed: bool,
    tolerance: float,
) -> CalibrationAuditDecisions:
    glm_improvement = bias_only_nll - full_glm_nll
    glm = (
        "GLM_CALIBRATION_ONLY"
        if glm_improvement < tolerance
        else "GLM_ADDITIONAL_SIGNAL"
    )
    if affine_nll <= constant_rate_nll - tolerance:
        frozen_snn = "AFFINE_EXCEEDS_RATE_BASELINE"
    elif intercept_nll <= constant_rate_nll + tolerance:
        frozen_snn = "INTERCEPT_REACHES_RATE_BASELINE"
    elif affine_nll <= constant_rate_nll + tolerance:
        frozen_snn = "AFFINE_REACHES_RATE_BASELINE"
    else:
        frozen_snn = "REPRESENTATION_NOT_SUPPORTED"
    threshold = (
        "THRESHOLD_BASELINE_NESTED"
        if threshold_passed
        else "THRESHOLD_BASELINE_NOT_NESTED"
    )
    return CalibrationAuditDecisions(glm, frozen_snn, threshold)


if __name__ == "__main__":
    main()
