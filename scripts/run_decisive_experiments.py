# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numpy", "pyyaml", "torch"]
# ///
# How to run: python scripts/run_decisive_experiments.py --help

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from data.input_identity import validate_experiment_input
from evaluation.factorial_oracles import (
    FactorialOracleRequest,
    audit_factorial_oracles,
)
from evaluation.factorial_reachability import (
    FactorialReachabilityRequest,
    audit_factorial_reachability,
)
from evaluation.target_gradient_comparison import (
    compare_full_batch_target_gradients,
)
from evaluation.teacher_identifiability import (
    audit_empirical_identifiability,
    reconstruct_teacher_targets,
)
from evaluation.trial_power import TrialPowerRequest, audit_trial_power_curve
from evaluation.type_gain_oracle import GainEvaluationRequest
from evaluation.type_gain_reachability import (
    TypeGainReachabilityRequest,
    audit_type_gain_reachability,
)
from models.response_snn import build_response_retina_model
from scripts.run_experiment import _cone_spacing, _restore_trainer_lineage
from training.response_checkpointing import load_response_checkpoint
from training.response_config import load_response_config
from training.response_data import prepare_response_data
from training.response_trainer import ResponseTrainer


class DecisiveExperimentError(ValueError):
    pass


def main() -> None:
    args = parse_decisive_args(sys.argv[1:])
    _run(args)


def parse_decisive_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run six short adaptive-RF identifiability experiments."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strong-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-steps", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--power-seeds", type=int, default=None)
    parser.add_argument("--power-bootstrap-iterations", type=int, default=None)
    parser.add_argument("--oracle-iterations", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.bootstrap_iterations = _resolve(args.bootstrap_iterations, args.smoke, 20, 2000)
    args.power_seeds = _resolve(args.power_seeds, args.smoke, 1, 100)
    args.power_bootstrap_iterations = _resolve(
        args.power_bootstrap_iterations,
        args.smoke,
        20,
        500,
    )
    args.oracle_iterations = _resolve(args.oracle_iterations, args.smoke, 1, 40)
    return args


def _run(args: argparse.Namespace) -> None:

    config = load_response_config(args.config)
    torch.manual_seed(config.seed)
    data = prepare_response_data(config.data)
    validate_experiment_input(data.input_identity, data.dt_ms)
    priors = load_type_priors(
        config.model.type_prior_path,
        required_type_ids=tuple(sorted(set(data.cells.type_ids))),
    )
    device = torch.device(args.device)
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
    trainer = ResponseTrainer(model, config, data, device)
    checkpoint_state = load_response_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=trainer.optimizer,
        generator=trainer.sampling_generator,
        fingerprint=data.fingerprint,
        target_kind=data.target_kind.value,
        config=config,
    )
    _restore_trainer_lineage(trainer, checkpoint_state)

    validation_path = _single_path(config.data.validation_glob)
    teacher_targets = reconstruct_teacher_targets(validation_path)
    if teacher_targets.conditional_probabilities.shape != tuple(
        data.validation.spike_counts.shape
    ):
        raise DecisiveExperimentError(
            "Reconstructed teacher targets do not match prepared validation data"
        )
    standard_identifiability = audit_empirical_identifiability(
        validation_path,
        probe_steps=args.probe_steps,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=config.seed,
    )
    strong_identifiability = audit_empirical_identifiability(
        args.strong_validation,
        probe_steps=args.probe_steps,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=config.seed,
    )
    target_gradients = compare_full_batch_target_gradients(
        trainer,
        torch.as_tensor(teacher_targets.conditional_probabilities),
        probe_steps=args.probe_steps,
    )
    teacher_gains = torch.as_tensor(
        teacher_targets.teacher_signed_gains,
        device=device,
    )
    gain_evaluation = GainEvaluationRequest(
        model,
        data.validation,
        data.cells.type_ids,
        config.evaluation.rf_lag_steps,
    )
    reachability = audit_type_gain_reachability(
        TypeGainReachabilityRequest(
            model,
            data.validation,
            data.cells.type_ids,
            teacher_gains,
            config.evaluation.rf_lag_steps,
            oracle_steps=0,
        )
    )
    trial_power = audit_trial_power_curve(
        TrialPowerRequest(
            validation_path,
            monte_carlo_seeds=args.power_seeds,
            bootstrap_iterations=args.power_bootstrap_iterations,
            probe_steps=args.probe_steps,
            seed=config.seed,
        )
    )
    factorial_reachability = audit_factorial_reachability(
        FactorialReachabilityRequest(
            model,
            data.validation,
            data.cells.type_ids,
            tuple(int(value) for value in data.cells.polarities),
            teacher_gains,
            config.evaluation.rf_lag_steps,
        )
    )
    factorial_oracles = audit_factorial_oracles(
        FactorialOracleRequest(
            gain_evaluation,
            teacher_gains,
            max_iterations=args.oracle_iterations,
        )
    )
    result = {
        "experiment_1_empirical_target_identifiability": {
            "standard": asdict(standard_identifiability),
            "strong": asdict(strong_identifiability),
            "strong_minus_standard_changed_spike_bin_fraction": (
                strong_identifiability.changed_spike_bin_fraction
                - standard_identifiability.changed_spike_bin_fraction
            ),
        },
        "experiment_2_hard_vs_soft_full_batch_gradient": asdict(target_gradients),
        "experiment_3_type_gain_reachability": asdict(reachability),
        "experiment_4_trial_count_power_curve": asdict(trial_power),
        "experiment_5_four_cell_factorial_jacobian": asdict(
            factorial_reachability
        ),
        "experiment_6_deterministic_factorial_oracles": asdict(
            factorial_oracles
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _resolve(value: int | None, smoke: bool, smoke_default: int, default: int) -> int:
    if value is not None:
        return value
    return smoke_default if smoke else default


def _single_path(pattern: str) -> Path:
    matches = tuple(Path(path) for path in sorted(glob.glob(pattern)))
    if len(matches) != 1:
        raise DecisiveExperimentError(
            "Decisive experiment requires exactly one validation HDF5 file"
        )
    return matches[0]


if __name__ == "__main__":
    main()
