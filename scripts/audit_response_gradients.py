# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pyyaml", "torch"]
# ///
# How to run: python scripts/audit_response_gradients.py --help

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from data.input_identity import validate_experiment_input
from evaluation.gradient_decomposition import (
    audit_gradient_decomposition,
    write_gradient_decomposition,
)
from evaluation.probe_likelihood import (
    evaluate_validation_probe_likelihood,
    write_probe_likelihood,
)
from models.response_snn import build_response_retina_model
from scripts.run_experiment import _cone_spacing, _restore_trainer_lineage
from training.response_checkpointing import load_response_checkpoint
from training.response_config import load_response_config
from training.response_data import prepare_response_data
from training.response_trainer import ResponseTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit context/probe raw gradients and Adam effective updates."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-steps", type=int, default=64)
    parser.add_argument("--probe-likelihood-output")
    args = parser.parse_args()

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
    initialized_model = copy.deepcopy(model)
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
    write_gradient_decomposition(
        args.output,
        audit_gradient_decomposition(trainer, args.probe_steps),
    )
    if args.probe_likelihood_output:
        write_probe_likelihood(
            args.probe_likelihood_output,
            evaluate_validation_probe_likelihood(
                trainer,
                initialized_model,
                args.probe_steps,
            ),
        )


if __name__ == "__main__":
    main()
