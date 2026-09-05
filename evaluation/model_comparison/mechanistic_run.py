from __future__ import annotations

from dataclasses import asdict
import hashlib

import torch

from evaluation.mechanistic_retina.artifacts import FINAL_TEST_BOUNDARY
from evaluation.mechanistic_retina.mechanism_checkpoints import tensors_sha256
from evaluation.mechanistic_retina.mechanism_runtime import build_student, pathway_rfs
from evaluation.mechanistic_retina.pathway_decomposition import effective_pathway_rf
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.model_comparison.artifacts import (
    load_comparison_checkpoint,
    save_comparison_checkpoint,
)
from evaluation.model_comparison.model_eval import ModelEvaluationRequest, evaluate_run
from evaluation.model_comparison.parameters import parameter_inventory, parameter_snapshot
from evaluation.model_comparison.prediction import fit_bias
from evaluation.model_comparison.rf import rf_cosine
from evaluation.model_comparison.run_data import BankRunData
from evaluation.model_comparison.types import ProgressEvent, RunResult, TrainingPoint
from models.mechanistic_retina.contracts import MECHANISTIC_MODEL_REVISION
from training.mechanistic_retina.sampled import SampledTrainingRequest, train_sampled_model
from training.mechanistic_retina.optimizer import phase1_parameters


class MechanisticRunError(RuntimeError):
    pass


def run_mechanistic(request: BankRunData, model_seed: int) -> RunResult:
    model = build_student(request.data, model_seed)
    initial_parameters = parameter_snapshot(model)
    raw_state_sha256 = tensors_sha256(model.state_dict())
    training = train_sampled_model(
        SampledTrainingRequest(
            model,
            request.data.train_cones,
            request.train_spikes,
            request.train_mask,
            request.data.validation_cones,
            request.validation_spikes,
            request.validation_mask,
            request.data.validation_probability[:, 0],
            request.config.steps,
            request.config.checkpoints,
            request.config.learning_rate,
            request.config.batch_size,
            model_seed,
        )
    )
    points = tuple(
        TrainingPoint(point.step, point.train_nll, 0.0)
        for point in training.checkpoints
    )
    for point in points:
        request.progress(
            ProgressEvent(
                "Mechanistic Retina",
                request.bank_seed,
                model_seed,
                point.step,
                point.train_nll,
            )
        )
    checkpoint = _checkpoint(request, model, model_seed, raw_state_sha256)
    extras = _mechanism_payload(request, model)
    inventory = parameter_inventory(
        model,
        phase1_parameters(model),
        initial_parameters=initial_parameters,
    )
    extras.update(
        {
            "raw_initialization_sha256": raw_state_sha256,
            "checkpoint_path": str(checkpoint.path.relative_to(request.root)),
            "checkpoint_sha256": checkpoint.sha256,
            "checkpoint_bytes": checkpoint.bytes,
            "checkpoint_roundtrip_verified": True,
            "parameter_inventory": {
                "total": inventory.total,
                "requires_grad": inventory.requires_grad,
                "optimizer_listed": inventory.optimizer_listed,
                "nonzero_gradient": inventory.nonzero_gradient,
                "actually_updated": inventory.actually_updated,
            },
            "rf_supervision": False,
            "noise_free_warm_start": False,
        }
    )
    logits = lambda cones, history: model.forward_sequence(
        cones, observed_counts=history
    ).logits
    return evaluate_run(
        ModelEvaluationRequest(
            "Mechanistic Retina",
            request.bank_seed,
            model_seed,
            sum(parameter.numel() for parameter in model.parameters()),
            logits,
            request.data.validation_cones,
            request.validation_spikes,
            request.validation_mask,
            request.data.validation_probability[:, 0],
            fit_bias(request.train_spikes, request.train_mask),
            request.candidate,
            request.data.cone_positions,
            request.data.cell_positions,
            points,
            training.gradients_finite,
            extras,
        )
    )


def _checkpoint(request, model, model_seed, raw_state_sha256):
    path = (
        request.root
        / request.config.run_dir
        / "mechanistic"
        / f"bank-{request.bank_seed}"
        / f"seed-{model_seed}"
        / "final.pt"
    )
    metadata = {
        "architecture_revision": (
            f"mechanism_identifiable-r{MECHANISTIC_MODEL_REVISION}"
        ),
        "teacher": "Candidate0",
        "teacher_rf_sha256": request.candidate.rf_sha256,
        "bank_seed": request.bank_seed,
        "model_seed": model_seed,
        "trial_budget": request.config.trials,
        "step": request.config.steps,
        "raw_initialization_sha256": raw_state_sha256,
        "cell_order": [value.cell_id for value in request.candidate.metadata],
        "cone_order": list(range(request.data.train_cones.shape[-1])),
        "lag_order": list(range(16)),
        "final_test_boundary": list(FINAL_TEST_BOUNDARY),
    }
    saved = save_comparison_checkpoint(path, model, metadata)
    restored = build_student(request.data, model_seed)
    loaded = load_comparison_checkpoint(path, restored)
    if loaded != metadata or tensors_sha256(restored.state_dict()) != tensors_sha256(model.state_dict()):
        raise MechanisticRunError("mechanistic final checkpoint roundtrip differs")
    return saved


def _mechanism_payload(request, model):
    cones = request.data.validation_cones[:2]
    history = request.validation_spikes[:2, 0]
    total = effective_rf(model, cones, history)
    detailed = dict(effective_pathway_rf(model, cones, history))
    interventions = pathway_rfs(model, cones, history)
    teacher = request.candidate.rf.unsqueeze(0).expand(cones.shape[0], -1, -1, -1)
    pathway_sum = sum(detailed.values(), torch.zeros_like(total))
    with torch.no_grad():
        output = model.forward_sequence(cones, observed_counts=history)
        current_sum = (
            output.bc_sustained_current
            + output.bc_transient_current
            + output.amacrine_local_current
            + output.amacrine_transient_current
        )
        gates = model.gates.values(frozenset())
    pathways = {
        **detailed,
        "H1-intervention": interventions["H1"],
    }
    return {
        "conditional_context_count": cones.shape[0],
        "pathway_rf": {
            name: {
                "shape": list(value.shape),
                "norm": float(value.norm()),
                "teacher_cosine": rf_cosine(value, teacher),
                "sha256": tensors_sha256({name: value}),
            }
            for name, value in pathways.items()
        },
        "pathway_rf_sum_error": float((pathway_sum - total).abs().max()),
        "gate_values": {
            "h1": float(gates.h1),
            "ac_local": float(gates.ac_local),
            "ac_transient": float(gates.ac_transient),
            "history": float(gates.history),
        },
        "pathway_current_abs_mean": {
            "BC-sustained": float(output.bc_sustained_current.abs().mean()),
            "BC-transient": float(output.bc_transient_current.abs().mean()),
            "AC-local": float(output.amacrine_local_current.abs().mean()),
            "AC-transient": float(output.amacrine_transient_current.abs().mean()),
        },
        "current_sum_error": float((current_sum - output.total_current).abs().max()),
    }


def config_sha256(request: BankRunData) -> str:
    path = request.root / "configs/model_comparison_t2.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["config_sha256", "run_mechanistic"]
