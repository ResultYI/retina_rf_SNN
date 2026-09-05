from __future__ import annotations

from dataclasses import replace

import torch

from baselines.graph_tcn import GraphTCN, select_hidden_width
from baselines.lnln_subunit import LNPNLSubunit, select_subunit_count
from baselines.point_process_glm import PointProcessGLM
from evaluation.mechanistic_retina.mechanism_checkpoints import tensors_sha256
from evaluation.model_comparison.model_eval import ModelEvaluationRequest, evaluate_run
from evaluation.model_comparison.prediction import fit_bias, predict_trials
from evaluation.model_comparison.rf import glm_filter_rf, rf_cosine
from evaluation.model_comparison.run_data import BankRunData
from evaluation.model_comparison.training import (
    BaselineTrainingRequest,
    initialize_bias,
    train_baseline,
    train_glm_lbfgs,
)
from evaluation.model_comparison.types import ProgressEvent, RunResult, TrainingPoint
from training.mechanistic_retina.losses import expected_bernoulli_nll


def run_bias(request: BankRunData) -> RunResult:
    bias = fit_bias(request.train_spikes, request.train_mask)
    logits = lambda cones, history: bias.view(1, 1, -1).expand(
        cones.shape[0], cones.shape[1], -1
    )
    train_logits = logits(request.data.train_cones[:1], request.train_spikes[:1, 0])
    train_nll = expected_bernoulli_nll(
        train_logits, request.train_spikes[:1, 0], request.train_mask[:1, 0]
    )
    return evaluate_run(
        _evaluation_request(
            request,
            "Bias",
            None,
            16,
            logits,
            (TrainingPoint(0, float(train_nll), 0.0),),
            True,
            {"fit": "train-only empirical Bernoulli rate"},
            rf_enabled=False,
        )
    )


def run_glm(request: BankRunData) -> RunResult:
    model = PointProcessGLM(29, 16, 16, mode="full_glm")
    initialize_bias(model, request.train_spikes, request.train_mask)
    callback = lambda step, loss: request.progress(
        ProgressEvent("GLM-SH", request.bank_seed, None, step, loss)
    )
    training = train_glm_lbfgs(
        BaselineTrainingRequest(
            model,
            model,
            request.data.train_cones,
            request.train_spikes,
            request.train_mask,
            request.config.steps,
            request.config.checkpoints,
            request.config.learning_rate,
            request.config.batch_size,
            request.bank_seed,
            callback,
        )
    )
    result = evaluate_run(
        _evaluation_request(
            request,
            "GLM-SH",
            None,
            sum(parameter.numel() for parameter in model.parameters()),
            model,
            training.checkpoints,
            training.gradients_finite,
            {
                "solver": "full-batch L-BFGS strong-Wolfe, train split only",
                "solver_converged": training.converged,
                "solver_iterations": training.checkpoints[-1].step,
                "full_objective_gradient_infinity_norm": training.checkpoints[-1].gradient_infinity_norm,
                "no_cross_cell_coupling": True,
                "rf_supervision": False,
            },
        )
    )
    if result.rf_tensor is None:
        raise RuntimeError("GLM RF was not evaluated")
    parameter_rf = glm_filter_rf(model, result.rf_tensor.shape[0])
    extras = dict(result.extras)
    extras.update(
        {
            "parameter_filter_jacobian_cosine": rf_cosine(parameter_rf, result.rf_tensor),
            "parameter_filter_sha256": tensors_sha256({"K": parameter_rf}),
            "jacobian_rf_sha256": tensors_sha256({"RF": result.rf_tensor}),
        }
    )
    return replace(result, extras=extras)


def run_lnln(request: BankRunData, model_seed: int) -> RunResult:
    torch.manual_seed(model_seed)
    count = select_subunit_count(request.match_target_parameters, 16, 4)
    model = LNPNLSubunit(
        request.data.cone_positions,
        request.data.cell_positions,
        request.data.cell_types,
        request.data.polarities,
        count,
    )
    initialize_bias(model, request.train_spikes, request.train_mask)
    training = _train(request, "LN-LN", model_seed, model, model)
    kernels = model.subunit_kernels().detach()
    return evaluate_run(
        _evaluation_request(
            request,
            "LN-LN",
            model_seed,
            sum(parameter.numel() for parameter in model.parameters()),
            model,
            training.checkpoints,
            training.gradients_finite,
            {
                "subunits_per_cell": count,
                "subunit_rf_shape": list(kernels.shape),
                "subunit_rf_norms": [
                    float(value) for value in kernels.flatten(2).norm(dim=2).flatten()
                ],
                "subunit_rf_sha256": tensors_sha256({"subunits": kernels}),
                "subunit_locality_fraction": 1.0,
                "shared_subunit_fraction": model.shared_subunit_fraction,
                "rf_supervision": False,
            },
            auxiliary=kernels,
        )
    )


def run_graph_tcn(request: BankRunData, model_seed: int) -> RunResult:
    torch.manual_seed(model_seed)
    width = select_hidden_width(request.match_target_parameters, 16)
    model = GraphTCN(
        request.data.cone_positions, request.data.cell_positions, width
    )
    initialize_bias(model, request.train_spikes, request.train_mask)
    training = _train(request, "Graph-TCN", model_seed, model, model)
    result = evaluate_run(
        _evaluation_request(
            request,
            "Graph-TCN",
            model_seed,
            sum(parameter.numel() for parameter in model.parameters()),
            model,
            training.checkpoints,
            training.gradients_finite,
            {
                "hidden_width": width,
                "causal_blocks": 2,
                "receptive_field_steps": model.receptive_field_steps,
                "gradient_attribution": list(model.gradient_attribution()),
                "rf_supervision": False,
                "global_shortcut": False,
            },
        )
    )
    expected = request.data.validation_probability[:, 0, None].expand_as(
        request.validation_spikes
    )
    deltas = []
    for channel in range(width):
        logits = predict_trials(
            lambda cones, history, index=channel: model(
                cones, history, ablate_channel=index
            ),
            request.data.validation_cones,
            request.validation_spikes,
        )
        ce = expected_bernoulli_nll(logits, expected, request.validation_mask)
        deltas.append(float(ce) - result.prediction.teacher_expected_ce)
    extras = dict(result.extras)
    extras["channel_ablation_ce_delta"] = deltas
    return replace(result, extras=extras)


def _train(request, name, seed, model, logits):
    callback = lambda step, loss: request.progress(
        ProgressEvent(name, request.bank_seed, seed, step, loss)
    )
    return train_baseline(
        BaselineTrainingRequest(
            model,
            logits,
            request.data.train_cones,
            request.train_spikes,
            request.train_mask,
            request.config.steps,
            request.config.checkpoints,
            request.config.learning_rate,
            request.config.batch_size,
            request.bank_seed if seed is None else seed,
            callback,
        )
    )


def _evaluation_request(
    request,
    name,
    seed,
    parameter_count,
    logits,
    training,
    finite,
    extras,
    *,
    auxiliary=None,
    rf_enabled=True,
):
    bias = fit_bias(request.train_spikes, request.train_mask)
    return ModelEvaluationRequest(
        name,
        request.bank_seed,
        seed,
        parameter_count,
        logits,
        request.data.validation_cones,
        request.validation_spikes,
        request.validation_mask,
        request.data.validation_probability[:, 0],
        bias,
        request.candidate,
        request.data.cone_positions,
        request.data.cell_positions,
        training,
        finite,
        extras,
        auxiliary,
        rf_enabled,
    )


__all__ = ["run_bias", "run_glm", "run_graph_tcn", "run_lnln"]
