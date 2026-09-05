from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from baselines.graph_tcn import GraphTCN, select_hidden_width
from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    constant_rate_logits,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.real_sampled import (
    SpikePredictionMetrics,
    spike_prediction_metrics,
)


@dataclass(frozen=True, slots=True)
class CompactNeuralTrainingRequest:
    train: RealSequenceSplit
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    target_parameters: int
    seed: int
    maximum_steps: int = 2_000
    patience: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    history_lags: int = 4


@dataclass(frozen=True, slots=True)
class CompactNeuralTrainingResult:
    model: GraphTCN
    train_nll_initial: float
    train_nll_trained: float
    best_step: int
    stop_step: int
    gradients_finite: bool
    actually_updated: tuple[str, ...]
    validation_used: bool = False


class CompactNeuralError(ValueError):
    pass


def graph_tcn_spatial_drive(model: GraphTCN, cones: torch.Tensor) -> torch.Tensor:
    graph = torch.einsum("btc,cd->btd", cones, model.cone_graph)
    return torch.einsum("btc,nc->btn", graph, model.cell_pool)


def compact_neural_logits(
    model: GraphTCN,
    spatial_drive: torch.Tensor,
    observed_counts: torch.Tensor,
) -> torch.Tensor:
    hidden = model.input_projection(spatial_drive.unsqueeze(-1))
    batch, time, cells, width = hidden.shape
    sequence = hidden.permute(0, 2, 3, 1).reshape(batch * cells, width, time)
    for block in model.blocks:
        sequence = block(sequence)
    features = sequence.reshape(batch, cells, width, time).permute(0, 3, 1, 2)
    logits = (features * model.readout[None, None]).sum(dim=-1) + model.bias
    result = logits.clone()
    for lag in range(1, model.history.shape[1] + 1):
        if lag >= observed_counts.shape[1]:
            break
        result[:, lag:] += observed_counts[:, :-lag] * model.history[:, lag - 1]
    return result


def fit_compact_neural_baseline(
    request: CompactNeuralTrainingRequest,
) -> CompactNeuralTrainingResult:
    _validate_request(request)
    torch.manual_seed(request.seed)
    width = select_hidden_width(request.target_parameters, 1)
    model = GraphTCN(
        request.cone_positions,
        request.cell_positions,
        width,
        history_lags=request.history_lags,
    )
    train = request.train
    bias = constant_rate_logits(
        train.spike_events,
        train.valid_mask,
        train.spike_events[:1, :1],
        train.valid_mask[:1, :1],
    )[0, 0]
    with torch.no_grad():
        model.bias.copy_(bias)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    spatial = graph_tcn_spatial_drive(model, train.cone_drive).detach()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=request.learning_rate,
        weight_decay=request.weight_decay,
    )

    def data_nll() -> torch.Tensor:
        return expected_bernoulli_nll(
            compact_neural_logits(model, spatial, train.spike_events),
            train.spike_events,
            train.valid_mask,
        )

    initial_nll = float(data_nll().detach())
    best_loss = float("inf")
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_step = 0
    stale = 0
    gradients_finite = True
    stop_step = request.maximum_steps
    for step in range(1, request.maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = data_nll()
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            stop_step = step
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        current = float(data_nll().detach())
        if current < best_loss - 1e-7:
            best_loss = current
            best_step = step
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= request.patience:
            stop_step = step
            break
    model.load_state_dict(best_state)
    updated = tuple(
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(initial[name], parameter.detach())
    )
    return CompactNeuralTrainingResult(
        model=model,
        train_nll_initial=initial_nll,
        train_nll_trained=float(data_nll().detach()),
        best_step=best_step,
        stop_step=stop_step,
        gradients_finite=gradients_finite,
        actually_updated=updated,
    )


def evaluate_compact_neural(
    model: GraphTCN,
    split: RealSequenceSplit,
) -> tuple[SpikePredictionMetrics, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        spatial = graph_tcn_spatial_drive(model, split.cone_drive)
        logits = compact_neural_logits(model, spatial, split.spike_events)
    return spike_prediction_metrics(logits, split.spike_events, split.valid_mask), logits


def _validate_request(request: CompactNeuralTrainingRequest) -> None:
    values = (request.learning_rate, request.weight_decay)
    if request.train.spike_events.shape[-1] != 1:
        raise CompactNeuralError("compact neural baseline requires exactly one cell")
    if request.maximum_steps < 1 or request.patience < 1:
        raise CompactNeuralError("compact neural training budgets must be positive")
    if request.target_parameters < 1 or request.history_lags < 1:
        raise CompactNeuralError("compact neural model dimensions must be positive")
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise CompactNeuralError("compact neural optimizer settings must be finite")


__all__ = [
    "CompactNeuralTrainingRequest",
    "CompactNeuralTrainingResult",
    "CompactNeuralError",
    "compact_neural_logits",
    "evaluate_compact_neural",
    "fit_compact_neural_baseline",
    "graph_tcn_spatial_drive",
]
