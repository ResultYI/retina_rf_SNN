from __future__ import annotations

from torch import nn
import torch

from training.response_config import ResponseExperimentConfig


def configure_cell_residual_learning(
    model: nn.Module,
    *,
    learnable: bool,
) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("rgc.") and name.endswith(".cell_residual_raw"):
            parameter.requires_grad_(learnable)


def freeze_threshold(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("rgc.threshold."):
            parameter.requires_grad_(False)


def build_response_optimizer(
    model: nn.Module,
    config: ResponseExperimentConfig,
) -> torch.optim.AdamW:
    response_bias = []
    rgc = []
    upstream = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "rgc.response_bias":
            response_bias.append(parameter)
        elif name.startswith("rgc."):
            rgc.append(parameter)
        else:
            upstream.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "name": "response_bias",
                "params": response_bias,
                "lr": config.training.response_bias_lr,
            },
            {
                "name": "rgc",
                "params": rgc,
                "lr": config.training.rgc_lr,
            },
            {
                "name": "upstream",
                "params": upstream,
                "lr": config.training.learning_rate,
            },
        ],
        weight_decay=0.0,
    )


__all__ = [
    "build_response_optimizer",
    "configure_cell_residual_learning",
    "freeze_threshold",
]
