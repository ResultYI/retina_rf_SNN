from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from training.metrics import gradient_norm


@dataclass(frozen=True, slots=True)
class ComponentGradientNorms:
    model_gradient_norm: float
    decoder_gradient_norm: float
    h1_gradient_norm: float
    bipolar_gradient_norm: float
    amacrine_gradient_norm: float
    rgc_spatial_gradient_norm: float
    rgc_temporal_gradient_norm: float
    rgc_threshold_gradient_norm: float
    temporal_gradient_norm: float


def component_gradient_norms(
    model: nn.Module,
    decoder: nn.Module,
) -> ComponentGradientNorms:
    named_parameters = tuple(model.named_parameters())
    temporal_tokens = ("tau", "gain", "mix")
    return ComponentGradientNorms(
        model_gradient_norm=gradient_norm(model.parameters()),
        decoder_gradient_norm=gradient_norm(decoder.parameters()),
        h1_gradient_norm=gradient_norm(
            parameter
            for name, parameter in named_parameters
            if name.startswith("h1.")
        ),
        bipolar_gradient_norm=gradient_norm(
            parameter
            for name, parameter in named_parameters
            if name.startswith("bipolar.")
        ),
        amacrine_gradient_norm=gradient_norm(
            parameter
            for name, parameter in named_parameters
            if name.startswith("amacrine.")
        ),
        rgc_spatial_gradient_norm=gradient_norm(
            parameter
            for name, parameter in named_parameters
            if name == "rgc.raw_spatial_sigma"
        ),
        rgc_temporal_gradient_norm=gradient_norm(
            parameter
            for name, parameter in named_parameters
            if name.startswith("rgc.")
            and any(token in name for token in temporal_tokens)
        ),
        rgc_threshold_gradient_norm=gradient_norm(
            parameter
            for name, parameter in named_parameters
            if name == "rgc.raw_threshold"
        ),
        temporal_gradient_norm=gradient_norm(
            parameter
            for name, parameter in named_parameters
            if any(token in name for token in temporal_tokens)
        ),
    )


__all__ = ["ComponentGradientNorms", "component_gradient_norms"]
