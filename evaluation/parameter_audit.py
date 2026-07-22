from __future__ import annotations

from dataclasses import dataclass

import torch

from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel


@dataclass(frozen=True, slots=True)
class ParameterAuditEntry:
    name: str
    shape: tuple[int, ...]
    finite: bool
    trainable: bool
    lower_bound: float | None
    upper_bound: float | None
    initial_minimum: float
    initial_maximum: float
    final_minimum: float
    final_maximum: float
    maximum_absolute_delta: float
    mean_absolute_delta: float
    minimum_fraction_of_range: float | None
    maximum_fraction_of_range: float | None
    near_boundary_fraction: float | None


ParameterSpec = tuple[torch.Tensor, tuple[float, float] | None, bool]


def audit_parameters(
    trained_model: RetinaModel,
    trained_decoder: TiedLocalDecoder,
    *,
    initialized_model: RetinaModel | None = None,
    initialized_decoder: TiedLocalDecoder | None = None,
) -> tuple[ParameterAuditEntry, ...]:
    final_specs = _parameter_specs(trained_model, trained_decoder)
    initial_specs = (
        _parameter_specs(initialized_model, initialized_decoder)
        if initialized_model is not None and initialized_decoder is not None
        else final_specs
    )
    if final_specs.keys() != initial_specs.keys():
        raise ValueError("Initialized and trained parameter groups do not match")
    return tuple(
        _entry(name, initial_specs[name][0], *final_specs[name])
        for name in final_specs
    )


def _parameter_specs(
    model: RetinaModel,
    decoder: TiedLocalDecoder,
) -> dict[str, ParameterSpec]:
    h1 = model.h1
    bipolar = model.bipolar
    amacrine = model.amacrine
    rgc = model.rgc
    return {
        "h1.gain": (h1.gain, (0.0, h1._gain_max), True),
        "h1.tau_ms": (h1.tau_ms, (h1._tau_min_ms, h1._tau_max_ms), True),
        "bipolar.tau_sustained_ms": (
            bipolar.tau_ms[0],
            tuple(float(value) for value in bipolar.tau_bounds_ms[0]),
            True,
        ),
        "bipolar.tau_transient_ms": (
            bipolar.tau_ms[1],
            tuple(float(value) for value in bipolar.tau_bounds_ms[1]),
            True,
        ),
        "bipolar.g_ab_sustained": (
            bipolar.g_ab[0],
            (0.0, bipolar._g_ab_max[0]),
            True,
        ),
        "bipolar.g_ab_transient": (
            bipolar.g_ab[1],
            (0.0, bipolar._g_ab_max[1]),
            True,
        ),
        "bipolar.polarity_gain": (
            bipolar.polarity_gain,
            bipolar._polarity_gain_bounds,
            True,
        ),
        "bipolar.polarity_threshold": (
            bipolar.polarity_threshold,
            bipolar._polarity_threshold_bounds,
            True,
        ),
        "bipolar.rectifier_softness": (
            bipolar.rectifier_softness,
            bipolar._rectifier_softness_bounds,
            True,
        ),
        "amacrine.tau_sustained_ms": (
            amacrine.tau_ms[0],
            tuple(float(value) for value in amacrine.tau_bounds_ms[0]),
            True,
        ),
        "amacrine.tau_transient_ms": (
            amacrine.tau_ms[1],
            tuple(float(value) for value in amacrine.tau_bounds_ms[1]),
            True,
        ),
        "amacrine.g_ba_sustained": (
            amacrine.g_ba[0],
            (0.0, amacrine._g_ba_max[0]),
            True,
        ),
        "amacrine.g_ba_transient": (
            amacrine.g_ba[1],
            (0.0, amacrine._g_ba_max[1]),
            True,
        ),
        "rgc.spatial_sigma": (rgc.spatial_sigma, rgc.sigma_bounds, True),
        "rgc.sustained_mix": (rgc.sustained_mix, (0.0, 1.0), True),
        "rgc.membrane_tau_ms": (rgc.membrane_tau_ms, rgc.tau_bounds, True),
        "rgc.adaptation_tau_ms": (rgc.adaptation_tau_ms, rgc.tau_bounds, True),
        "rgc.adaptation_gain": (
            rgc.adaptation_gain,
            (0.0, rgc.adaptation_gain_max),
            True,
        ),
        "rgc.amacrine_gain": (
            rgc.amacrine_gain,
            (0.0, rgc.amacrine_gain_max),
            True,
        ),
        "rgc.threshold": (rgc.threshold, (0.02, 2.0), True),
        "rgc.subunit_tau_ms": (rgc.subunit_tau_ms, rgc.tau_bounds, True),
        "rgc.subunit_gain": (
            rgc.subunit_gain,
            (0.0, rgc.subunit_gain_max),
            True,
        ),
        "rgc.readout_rate_tau_ms": (rgc.readout_rate_tau_ms, None, False),
        "decoder.unit_gain": (
            decoder.unit_gain,
            (0.0, decoder.gain_max),
            True,
        ),
        "decoder.cone_bias": (decoder.cone_bias, None, True),
    }


def _entry(
    name: str,
    initial: torch.Tensor,
    final: torch.Tensor,
    bounds: tuple[float, float] | None,
    trainable: bool,
) -> ParameterAuditEntry:
    initial = initial.detach()
    final = final.detach()
    if initial.shape != final.shape:
        raise ValueError(f"Parameter shape changed for {name}")
    delta = (final - initial).abs()
    lower, upper = bounds if bounds is not None else (None, None)
    if bounds is None:
        minimum_fraction = maximum_fraction = near_boundary = None
    else:
        width = max(upper - lower, torch.finfo(final.dtype).eps)
        fractions = (final - lower) / width
        minimum_fraction = float(fractions.min())
        maximum_fraction = float(fractions.max())
        near_boundary = float(
            ((fractions <= 0.01) | (fractions >= 0.99)).float().mean()
        )
    return ParameterAuditEntry(
        name=name,
        shape=tuple(final.shape),
        finite=bool(torch.isfinite(initial).all() and torch.isfinite(final).all()),
        trainable=trainable,
        lower_bound=lower,
        upper_bound=upper,
        initial_minimum=float(initial.min()),
        initial_maximum=float(initial.max()),
        final_minimum=float(final.min()),
        final_maximum=float(final.max()),
        maximum_absolute_delta=float(delta.max()),
        mean_absolute_delta=float(delta.mean()),
        minimum_fraction_of_range=minimum_fraction,
        maximum_fraction_of_range=maximum_fraction,
        near_boundary_fraction=near_boundary,
    )


__all__ = ["ParameterAuditEntry", "audit_parameters"]
