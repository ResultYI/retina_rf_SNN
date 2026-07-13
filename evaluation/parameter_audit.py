from __future__ import annotations

from dataclasses import dataclass

import torch

from training.stage1 import Stage1Components


@dataclass(frozen=True, slots=True)
class BoundedParameterStatus:
    name: str
    value: float
    lower: float
    upper: float
    boundary_fraction: float
    near_boundary: bool


def audit_stage1_parameters(
    components: Stage1Components,
) -> tuple[BoundedParameterStatus, ...]:
    core = components.core
    statuses = [
        _status(
            "h1.tau_ms",
            core.h1.tau_ms,
            (core.h1._tau_min_ms, core.h1._tau_max_ms),
        ),
        _status("h1.gain", core.h1.gain, (0.0, core.h1._gain_max)),
    ]
    statuses.extend(
        _vector_statuses(
            ("bipolar.tau_sustained_ms", "bipolar.tau_transient_ms"),
            core.bipolar.tau_ms,
            core.bipolar.tau_bounds_ms,
        )
    )
    statuses.append(
        _order_margin_status(
            "bipolar.tau_order_margin_ms",
            core.bipolar.tau_ms,
            core.bipolar.tau_bounds_ms,
        )
    )
    statuses.extend(
        _vector_statuses(
            ("bipolar.g_ab_sustained", "bipolar.g_ab_transient"),
            core.bipolar.g_ab,
            core.bipolar.g_ab.new_tensor(
                tuple((0.0, bound) for bound in core.bipolar._g_ab_max)
            ),
        )
    )
    statuses.append(
        _order_margin_status(
            "amacrine.tau_order_margin_ms",
            core.amacrine.tau_ms,
            core.amacrine.tau_bounds_ms,
        )
    )
    statuses.extend(
        _vector_statuses(
            ("amacrine.tau_sustained_ms", "amacrine.tau_transient_ms"),
            core.amacrine.tau_ms,
            core.amacrine.tau_bounds_ms,
        )
    )
    statuses.append(
        _order_margin_status(
            "rgc.tau_order_margin_ms",
            core.rgc.dynamics.tau_ms,
            core.rgc.dynamics.tau_bounds_ms,
        )
    )
    statuses.extend(
        _vector_statuses(
            ("amacrine.g_ba_sustained", "amacrine.g_ba_transient"),
            core.amacrine.g_ba,
            core.amacrine.g_ba.new_tensor(
                tuple((0.0, bound) for bound in core.amacrine._g_ba_max)
            ),
        )
    )
    statuses.extend(
        _vector_statuses(
            ("rgc.adaptation_tau_ms", "rgc.membrane_tau_ms"),
            core.rgc.dynamics.tau_ms,
            core.rgc.dynamics.tau_bounds_ms,
        )
    )
    statuses.extend(
        _vector_statuses(
            ("rgc.g_ag_midget", "rgc.g_ag_parasol", "rgc.g_ag_residual"),
            core.rgc.g_ag,
            core.rgc.g_ag.new_tensor(
                tuple((0.0, bound) for bound in core.rgc._g_ag_max)
            ),
        )
    )
    statuses.extend(
        _vector_statuses(
            (
                "rgc.midget_sustained_mix",
                "rgc.midget_transient_mix",
                "rgc.parasol_sustained_mix",
                "rgc.parasol_transient_mix",
            ),
            core.rgc.kinetic_mix.flatten(),
            core.rgc.kinetic_mix.new_tensor(((0.0, 1.0),) * 4),
        )
    )
    for prefix, projection in (
        ("decoder.fine_residual", components.decoder.fine_residual),
        ("decoder.coarse_residual", components.decoder.coarse_residual),
    ):
        maximum = projection._weight_max
        if maximum is None:
            continue
        values = projection.effective_weight.flatten()
        statuses.extend(
            _vector_statuses(
                tuple(f"{prefix}[{index}]" for index in range(values.numel())),
                values,
                values.new_tensor(((-maximum, maximum),) * values.numel()),
            )
        )
    return tuple(statuses)


def _vector_statuses(
    names: tuple[str, ...],
    values: torch.Tensor,
    bounds: torch.Tensor,
) -> tuple[BoundedParameterStatus, ...]:
    return tuple(
        _status(name, value, (float(bound[0]), float(bound[1])))
        for name, value, bound in zip(names, values, bounds, strict=True)
    )


def _status(
    name: str,
    value: torch.Tensor,
    bounds: tuple[float, float],
) -> BoundedParameterStatus:
    lower, upper = bounds
    scalar = float(value.detach())
    fraction = min(scalar - lower, upper - scalar) / (upper - lower)
    boundary_fraction = min(0.5, max(0.0, fraction))
    return BoundedParameterStatus(
        name=name,
        value=scalar,
        lower=lower,
        upper=upper,
        boundary_fraction=boundary_fraction,
        near_boundary=boundary_fraction <= 0.05,
    )


def _order_margin_status(
    name: str,
    ordered_values: torch.Tensor,
    bounds: torch.Tensor,
) -> BoundedParameterStatus:
    maximum_margin = float(bounds[0, 1] - bounds[1, 0])
    margin = ordered_values[0] - ordered_values[1]
    return _status(name, margin, (0.0, maximum_margin))
