from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from configs.rgc_type_priors import ParameterPrior, RGCTypePriors
from data.rgc_response import CellMetadata


class TypedRGCError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TypedRGCState:
    membrane: torch.Tensor
    adaptation: torch.Tensor
    rate: torch.Tensor
    subunit_energy: torch.Tensor


@dataclass(frozen=True, slots=True)
class TypedRGCStepOutput:
    spike_logits: torch.Tensor
    spike_probability: torch.Tensor
    hard_spikes: torch.Tensor
    filtered_rate: torch.Tensor
    generator_potential: torch.Tensor


@dataclass(frozen=True, slots=True)
class TypedRGCOutput:
    spike_logits: torch.Tensor
    spike_probability: torch.Tensor
    hard_spikes: torch.Tensor
    filtered_rate: torch.Tensor
    generator_potential: torch.Tensor


class TypeConditionedParameter(nn.Module):
    def __init__(
        self,
        priors: tuple[ParameterPrior, ...],
        cell_type_indices: torch.Tensor,
        residual_scale: float,
    ) -> None:
        super().__init__()
        means = torch.tensor([prior.mean for prior in priors], dtype=torch.float32)
        lower = torch.tensor([prior.lower for prior in priors], dtype=torch.float32)
        upper = torch.tensor([prior.upper for prior in priors], dtype=torch.float32)
        fraction = ((means - lower) / (upper - lower)).clamp(1e-5, 1 - 1e-5)
        self.type_base_raw = nn.Parameter(torch.logit(fraction))
        self.register_buffer(
            "prior_type_base_raw",
            self.type_base_raw.detach().clone(),
        )
        self.cell_residual_raw = nn.Parameter(
            torch.zeros(cell_type_indices.numel(), dtype=torch.float32)
        )
        self.register_buffer("cell_type_indices", cell_type_indices.to(torch.long))
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self._residual_scale = residual_scale

    def forward(self) -> torch.Tensor:
        indices = self.cell_type_indices
        raw = self.type_base_raw.index_select(0, indices)
        raw = raw + self._residual_scale * torch.tanh(self.cell_residual_raw)
        lower = self.lower.index_select(0, indices)
        upper = self.upper.index_select(0, indices)
        return lower + (upper - lower) * torch.sigmoid(raw)

    def residual_penalty(self) -> torch.Tensor:
        return self.cell_residual_raw.square().mean()

    def type_prior_penalty(self) -> torch.Tensor:
        return (self.type_base_raw - self.prior_type_base_raw).square().mean()


class TypedRGCPopulation(nn.Module):
    parameter_names = (
        "spatial_sigma",
        "sustained_mix",
        "membrane_tau_ms",
        "adaptation_tau_ms",
        "adaptation_gain",
        "amacrine_gain",
        "threshold",
        "subunit_tau_ms",
        "subunit_gain",
    )

    def __init__(
        self,
        cone_positions_degs: torch.Tensor,
        cells: CellMetadata,
        priors: RGCTypePriors,
        *,
        dt_ms: float,
        support_radius_degs: float,
        readout_rate_tau_ms: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        cones = torch.as_tensor(cone_positions_degs, dtype=torch.float32)
        centers = torch.as_tensor(cells.positions_degs, dtype=torch.float32)
        if cones.ndim != 2 or centers.ndim != 2 or cones.shape[1] != 2:
            raise TypedRGCError("Cone and cell positions must have shape [count,2]")
        if support_radius_degs <= 0 or dt_ms <= 0 or readout_rate_tau_ms <= 0:
            raise TypedRGCError("RGC time and support values must be positive")
        type_lookup = {type_id: index for index, type_id in enumerate(priors.type_ids)}
        try:
            type_indices = torch.tensor(
                [type_lookup[type_id] for type_id in cells.type_ids],
                dtype=torch.long,
            )
        except KeyError as exc:
            raise TypedRGCError(f"Missing RGC type prior: {exc.args[0]}") from exc
        polarities = torch.as_tensor(cells.polarities, dtype=torch.long)
        if polarities.shape != (centers.shape[0],):
            raise TypedRGCError("Cell polarity shape is invalid")
        distance_sq = torch.cdist(centers, cones).square()
        support = distance_sq <= support_radius_degs**2
        if not support.any(dim=1).all():
            raise TypedRGCError("Every recorded cell needs a cone inside its support")
        self.register_buffer("cone_positions_degs", cones)
        self.register_buffer("cell_centers_degs", centers)
        self.register_buffer("cell_polarities", polarities)
        self.register_buffer("distance_sq_degs", distance_sq)
        self.register_buffer("support_mask", support)
        self._dt_ms = float(dt_ms)
        self._rate_tau_ms = float(readout_rate_tau_ms)
        self._surrogate_slope = float(surrogate_slope)
        self._residual_weight = priors.cell_residual_weight
        self._type_prior_weight = priors.type_prior_weight
        for name in self.parameter_names:
            values = tuple(prior.parameter(name) for prior in priors.types)
            setattr(
                self,
                name,
                TypeConditionedParameter(
                    values,
                    type_indices,
                    priors.cell_residual_scale,
                ),
            )

    @property
    def cell_count(self) -> int:
        return int(self.cell_polarities.numel())

    def compute_spatial_weights(self) -> torch.Tensor:
        sigma = self.spatial_sigma().square().unsqueeze(1)
        logits = -self.distance_sq_degs / (2 * sigma)
        return torch.softmax(logits.masked_fill(~self.support_mask, -torch.inf), dim=-1)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TypedRGCState:
        shape = (batch_size, self.cell_count)
        return TypedRGCState(
            membrane=torch.zeros(shape, device=device, dtype=dtype),
            adaptation=torch.zeros(shape, device=device, dtype=dtype),
            rate=torch.zeros(shape, device=device, dtype=dtype),
            subunit_energy=torch.zeros(
                batch_size, 2, self.cell_count, device=device, dtype=dtype
            ),
        )

    def residual_penalty(self) -> torch.Tensor:
        penalties = [
            getattr(self, name).residual_penalty() for name in self.parameter_names
        ]
        return self._residual_weight * torch.stack(penalties).mean()

    def physiology_prior_penalty(self) -> torch.Tensor:
        type_penalties = [
            getattr(self, name).type_prior_penalty() for name in self.parameter_names
        ]
        return self.residual_penalty() + self._type_prior_weight * torch.stack(
            type_penalties
        ).mean()

    def forward(
        self,
        bipolar: torch.Tensor,
        amacrine: torch.Tensor,
        previous: TypedRGCState,
        spatial_weights: torch.Tensor,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[TypedRGCStepOutput, TypedRGCState]:
        pooled = torch.einsum("uc,bpkc->bpku", spatial_weights, bipolar)
        pooled_amacrine = torch.einsum("uc,bpkc->bpku", spatial_weights, amacrine)
        batch = bipolar.shape[0]
        polarity_index = self.cell_polarities.view(1, 1, 1, -1).expand(
            batch, 1, 2, -1
        )
        selected = pooled.gather(1, polarity_index).squeeze(1)
        selected_amacrine = pooled_amacrine.gather(1, polarity_index).squeeze(1)
        subunit_leak = torch.exp(-self._dt_ms / self.subunit_tau_ms()).view(
            1, 1, -1
        )
        energy = subunit_leak * previous.subunit_energy + (
            1 - subunit_leak
        ) * selected.square()
        adapted = selected / (1 + self.subunit_gain().view(1, 1, -1) * energy)
        mix = self.sustained_mix().view(1, -1)
        drive = mix * adapted[:, 0] + (1 - mix) * adapted[:, 1]
        inhibition = selected_amacrine.mean(dim=1)
        current = drive - self.amacrine_gain().view(1, -1) * inhibition
        membrane_leak = torch.exp(-self._dt_ms / self.membrane_tau_ms()).view(
            1, -1
        )
        generator = membrane_leak * previous.membrane + (
            1 - membrane_leak
        ) * (current - self.adaptation_gain().view(1, -1) * previous.adaptation)
        logits = self._surrogate_slope * (
            generator - self.threshold().view(1, -1)
        )
        probability = torch.sigmoid(logits)
        hard = (logits >= 0).to(logits.dtype)
        state_event = hard.detach() if observed_counts is None else observed_counts
        reset_event = (state_event > 0).to(logits.dtype)
        adaptation_leak = torch.exp(
            -self._dt_ms / self.adaptation_tau_ms()
        ).view(1, -1)
        rate_leak = torch.exp(
            torch.tensor(-self._dt_ms / self._rate_tau_ms, device=logits.device)
        )
        next_state = TypedRGCState(
            membrane=generator * (1 - reset_event),
            adaptation=adaptation_leak * previous.adaptation
            + (1 - adaptation_leak) * state_event,
            rate=rate_leak * previous.rate + (1 - rate_leak) * state_event,
            subunit_energy=energy,
        )
        return (
            TypedRGCStepOutput(
                spike_logits=logits,
                spike_probability=probability,
                hard_spikes=hard,
                filtered_rate=next_state.rate,
                generator_potential=generator,
            ),
            next_state,
        )


__all__ = [
    "TypedRGCError",
    "TypedRGCOutput",
    "TypedRGCPopulation",
    "TypedRGCState",
    "TypedRGCStepOutput",
]
