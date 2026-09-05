from __future__ import annotations

# noqa: SIZE_OK - bounded parameters and their stateful RGC population are one unit.

from dataclasses import dataclass
import math
from typing import Final, Literal, TypeAlias, assert_never

import torch
from torch import nn

from configs.rgc_type_priors import ParameterPrior, RGCTypePriors
from data.rgc_response import CellMetadata
from models.cells.parameter_sharing import (
    ParameterSharingError,
    ParameterSharingMode,
    RGC_PARAMETER_NAMES,
    parameter_sharing_groups,
)


class TypedRGCError(ValueError):
    pass


ReadoutMode: TypeAlias = Literal[
    "v2_direct_logit",
    "v3_mechanism_preserving",
]
READOUT_MODES: Final = frozenset(("v2_direct_logit", "v3_mechanism_preserving"))


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
class RGCMechanismFeatures:
    selected_bipolar: torch.Tensor
    adapted_bipolar: torch.Tensor
    selected_amacrine: torch.Tensor
    subunit_energy: torch.Tensor


@dataclass(frozen=True, slots=True)
class RGCCurrentDrive:
    current: torch.Tensor
    subunit_energy: torch.Tensor
    direct_logit: torch.Tensor | None = None


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
        *,
        use_cell_residuals: bool = True,
        initial_value: float | None = None,
    ) -> None:
        super().__init__()
        means = torch.tensor(
            [
                prior.mean if initial_value is None else initial_value
                for prior in priors
            ],
            dtype=torch.float32,
        )
        lower = torch.tensor([prior.lower for prior in priors], dtype=torch.float32)
        upper = torch.tensor([prior.upper for prior in priors], dtype=torch.float32)
        if not bool(((means >= lower) & (means <= upper)).all()):
            raise TypedRGCError("matched initialization must fit every prior bound")
        fraction = ((means - lower) / (upper - lower)).clamp(1e-5, 1 - 1e-5)
        self.type_base_raw = nn.Parameter(torch.logit(fraction))
        self.register_buffer(
            "prior_type_base_raw",
            self.type_base_raw.detach().clone(),
        )
        if use_cell_residuals:
            self.cell_residual_raw = nn.Parameter(
                torch.zeros(cell_type_indices.numel(), dtype=torch.float32)
            )
        self.register_buffer("cell_type_indices", cell_type_indices.to(torch.long))
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self._residual_scale = residual_scale
        self._use_cell_residuals = use_cell_residuals

    def forward(self) -> torch.Tensor:
        indices = self.cell_type_indices
        raw = self.type_base_raw.index_select(0, indices)
        if self._use_cell_residuals:
            raw = raw + self._residual_scale * torch.tanh(self.cell_residual_raw)
        lower = self.lower.index_select(0, indices)
        upper = self.upper.index_select(0, indices)
        return lower + (upper - lower) * torch.sigmoid(raw)

    def residual_penalty(self) -> torch.Tensor:
        if not self._use_cell_residuals:
            return self.type_base_raw.new_zeros(())
        return self.cell_residual_raw.square().mean()

    def type_prior_penalty(self) -> torch.Tensor:
        return (self.type_base_raw - self.prior_type_base_raw).square().mean()


class TypedRGCPopulation(nn.Module):
    parameter_names = RGC_PARAMETER_NAMES

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
        parameter_sharing_mode: ParameterSharingMode = "type_aware",
        parameter_sharing_seed: int = 0,
        matched_initialization: bool = False,
        enable_response_bias: bool = False,
        enable_synaptic_gain: bool = False,
        enable_direct_readout: bool = False,
        synaptic_gain_min: float = 0.1,
        synaptic_gain_max: float = 4.0,
        synaptic_gain_init: float = 1.0,
        readout_mode: ReadoutMode = "v2_direct_logit",
    ) -> None:
        super().__init__()
        cones = torch.as_tensor(cone_positions_degs, dtype=torch.float32)
        centers = torch.as_tensor(cells.positions_degs, dtype=torch.float32)
        if cones.ndim != 2 or centers.ndim != 2 or cones.shape[1] != 2:
            raise TypedRGCError("Cone and cell positions must have shape [count,2]")
        if support_radius_degs <= 0 or dt_ms <= 0 or readout_rate_tau_ms <= 0:
            raise TypedRGCError("RGC time and support values must be positive")
        if not isinstance(enable_response_bias, bool):
            raise TypedRGCError("enable_response_bias must be a boolean")
        if not isinstance(enable_synaptic_gain, bool):
            raise TypedRGCError("enable_synaptic_gain must be a boolean")
        if not isinstance(enable_direct_readout, bool):
            raise TypedRGCError("enable_direct_readout must be a boolean")
        if not isinstance(readout_mode, str) or readout_mode not in READOUT_MODES:
            raise TypedRGCError("readout_mode is invalid")
        if readout_mode == "v3_mechanism_preserving" and (
            enable_synaptic_gain or enable_direct_readout
        ):
            raise TypedRGCError(
                "V3 forbids duplicate synaptic gain and direct-to-logit readout"
            )
        gain_values = (synaptic_gain_min, synaptic_gain_max, synaptic_gain_init)
        if not all(math.isfinite(value) for value in gain_values):
            raise TypedRGCError("Synaptic gain bounds must be finite")
        if not synaptic_gain_min < synaptic_gain_init < synaptic_gain_max:
            raise TypedRGCError(
                "synaptic_gain_init must lie inside synaptic gain bounds"
            )
        try:
            groups = parameter_sharing_groups(
                cells,
                priors,
                parameter_sharing_mode,
                parameter_sharing_seed,
            )
        except ParameterSharingError as exc:
            raise TypedRGCError(str(exc)) from exc
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
        self._enable_response_bias = enable_response_bias
        self._enable_synaptic_gain = enable_synaptic_gain
        self._enable_direct_readout = enable_direct_readout
        self.readout_mode = readout_mode
        self._synaptic_gain_min = float(synaptic_gain_min)
        self._synaptic_gain_max = float(synaptic_gain_max)
        self.parameter_sharing_mode = groups.mode
        self.matched_initialization = matched_initialization
        self.shuffle_contract = groups.shuffle_contract
        self.observed_type_labels = cells.type_ids
        self.effective_type_labels = groups.effective_type_labels
        self.parameter_group_labels = groups.parameter_group_labels
        for name in self.parameter_names:
            values = tuple(prior.parameter(name) for prior in groups.priors)
            initial_value = None
            if matched_initialization:
                source = tuple(prior.parameter(name) for prior in priors.types)
                count = len(source)
                matched_prior = ParameterPrior(
                    mean=sum(prior.mean for prior in source) / count,
                    lower=sum(prior.lower for prior in source) / count,
                    upper=sum(prior.upper for prior in source) / count,
                )
                values = (matched_prior,) * len(groups.priors)
                initial_value = matched_prior.mean
            setattr(
                self,
                name,
                TypeConditionedParameter(
                    values,
                    groups.group_indices,
                    priors.cell_residual_scale,
                    use_cell_residuals=groups.use_cell_residuals,
                    initial_value=initial_value,
                ),
            )
        if enable_response_bias:
            self.response_bias = nn.Parameter(torch.zeros(self.cell_count))
        if enable_synaptic_gain:
            fraction = (synaptic_gain_init - synaptic_gain_min) / (
                synaptic_gain_max - synaptic_gain_min
            )
            raw_init = torch.logit(torch.tensor(fraction, dtype=torch.float32))
            self.synaptic_gain_raw = nn.Parameter(
                torch.full((self.cell_count,), float(raw_init))
            )
        if enable_direct_readout:
            shape = (2, self.cell_count)
            self.bipolar_readout_gain = nn.Parameter(torch.zeros(shape))
            self.amacrine_readout_gain = nn.Parameter(torch.zeros(shape))
        if readout_mode == "v3_mechanism_preserving":
            fraction = (synaptic_gain_init - synaptic_gain_min) / (
                synaptic_gain_max - synaptic_gain_min
            )
            raw_init = torch.logit(torch.tensor(fraction, dtype=torch.float32))
            legacy_gain = synaptic_gain_min + (
                synaptic_gain_max - synaptic_gain_min
            ) * torch.sigmoid(raw_init)
            mix = self.sustained_mix().detach()
            bipolar_gain = torch.stack(
                (legacy_gain * mix, legacy_gain * (1 - mix))
            )
            amacrine_gain = (
                self.amacrine_gain().detach().unsqueeze(0).expand(2, -1) / 2
            )
            self.bipolar_current_gain_raw = nn.Parameter(
                _softplus_inverse(bipolar_gain)
            )
            self.amacrine_current_gain_raw = nn.Parameter(
                _softplus_inverse(amacrine_gain)
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

    def synaptic_gain(self) -> torch.Tensor:
        if not self._enable_synaptic_gain:
            return self.threshold().new_ones(self.cell_count)
        gain_range = self._synaptic_gain_max - self._synaptic_gain_min
        return self._synaptic_gain_min + gain_range * torch.sigmoid(
            self.synaptic_gain_raw
        )

    def bipolar_current_gain(self) -> torch.Tensor:
        if self.readout_mode != "v3_mechanism_preserving":
            raise TypedRGCError("Bipolar current gains require V3")
        return nn.functional.softplus(self.bipolar_current_gain_raw)

    def amacrine_current_gain(self) -> torch.Tensor:
        if self.readout_mode != "v3_mechanism_preserving":
            raise TypedRGCError("Amacrine current gains require V3")
        return nn.functional.softplus(self.amacrine_current_gain_raw)

    def mechanism_features(
        self,
        bipolar: torch.Tensor,
        amacrine: torch.Tensor,
        previous: TypedRGCState,
        spatial_weights: torch.Tensor,
    ) -> RGCMechanismFeatures:
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
        return RGCMechanismFeatures(selected, adapted, selected_amacrine, energy)

    def forward(
        self,
        bipolar: torch.Tensor,
        amacrine: torch.Tensor,
        previous: TypedRGCState,
        spatial_weights: torch.Tensor,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[TypedRGCStepOutput, TypedRGCState]:
        features = self.mechanism_features(
            bipolar,
            amacrine,
            previous,
            spatial_weights,
        )
        return self.forward_from_mechanism_features(
            features,
            previous,
            observed_counts,
        )

    def forward_from_mechanism_features(
        self,
        features: RGCMechanismFeatures,
        previous: TypedRGCState,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[TypedRGCStepOutput, TypedRGCState]:
        match self.readout_mode:
            case "v2_direct_logit":
                mix = self.sustained_mix().view(1, -1)
                drive = mix * features.adapted_bipolar[:, 0] + (
                    1 - mix
                ) * features.adapted_bipolar[:, 1]
                inhibition = features.selected_amacrine.mean(dim=1)
                if self._enable_synaptic_gain:
                    drive = self.synaptic_gain().view(1, -1) * drive
                current = drive - self.amacrine_gain().view(1, -1) * inhibition
            case "v3_mechanism_preserving":
                current = (
                    self.bipolar_current_gain().unsqueeze(0)
                    * features.adapted_bipolar
                ).sum(dim=1) - (
                    self.amacrine_current_gain().unsqueeze(0)
                    * features.selected_amacrine
                ).sum(dim=1)
            case unreachable:
                assert_never(unreachable)
        direct_logit = None
        if self._enable_direct_readout:
            direct_logit = (
                self.bipolar_readout_gain.unsqueeze(0) * features.selected_bipolar
                + self.amacrine_readout_gain.unsqueeze(0)
                * features.selected_amacrine
            ).sum(dim=1)
        return self.forward_from_current(
            RGCCurrentDrive(current, features.subunit_energy, direct_logit),
            previous,
            observed_counts,
        )

    def forward_from_current(
        self,
        drive: RGCCurrentDrive,
        previous: TypedRGCState,
        observed_counts: torch.Tensor | None = None,
    ) -> tuple[TypedRGCStepOutput, TypedRGCState]:
        membrane_leak = torch.exp(-self._dt_ms / self.membrane_tau_ms()).view(
            1, -1
        )
        generator = membrane_leak * previous.membrane + (
            1 - membrane_leak
        ) * (drive.current - self.adaptation_gain().view(1, -1) * previous.adaptation)
        logits = self.logits_from_generator(generator)
        if drive.direct_logit is not None:
            logits = logits + drive.direct_logit
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
            subunit_energy=drive.subunit_energy,
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

    def logits_from_generator(self, generator: torch.Tensor) -> torch.Tensor:
        logits = self._surrogate_slope * (
            generator - self.threshold().view(*([1] * (generator.ndim - 1)), -1)
        )
        if not self._enable_response_bias:
            return logits
        return logits + self.response_bias.view(*([1] * (generator.ndim - 1)), -1)


def _softplus_inverse(values: torch.Tensor) -> torch.Tensor:
    positive = values.clamp_min(torch.finfo(values.dtype).eps)
    return positive + torch.log(-torch.expm1(-positive))


__all__ = [
    "ParameterSharingMode",
    "ReadoutMode",
    "RGCCurrentDrive",
    "RGCMechanismFeatures",
    "TypedRGCError",
    "TypedRGCOutput",
    "TypedRGCPopulation",
    "TypedRGCState",
    "TypedRGCStepOutput",
]
