from __future__ import annotations

# noqa: SIZE_OK — one cohesive model graph; splitting would obscure tied encoder/decoder state.

import math
import weakref
from dataclasses import dataclass

import torch
from torch import nn

from configs.physiology_profiles import human_macaque_v1
from data.geometry import PositionArray
from models.cells.amacrine import LocalAmacrineLayer
from models.cells.bipolar import BipolarLayer, BipolarState
from models.cells.horizontal import H1HorizontalNetwork


class V9ModelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AnonymousBankConfig:
    dt_ms: float
    support_radius_degs: float
    sigma_min_degs: float
    sigma_max_degs: float
    sigma_initial_degs: float
    seed: int = 19
    symmetry_noise_fraction: float = 0.01
    surrogate_slope: float = 5.0

    def __post_init__(self) -> None:
        values = (
            self.dt_ms,
            self.support_radius_degs,
            self.sigma_min_degs,
            self.sigma_max_degs,
            self.sigma_initial_degs,
            self.symmetry_noise_fraction,
            self.surrogate_slope,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise V9ModelError("Anonymous-bank configuration must be positive and finite")
        if not self.sigma_min_degs < self.sigma_initial_degs < self.sigma_max_degs:
            raise V9ModelError("Initial sigma must lie strictly inside its bounds")


@dataclass(frozen=True, slots=True)
class AnonymousRGCState:
    membrane: torch.Tensor
    adaptation: torch.Tensor
    rate: torch.Tensor
    subunit_energy: torch.Tensor


@dataclass(frozen=True, slots=True)
class AnonymousRGCOutput:
    spikes: torch.Tensor
    rates: torch.Tensor


@dataclass(frozen=True, slots=True)
class V9RetinaState:
    h1: torch.Tensor
    bipolar: BipolarState
    amacrine: torch.Tensor
    rgc: AnonymousRGCState


class AnonymousRGCBanks(nn.Module):
    _PARAMETERS = {
        "mix": (0.0, 1.0, 0.5),
        "membrane_tau_ms": (5.0, 80.0, 20.0),
        "adaptation_tau_ms": (20.0, 250.0, 80.0),
        "adaptation_gain": (0.0, 1.0, 0.10),
        "rate_tau_ms": (5.0, 250.0, 50.0),
        "amacrine_gain": (0.0, 0.30, 0.02),
        "threshold": (0.05, 1.0, 0.20),
        "subunit_tau_ms": (10.0, 200.0, 50.0),
        "subunit_gain": (0.0, 3.0, 0.50),
    }

    def __init__(
        self,
        cone_positions_degs: PositionArray,
        config: AnonymousBankConfig,
    ) -> None:
        super().__init__()
        positions = torch.as_tensor(cone_positions_degs, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
            raise V9ModelError("cone_positions_degs must have shape [Ncone,2]")
        if not torch.isfinite(positions).all():
            raise V9ModelError("cone_positions_degs must be finite")
        distances = torch.cdist(positions, positions)
        support = distances <= config.support_radius_degs
        support.fill_diagonal_(True)
        indices = support.nonzero(as_tuple=False).T.contiguous()
        distance_sq = distances[indices[0], indices[1]].square()
        scale = max(config.support_radius_degs, torch.finfo(torch.float32).eps)

        self.register_buffer("unit_centers_degs", positions)
        self.register_buffer("support_indices", indices)
        self.register_buffer("normalized_distance_sq", distance_sq / (scale * scale))
        self.register_buffer("distance_sq_degs", distance_sq)
        self.raw_spatial_residual = nn.Parameter(
            self._noise((2, indices.shape[1]), config, scale=1.0)
        )
        self.raw_sigma = nn.Parameter(
            self._raw_pair(
                config.sigma_initial_degs,
                config.sigma_min_degs,
                config.sigma_max_degs,
                config,
                salt=1,
            )
        )
        self.raw_bank_parameters = nn.ParameterDict(
            {
                name: nn.Parameter(self._raw_pair(initial, lower, upper, config, salt=index + 2))
                for index, (name, (lower, upper, initial)) in enumerate(
                    self._PARAMETERS.items()
                )
            }
        )
        self._sigma_bounds = (config.sigma_min_degs, config.sigma_max_degs)
        self._dt_ms = config.dt_ms
        self._surrogate_slope = config.surrogate_slope
        self._unit_count = positions.shape[0]

    @staticmethod
    def _noise(
        shape: tuple[int, ...],
        config: AnonymousBankConfig,
        *,
        scale: float,
        salt: int = 0,
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(config.seed + 104729 * salt)
        return torch.randn(shape, generator=generator) * (
            config.symmetry_noise_fraction * max(abs(scale), 1.0)
        )

    @classmethod
    def _raw_pair(
        cls,
        initial: float,
        lower: float,
        upper: float,
        config: AnonymousBankConfig,
        *,
        salt: int,
    ) -> torch.Tensor:
        fraction = (initial - lower) / (upper - lower)
        raw = torch.logit(torch.tensor(fraction, dtype=torch.float32))
        return raw.expand(2).clone() + cls._noise(
            (2,), config, scale=float(raw), salt=salt
        )

    def bounded(self, name: str) -> torch.Tensor:
        try:
            lower, upper, _ = self._PARAMETERS[name]
            raw = self.raw_bank_parameters[name]
        except KeyError as exc:
            raise V9ModelError(f"Unknown bank parameter: {name}") from exc
        return lower + (upper - lower) * torch.sigmoid(raw)

    @property
    def sigma_degs(self) -> torch.Tensor:
        lower, upper = self._sigma_bounds
        return lower + (upper - lower) * torch.sigmoid(self.raw_sigma)

    @property
    def spatial_pools(self) -> tuple[torch.Tensor, torch.Tensor]:
        rows, _ = self.support_indices
        pools = []
        for bank in range(2):
            logits = self.raw_spatial_residual[bank] - self.distance_sq_degs / (
                2.0 * self.sigma_degs[bank].square()
            )
            sparse = torch.sparse_coo_tensor(
                self.support_indices,
                logits,
                (self._unit_count, self._unit_count),
                device=logits.device,
                dtype=logits.dtype,
            ).coalesce()
            pools.append(torch.sparse.softmax(sparse, dim=1).coalesce())
        return pools[0], pools[1]

    def decoder_pool(self, bank: int) -> torch.Tensor:
        pool = self.spatial_pools[bank].coalesce()
        source_rows = pool.indices()[0]
        target_rows = pool.indices()[1]
        values = pool.values()
        row_sums = values.new_zeros(self._unit_count).scatter_add(
            0, target_rows, values
        )
        normalized = values / row_sums[target_rows].clamp_min(
            torch.finfo(values.dtype).eps
        )
        return torch.sparse_coo_tensor(
            torch.stack((target_rows, source_rows)),
            normalized,
            (self._unit_count, self._unit_count),
            device=values.device,
            dtype=values.dtype,
        ).coalesce()

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> AnonymousRGCState:
        shape = (batch_size, 2, 2, self._unit_count)
        return AnonymousRGCState(
            membrane=torch.zeros(shape, device=device, dtype=dtype),
            adaptation=torch.zeros(shape, device=device, dtype=dtype),
            rate=torch.zeros(shape, device=device, dtype=dtype),
            subunit_energy=torch.zeros(
                batch_size,
                2,
                2,
                2,
                self._unit_count,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(
        self,
        bipolar_output: torch.Tensor,
        amacrine_output: torch.Tensor,
        previous: AnonymousRGCState | None = None,
    ) -> tuple[AnonymousRGCOutput, AnonymousRGCState]:
        expected = (bipolar_output.shape[0], 2, 2, self._unit_count)
        if bipolar_output.shape != expected or amacrine_output.shape != expected:
            raise V9ModelError("RGC inputs must have shape [batch,2,2,Ncone]")
        if previous is None:
            previous = self.initial_state(
                bipolar_output.shape[0], bipolar_output.device, bipolar_output.dtype
            )
        subunit_leak = torch.exp(-self._dt_ms / self.bounded("subunit_tau_ms"))
        energy = (
            subunit_leak[None, :, None, None, None] * previous.subunit_energy
            + (1.0 - subunit_leak[None, :, None, None, None])
            * bipolar_output[:, None].square()
        )
        adapted = bipolar_output[:, None] / (
            1.0
            + self.bounded("subunit_gain")[None, :, None, None, None] * energy
        )
        mix = self.bounded("mix")
        kinetic_weights = torch.stack((mix, 1.0 - mix), dim=1)
        bipolar_drive = torch.einsum("nk,bnpkc->bnpc", kinetic_weights, adapted)
        amacrine_drive = torch.einsum(
            "nk,bpkc->bnpc", kinetic_weights, amacrine_output
        )
        pools = self.spatial_pools
        currents = []
        for bank, pool in enumerate(pools):
            excitatory = _sparse_pool(pool, bipolar_drive[:, bank])
            inhibitory = _sparse_pool(pool, amacrine_drive[:, bank])
            currents.append(
                excitatory
                - self.bounded("amacrine_gain")[bank] * inhibitory
            )
        current = torch.stack(currents, dim=1)

        membrane_tau = self.bounded("membrane_tau_ms")
        membrane_leak = torch.exp(-self._dt_ms / membrane_tau)[None, :, None, None]
        pre_reset = membrane_leak * previous.membrane + (
            1.0 - membrane_leak
        ) * (current - previous.adaptation)
        threshold = self.bounded("threshold")[None, :, None, None]
        hard = (pre_reset >= threshold).to(pre_reset.dtype)
        soft = torch.sigmoid(self._surrogate_slope * (pre_reset - threshold))
        spikes = hard + (soft - soft.detach())
        membrane = pre_reset * (1.0 - spikes.detach())

        adaptation_leak = torch.exp(
            -self._dt_ms / self.bounded("adaptation_tau_ms")
        )[None, :, None, None]
        adaptation = adaptation_leak * previous.adaptation + (
            1.0 - adaptation_leak
        ) * self.bounded("adaptation_gain")[None, :, None, None] * spikes
        rate_leak = torch.exp(-self._dt_ms / self.bounded("rate_tau_ms"))[
            None, :, None, None
        ]
        rate = rate_leak * previous.rate + (1.0 - rate_leak) * spikes
        state = AnonymousRGCState(membrane, adaptation, rate, energy)
        return AnonymousRGCOutput(spikes, rate), state

    def wiring_cost(self) -> torch.Tensor:
        costs = []
        for pool in self.spatial_pools:
            costs.append((pool.values() * self.normalized_distance_sq).sum() / self._unit_count)
        return torch.stack(costs).mean()

    def residual_cost(self) -> torch.Tensor:
        return self.raw_spatial_residual.square().mean()

    def effective_radius_degs(self) -> torch.Tensor:
        radii = []
        for pool in self.spatial_pools:
            radii.append(
                torch.sqrt(
                    (pool.values() * self.distance_sq_degs).sum()
                    / self._unit_count
                )
            )
        return torch.stack(radii)


class V9RetinaCore(nn.Module):
    def __init__(
        self,
        h1: H1HorizontalNetwork,
        bipolar: BipolarLayer,
        amacrine: LocalAmacrineLayer,
        rgc: AnonymousRGCBanks,
    ) -> None:
        super().__init__()
        self.h1 = h1
        self.bipolar = bipolar
        self.amacrine = amacrine
        self.rgc = rgc

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> V9RetinaState:
        return V9RetinaState(
            self.h1.initial_state(batch_size, device, dtype),
            self.bipolar.initial_state(batch_size, device, dtype),
            self.amacrine.initial_state(batch_size, device, dtype),
            self.rgc.initial_state(batch_size, device, dtype),
        )

    def step(
        self, cone_t: torch.Tensor, state: V9RetinaState
    ) -> tuple[AnonymousRGCOutput, V9RetinaState]:
        cone_mod, h1 = self.h1(cone_t, state.h1)
        bipolar = self.bipolar(cone_mod, state.bipolar, amacrine_prev=state.amacrine)
        amacrine = self.amacrine(bipolar.output, state.amacrine)
        output, rgc = self.rgc(bipolar.output, amacrine, state.rgc)
        return output, V9RetinaState(h1, bipolar, amacrine, rgc)

    def forward_sequence(
        self,
        sequence: torch.Tensor,
        state: V9RetinaState | None = None,
    ) -> tuple[AnonymousRGCOutput, V9RetinaState]:
        if sequence.ndim != 3 or sequence.shape[1] < 1:
            raise V9ModelError("sequence must have shape [batch,time,Ncone]")
        if state is None:
            state = self.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
        spikes = []
        rates = []
        for cone_t in sequence.unbind(dim=1):
            output, state = self.step(cone_t, state)
            spikes.append(output.spikes)
            rates.append(output.rates)
        return AnonymousRGCOutput(
            torch.stack(spikes, dim=1),
            torch.stack(rates, dim=1),
        ), state


class TiedLocalDecoder(nn.Module):
    def __init__(self, encoder: AnonymousRGCBanks, magnitude_max: float = 5.0) -> None:
        super().__init__()
        if not math.isfinite(magnitude_max) or magnitude_max <= 0:
            raise V9ModelError("magnitude_max must be positive and finite")
        self._encoder_ref = weakref.ref(encoder)
        initial = torch.full((2, 2), 0.1 / magnitude_max)
        self.raw_magnitude = nn.Parameter(torch.logit(initial))
        self.bias = nn.Parameter(torch.zeros(()))
        self._magnitude_max = magnitude_max

    @property
    def magnitude(self) -> torch.Tensor:
        return self._magnitude_max * torch.sigmoid(self.raw_magnitude)

    def forward(self, output: AnonymousRGCOutput) -> torch.Tensor:
        rates = output.rates
        if rates.ndim not in {4, 5} or rates.shape[-3:] != (
            2,
            2,
            rates.shape[-1],
        ):
            raise V9ModelError("RGC rates must end with [bank,polarity,unit]")
        encoder = self._encoder_ref()
        if encoder is None:
            raise V9ModelError("Tied decoder encoder is unavailable")
        if rates.shape[-1] != encoder.unit_centers_degs.shape[0]:
            raise V9ModelError("RGC rates and tied encoder unit count differ")
        decoded = []
        for bank in range(2):
            pool = encoder.decoder_pool(bank)
            source = rates[..., bank, :, :]
            flat = source.reshape(-1, source.shape[-1])
            decoded.append(
                torch.sparse.mm(pool, flat.T).T.reshape(
                    *source.shape[:-1], source.shape[-1]
                )
            )
        projected = torch.stack(decoded, dim=-3)
        signs = projected.new_tensor((1.0, -1.0))
        return torch.einsum(
            "...bpc,bp,p->...c", projected, self.magnitude, signs
        ) + self.bias


def build_v9_model(
    cone_positions_degs: PositionArray,
    *,
    dt_ms: float,
    eccentricity_deg: float,
    seed: int,
) -> tuple[V9RetinaCore, TiedLocalDecoder]:
    positions = torch.as_tensor(cone_positions_degs, dtype=torch.float32)
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    spacing = float(distances.min(dim=1).values.median())
    profile = human_macaque_v1(
        dt_ms=dt_ms,
        cone_spacing_deg=spacing,
        eccentricity_deg=eccentricity_deg,
    )
    support_radius = 3.60 * spacing
    bank_config = AnonymousBankConfig(
        dt_ms=dt_ms,
        support_radius_degs=support_radius,
        sigma_min_degs=max(0.25 * spacing, torch.finfo(torch.float32).eps),
        sigma_max_degs=support_radius,
        sigma_initial_degs=support_radius / 3.0,
        seed=seed,
    )
    rgc = AnonymousRGCBanks(cone_positions_degs, bank_config)
    core = V9RetinaCore(
        H1HorizontalNetwork(positions, profile.h1),
        BipolarLayer(positions, profile.bipolar),
        LocalAmacrineLayer(positions, profile.amacrine),
        rgc,
    )
    return core, TiedLocalDecoder(rgc, profile.decoder.current_weight_max)


def detach_v9_state(state: V9RetinaState) -> V9RetinaState:
    return V9RetinaState(
        state.h1.detach(),
        BipolarState(
            state.bipolar.output.detach(),
            state.bipolar.transient_baseline.detach(),
        ),
        state.amacrine.detach(),
        AnonymousRGCState(
            state.rgc.membrane.detach(),
            state.rgc.adaptation.detach(),
            state.rgc.rate.detach(),
            state.rgc.subunit_energy.detach(),
        ),
    )


def state_to_tensors(state: V9RetinaState) -> tuple[torch.Tensor, ...]:
    return (
        state.h1,
        state.bipolar.output,
        state.bipolar.transient_baseline,
        state.amacrine,
        state.rgc.membrane,
        state.rgc.adaptation,
        state.rgc.rate,
        state.rgc.subunit_energy,
    )


def state_from_tensors(tensors: tuple[torch.Tensor, ...]) -> V9RetinaState:
    if len(tensors) != 8:
        raise V9ModelError("V9 recurrent state requires eight tensors")
    return V9RetinaState(
        tensors[0],
        BipolarState(tensors[1], tensors[2]),
        tensors[3],
        AnonymousRGCState(tensors[4], tensors[5], tensors[6], tensors[7]),
    )


def _sparse_pool(weights: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    flat = source.reshape(-1, source.shape[-1])
    return torch.sparse.mm(weights, flat.T).T.reshape(
        *source.shape[:-1], weights.shape[0]
    )
