from __future__ import annotations

# noqa: SIZE_OK — one cohesive trainer lifecycle; splitting would add stateful mixin ceremony.

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint

from data.cone_response import load_cone_response, validate_formal_stimulus_splits
from data.dataset import apply_log_cone_stats, fit_log_cone_stats
from models.v9_retina import (
    AnonymousRGCOutput,
    TiedLocalDecoder,
    V9RetinaCore,
    V9RetinaState,
    detach_v9_state,
    state_from_tensors,
    state_to_tensors,
)


class V9TrainingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class V9TrainConfig:
    sequence_steps: int = 320
    burn_in_steps: int = 64
    differentiable_steps: int = 256
    context_only_steps: int = 160
    supervised_steps: int = 96
    checkpoint_block_steps: int = 32
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    core_and_bank_lr: float = 2e-4
    decoder_scalar_lr: float = 1e-4
    max_optimizer_steps: int = 1000
    min_optimizer_steps: int = 300
    validation_interval_steps: int = 50
    early_stopping_patience_validations: int = 8
    lr_warmup_fraction: float = 0.05
    energy_bootstrap_fraction: float = 0.10
    energy_budget_ratio: float = 0.90
    rho_energy: float = 1.0
    dual_lr: float = 0.01
    dual_max: float = 10.0
    lambda_wiring: float = 1e-3
    lambda_cross_bank_redundancy: float = 1e-3
    lambda_homeostasis: float = 1e-3
    lambda_unit_residual: float = 1e-5
    homeostasis_rate_min: float = 1e-4

    def __post_init__(self) -> None:
        if self.sequence_steps != self.burn_in_steps + self.differentiable_steps:
            raise V9TrainingError("Sequence length must equal burn-in plus differentiable steps")
        if self.differentiable_steps != self.context_only_steps + self.supervised_steps:
            raise V9TrainingError("Differentiable steps must equal context plus supervised steps")
        if self.differentiable_steps % self.checkpoint_block_steps:
            raise V9TrainingError("Checkpoint block size must divide differentiable steps")
        positive = (
            self.gradient_accumulation_steps,
            self.gradient_clip_norm,
            self.core_and_bank_lr,
            self.decoder_scalar_lr,
            self.max_optimizer_steps,
            self.min_optimizer_steps,
            self.validation_interval_steps,
            self.early_stopping_patience_validations,
            self.rho_energy,
            self.dual_lr,
            self.dual_max,
        )
        if not all(math.isfinite(float(value)) and value > 0 for value in positive):
            raise V9TrainingError("Training scales and step counts must be positive")
        fractions = (
            self.lr_warmup_fraction,
            self.energy_bootstrap_fraction,
            self.energy_budget_ratio,
        )
        if not all(0 < value < 1 for value in fractions):
            raise V9TrainingError("Schedule fractions and budget ratio must lie in (0,1)")
        if self.min_optimizer_steps > self.max_optimizer_steps:
            raise V9TrainingError("Minimum optimizer steps cannot exceed maximum")


@dataclass(slots=True)  # noqa: MUTABLE_OK — online dual/EMA state is intentionally mutable.
class EnergyBudgetState:
    bootstrap_steps: int
    budget_ratio: float
    dual_lr: float
    dual_max: float
    ema_decay: float = 0.95
    ema: float | None = None
    budget: float | None = None
    dual: float = 0.0
    optimizer_steps: int = 0

    def penalty(self, energy: torch.Tensor, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.budget is None:
            zero = energy.new_zeros(())
            return zero, zero
        violation = torch.relu(energy / self.budget - 1.0)
        return self.dual * violation + 0.5 * rho * violation.square(), violation

    def update(self, energy: float) -> None:
        self.optimizer_steps += 1
        self.ema = energy if self.ema is None else self.ema_decay * self.ema + (
            1.0 - self.ema_decay
        ) * energy
        if self.budget is None and self.optimizer_steps >= self.bootstrap_steps:
            self.budget = self.budget_ratio * self.ema
        elif self.budget is not None:
            self.dual = min(
                self.dual_max,
                max(0.0, self.dual + self.dual_lr * (self.ema / self.budget - 1.0)),
            )


@dataclass(frozen=True, slots=True)
class V9Losses:
    total: torch.Tensor
    raw_reconstruction: torch.Tensor
    normalized_reconstruction: torch.Tensor
    energy: torch.Tensor
    energy_penalty: torch.Tensor
    energy_violation: torch.Tensor
    wiring: torch.Tensor
    redundancy: torch.Tensor
    homeostasis: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    metrics: dict[str, float]
    gradient_norm: float
    temporal_gradient_norm: float
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class PreparedData:
    train: tuple[torch.Tensor, ...]
    validation: tuple[torch.Tensor, ...]
    positions_degs: np.ndarray
    dt_ms: float
    eccentricity_deg: float
    normalization_mean: np.ndarray
    normalization_scale: np.ndarray
    manifest: dict[str, Any]


def prepare_data(train_paths: Sequence[Path], validation_paths: Sequence[Path]) -> PreparedData:
    train_exports = tuple(load_cone_response(path) for path in train_paths)
    validation_exports = tuple(load_cone_response(path) for path in validation_paths)
    validate_formal_stimulus_splits(train_exports, validation_exports)
    all_exports = (*train_exports, *validation_exports)
    if not all_exports:
        raise V9TrainingError("At least one HDF5 export is required")
    reference = all_exports[0]
    if any(export.response.shape != reference.response.shape for export in all_exports):
        raise V9TrainingError("All v9 exports must share one [time,cone] shape")
    if reference.response.shape[0] != 320:
        raise V9TrainingError("V9 requires 320-step HDF5 sequences")
    mean, scale = fit_log_cone_stats(train_paths)

    def normalized(export: Any) -> torch.Tensor:
        return torch.from_numpy(
            apply_log_cone_stats(export.response, mean, scale, clip=5.0)
        )

    dt_ms = float(np.median(np.diff(reference.time_axis_seconds)) * 1000.0)
    manifest = {
        "mode": "synthetic_training_noise",
        "train_files": [str(path.resolve()) for path in train_paths],
        "validation_files": [str(path.resolve()) for path in validation_paths],
        "train_source_ids": [export.source_id for export in train_exports],
        "validation_source_ids": [export.source_id for export in validation_exports],
        "source_disjoint": True,
        "schema": {
            "response_shape": list(reference.response.shape),
            "positions_shape": list(reference.positions_degs.shape),
            "cone_types_shape": list(reference.cone_types.shape),
            "time_axis_shape": list(reference.time_axis_seconds.shape),
            "eye_trace_shape": list(reference.eye_trace_degs.shape),
            "dtype": str(reference.response.dtype),
            "dt_ms": dt_ms,
            "eccentricity_deg": reference.eccentricity_deg,
            "stimulus_source_kind": reference.stimulus_source_kind,
            "response_units": reference.units,
            "paired_noisy_clean": False,
        },
        "normalization": "per-cone train-only log mean/std; validation reuses train stats",
    }
    return PreparedData(
        train=tuple(normalized(export) for export in train_exports),
        validation=tuple(normalized(export) for export in validation_exports),
        positions_degs=reference.positions_degs,
        dt_ms=dt_ms,
        eccentricity_deg=reference.eccentricity_deg,
        normalization_mean=mean,
        normalization_scale=scale,
        manifest=manifest,
    )


def augment_clip(
    clean: torch.Tensor,
    generator: torch.Generator,
    *,
    transition_probability: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | bool]]:
    if clean.ndim != 2 or clean.shape[0] != 320:
        raise V9TrainingError("Clean clip must have shape [320,Ncone]")
    log_low = math.log(0.3)
    gain_one = math.exp(log_low + (0.0 - log_low) * torch.rand((), generator=generator).item())
    has_transition = torch.rand((), generator=generator).item() < transition_probability
    envelope = clean.new_full((clean.shape[0],), gain_one)
    gain_two = gain_one
    transition_step = -1
    transition_width = 0
    if has_transition:
        gain_two = math.exp(log_low + (0.0 - log_low) * torch.rand((), generator=generator).item())
        if abs(gain_two - gain_one) < 0.1:
            gain_two = 0.3 if gain_one > 0.65 else 1.0
        if torch.randint(0, 2, (), generator=generator).item() == 0:
            gain_one, gain_two = min(gain_one, gain_two), max(gain_one, gain_two)
        else:
            gain_one, gain_two = max(gain_one, gain_two), min(gain_one, gain_two)
        transition_step = int(torch.randint(112, 245, (), generator=generator).item())
        transition_width = int(torch.randint(4, 9, (), generator=generator).item())
        start = transition_step - transition_width // 2
        end = start + transition_width
        envelope[:start] = gain_one
        phase = torch.linspace(0.0, math.pi, transition_width, dtype=clean.dtype)
        envelope[start:end] = gain_one + (gain_two - gain_one) * (1.0 - torch.cos(phase)) / 2.0
        envelope[end:] = gain_two
    target = clean * envelope[:, None]
    noise_std = 0.10 + 0.15 * torch.rand((), generator=generator).item()
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    noisy = target + noise_std * noise
    return noisy, target, {
        "has_transition": has_transition,
        "gain_before": gain_one,
        "gain_after": gain_two,
        "transition_step": transition_step,
        "transition_width_steps": transition_width,
        "noise_std": noise_std,
    }


class V9Trainer:
    def __init__(
        self,
        core: V9RetinaCore,
        decoder: TiedLocalDecoder,
        config: V9TrainConfig,
        reconstruction_scale: float,
    ) -> None:
        if not math.isfinite(reconstruction_scale) or reconstruction_scale <= 0:
            raise V9TrainingError("Reconstruction scale must be positive and finite")
        self.core = core
        self.decoder = decoder
        self.config = config
        self.reconstruction_scale = reconstruction_scale
        self.optimizer = torch.optim.AdamW(
            (
                {"name": "core", "params": core.parameters(), "lr": config.core_and_bank_lr},
                {"name": "decoder", "params": decoder.parameters(), "lr": config.decoder_scalar_lr},
            ),
            weight_decay=0.0,
        )
        warmup_steps = max(1, round(config.max_optimizer_steps * config.lr_warmup_fraction))

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, config.max_optimizer_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_multiplier)
        self.energy_state = EnergyBudgetState(
            bootstrap_steps=max(1, round(config.max_optimizer_steps * config.energy_bootstrap_fraction)),
            budget_ratio=config.energy_budget_ratio,
            dual_lr=config.dual_lr,
            dual_max=config.dual_max,
        )
        self.optimizer_step = 0
        self.full_bptt = False

    def forward_clip(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        *,
        checkpointed: bool,
        full_bptt: bool = False,
    ) -> tuple[V9Losses, AnonymousRGCOutput, V9RetinaState]:
        if noisy.shape != clean.shape or noisy.ndim != 3:
            raise V9TrainingError("Noisy and clean clips must match [batch,time,cone]")
        if noisy.shape[1] != self.config.sequence_steps:
            raise V9TrainingError("Clip length does not match the v9 sequence configuration")
        state = self.core.initial_state(noisy.shape[0], noisy.device, torch.float32)
        if full_bptt:
            history, state = self.core.forward_sequence(noisy.float(), state)
        else:
            with torch.no_grad():
                _, state = self.core.forward_sequence(
                    noisy[:, : self.config.burn_in_steps].float(), state
                )
            state = detach_v9_state(state)
            region = noisy[:, self.config.burn_in_steps :].float()
            history, state = self._forward_region(region, state, checkpointed)

        prediction = self.decoder(history)
        supervised = self.config.supervised_steps
        raw_reconstruction = F.mse_loss(
            prediction[:, -supervised:], clean[:, -supervised:].float()
        )
        normalized_reconstruction = raw_reconstruction / self.reconstruction_scale
        target_count = clean.shape[-1]
        energy = history.spikes.sum() / (
            history.spikes.shape[0] * history.spikes.shape[1] * target_count
        )
        energy_penalty, energy_violation = self.energy_state.penalty(
            energy, self.config.rho_energy
        )
        rates = history.rates.float()
        centered = rates - rates.mean(dim=(0, 1), keepdim=True)
        rate_variance = centered.square().mean(dim=(0, 1), keepdim=True)
        standardized = centered / rate_variance.clamp_min(1e-12).sqrt()
        redundancy = (standardized[:, :, 0] * standardized[:, :, 1]).mean(
            dim=(0, 1)
        ).square().mean()
        group_rates = rates.mean(dim=(0, 1, 4))
        homeostasis = torch.relu(
            self.config.homeostasis_rate_min - group_rates
        ).square().mean()
        wiring = self.core.rgc.wiring_cost()
        residual = self.core.rgc.residual_cost()
        total = (
            normalized_reconstruction
            + energy_penalty
            + self.config.lambda_wiring * wiring
            + self.config.lambda_cross_bank_redundancy * redundancy
            + self.config.lambda_homeostasis * homeostasis
            + self.config.lambda_unit_residual * residual
        )
        return V9Losses(
            total,
            raw_reconstruction,
            normalized_reconstruction,
            energy,
            energy_penalty,
            energy_violation,
            wiring,
            redundancy,
            homeostasis,
            residual,
        ), history, state

    def _forward_region(
        self,
        region: torch.Tensor,
        state: V9RetinaState,
        checkpointed: bool,
    ) -> tuple[AnonymousRGCOutput, V9RetinaState]:
        if not checkpointed:
            return self.core.forward_sequence(region, state)
        state_tensors = state_to_tensors(state)
        spikes = []
        rates = []
        block_size = self.config.checkpoint_block_steps
        for start in range(0, region.shape[1], block_size):
            block = region[:, start : start + block_size]

            def run_block(
                block_input: torch.Tensor, *flat_state: torch.Tensor
            ) -> tuple[torch.Tensor, ...]:
                block_output, block_state = self.core.forward_sequence(
                    block_input, state_from_tensors(tuple(flat_state))
                )
                return (*state_to_tensors(block_state), block_output.spikes, block_output.rates)

            values = checkpoint(run_block, block, *state_tensors, use_reentrant=False)
            state_tensors = tuple(values[:8])
            spikes.append(values[8])
            rates.append(values[9])
        return AnonymousRGCOutput(
            torch.cat(spikes, dim=1), torch.cat(rates, dim=1)
        ), state_from_tensors(state_tensors)

    def train_optimizer_step(
        self,
        clips: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> OptimizerStepResult:
        if len(clips) != self.config.gradient_accumulation_steps:
            raise V9TrainingError("Optimizer step requires exactly gradient_accumulation_steps clips")
        self.core.train()
        self.decoder.train()
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        metric_rows: list[dict[str, float]] = []
        energies = []
        for noisy, clean in clips:
            losses, history, _ = self.forward_clip(
                noisy,
                clean,
                checkpointed=not self.full_bptt,
                full_bptt=self.full_bptt,
            )
            if not torch.isfinite(losses.total):
                raise V9TrainingError("Training produced a non-finite loss")
            (losses.total / len(clips)).backward()
            energies.append(float(losses.energy.detach()))
            metric_rows.append(_loss_metrics(losses, history))
        metrics = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }
        temporal_gradient_norm = _gradient_norm(
            parameter
            for name, parameter in self.core.named_parameters()
            if any(token in name for token in ("tau", "gain", "g_"))
        )
        gradient_norm = float(
            clip_grad_norm_(
                (*self.core.parameters(), *self.decoder.parameters()),
                self.config.gradient_clip_norm,
            )
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer_step += 1
        self.energy_state.update(sum(energies) / len(energies))
        metrics.update(
            {
                "energy_budget": self.energy_state.budget or 0.0,
                "energy_ema": self.energy_state.ema or 0.0,
                "dual": self.energy_state.dual,
                "lr_core": float(self.optimizer.param_groups[0]["lr"]),
                "lr_decoder": float(self.optimizer.param_groups[1]["lr"]),
            }
        )
        return OptimizerStepResult(
            metrics,
            gradient_norm,
            temporal_gradient_norm,
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        )

    @torch.no_grad()
    def evaluate(self, clips: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
        self.core.eval()
        self.decoder.eval()
        rows = []
        group_rates = []
        active = []
        for noisy, clean in clips:
            losses, history, _ = self.forward_clip(noisy, clean, checkpointed=False)
            rows.append(_loss_metrics(losses, history))
            group_rates.append(history.rates.mean(dim=(0, 1, 4)).cpu())
            active.append((history.rates.amax(dim=1) > 0).float().mean(dim=(0, 3)).cpu())
        result = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        result["bank_polarity_rates"] = torch.stack(group_rates).mean(dim=0).tolist()
        result["active_unit_fraction"] = torch.stack(active).mean(dim=0).tolist()
        result["energy_budget"] = self.energy_state.budget
        result["energy_violation"] = (
            max(0.0, result["energy"] / self.energy_state.budget - 1.0)
            if self.energy_state.budget
            else None
        )
        return result

    def checkpoint_payload(self, augmentation_generator: torch.Generator) -> dict[str, Any]:
        return {
            "schema": "v9_emergent_anonymous_rgc_dynamic_rf",
            "optimizer_step": self.optimizer_step,
            "full_bptt": self.full_bptt,
            "core": self.core.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "amp_scaler": None,
            "energy_state": asdict(self.energy_state),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "augmentation_rng_state": augmentation_generator.get_state(),
        }

    def restore(self, payload: dict[str, Any], augmentation_generator: torch.Generator) -> None:
        if payload.get("schema") != "v9_emergent_anonymous_rgc_dynamic_rf":
            raise V9TrainingError("Checkpoint schema mismatch")
        self.core.load_state_dict(payload["core"])
        self.decoder.load_state_dict(payload["decoder"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.optimizer_step = int(payload["optimizer_step"])
        self.full_bptt = bool(payload.get("full_bptt", False))
        self.energy_state = EnergyBudgetState(**payload["energy_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_state"]]
            )
        augmentation_generator.set_state(payload["augmentation_rng_state"].cpu())


def temporal_gradient_audit(
    trainer: V9Trainer,
    noisy: torch.Tensor,
    clean: torch.Tensor,
) -> dict[str, float | bool]:
    gradients = []
    for full_bptt in (False, True):
        trainer.optimizer.zero_grad(set_to_none=True)
        losses, _, _ = trainer.forward_clip(
            noisy, clean, checkpointed=False, full_bptt=full_bptt
        )
        losses.normalized_reconstruction.backward()
        pieces = []
        for name, parameter in trainer.core.named_parameters():
            if any(token in name for token in ("tau", "gain", "g_")):
                pieces.append(
                    torch.zeros_like(parameter).flatten()
                    if parameter.grad is None
                    else parameter.grad.detach().flatten()
                )
        gradients.append(torch.cat(pieces))
    g_256, g_320 = gradients
    norm_256 = float(g_256.norm())
    norm_320 = float(g_320.norm())
    cosine = float(F.cosine_similarity(g_256, g_320, dim=0))
    ratio = norm_256 / max(norm_320, torch.finfo(g_320.dtype).eps)
    trainer.optimizer.zero_grad(set_to_none=True)
    return {
        "cosine": cosine,
        "norm_ratio": ratio,
        "norm_256": norm_256,
        "norm_320": norm_320,
        "passed": cosine >= 0.95 and 0.8 <= ratio <= 1.2,
    }


def fixed_validation_clips(
    clips: Sequence[torch.Tensor], seed: int, device: torch.device
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    result = []
    for index, clip in enumerate(clips):
        generator = torch.Generator().manual_seed(seed + index)
        noisy, clean, _ = augment_clip(clip, generator)
        result.append((noisy[None].to(device), clean[None].to(device)))
    return tuple(result)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_metrics(losses: V9Losses, history: AnonymousRGCOutput) -> dict[str, float]:
    return {
        "loss_total": float(losses.total.detach()),
        "raw_reconstruction": float(losses.raw_reconstruction.detach()),
        "normalized_reconstruction": float(losses.normalized_reconstruction.detach()),
        "energy": float(losses.energy.detach()),
        "energy_penalty": float(losses.energy_penalty.detach()),
        "energy_violation": float(losses.energy_violation.detach()),
        "wiring": float(losses.wiring.detach()),
        "redundancy": float(losses.redundancy.detach()),
        "homeostasis": float(losses.homeostasis.detach()),
        "residual": float(losses.residual.detach()),
        "bank_a_rate": float(history.rates[:, :, 0].mean().detach()),
        "bank_b_rate": float(history.rates[:, :, 1].mean().detach()),
    }


def _gradient_norm(parameters: Any) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0
